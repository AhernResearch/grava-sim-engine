from __future__ import annotations

from typing import List

import numpy as np

from ..enums import MultiMetricIndex, WeightedMetricIndex
from ..geometry import coords_to_polygons, state_to_coords
from ..types import SceneContext, ScoringResult
from . import collision_metrics, comfort_metrics, reference, road_metrics, scene_geometry
from . import trajectory as trajectory_ops
from .base import ScorerBase


class PDMSScorer(ScorerBase):
    """NAVSIM v1 PDMS: (NC × DAC) × (5 EP + 5 TTC + 2 HC) / 12."""

    def score(self, waypoints_xy: np.ndarray, scene: SceneContext) -> ScoringResult:
        return self.score_batch(waypoints_xy[None, ...], scene)[0]

    def score_from_controls(
        self, control_signals: np.ndarray, scene: SceneContext
    ) -> ScoringResult:
        return self.score_batch_from_controls(control_signals[None, ...], scene)[0]

    def score_batch(self, trajectories_xy: np.ndarray, scene: SceneContext) -> List[ScoringResult]:
        batch_waypoints = trajectory_ops.coerce_trajectories(trajectories_xy)
        proposals = trajectory_ops.build_proposals(batch_waypoints, scene)
        simulated_states = self._simulator.simulate_proposals(
            ego_state=scene.ego_state, proposals=proposals, observation=scene.observation
        )
        return self._score_from_simulated(simulated_states, scene, len(batch_waypoints))

    def score_batch_from_controls(
        self, control_signals: np.ndarray, scene: SceneContext
    ) -> List[ScoringResult]:
        """Score from direct control signals, bypassing LQR controller."""
        control_signals = np.asarray(control_signals, dtype=np.float64)
        if control_signals.ndim == 2:
            control_signals = control_signals[None, ...]
        if control_signals.ndim != 3 or control_signals.shape[-1] != 2:
            raise ValueError(f"control_signals must be (B, T, 2), got {control_signals.shape}")
        simulated_states = self._simulator.simulate_from_controls(
            ego_state=scene.ego_state,
            control_signals=control_signals,
            observation=scene.observation,
        )
        return self._score_from_simulated(simulated_states, scene, len(control_signals))

    def _score_from_simulated(
        self, simulated_states: np.ndarray, scene: SceneContext, batch_size: int
    ) -> List[ScoringResult]:
        ego_coords = state_to_coords(simulated_states, self._vehicle)
        ego_polygons = coords_to_polygons(ego_coords)
        ego_areas = scene_geometry.calculate_ego_areas(ego_coords, scene)
        multi_metrics = np.ones((batch_size, len(MultiMetricIndex)), dtype=np.float64)
        weighted_metrics = np.ones((batch_size, len(WeightedMetricIndex)), dtype=np.float64)
        multi_metrics[:, MultiMetricIndex.NO_COLLISION] = collision_metrics.no_at_fault_collision(
            simulated_states, ego_polygons, ego_areas, scene
        )
        multi_metrics[:, MultiMetricIndex.DRIVABLE_AREA] = road_metrics.drivable_area_compliance(
            ego_areas
        )
        multi_metrics[:, MultiMetricIndex.DRIVING_DIRECTION] = (
            road_metrics.driving_direction_compliance(ego_coords, ego_areas, scene)
        )
        multi_metrics[:, MultiMetricIndex.TRAFFIC_LIGHT] = road_metrics.traffic_light_compliance(
            ego_polygons, scene
        )
        progress_raw = scene_geometry.progress(ego_coords, scene)
        pdm_masked_progress = reference.pdm_masked_progress(
            scene, simulator=self._simulator, vehicle=self._vehicle
        )
        weighted_metrics[:, WeightedMetricIndex.PROGRESS] = reference.normalize_progress(
            progress_raw,
            multi_metrics[:, MultiMetricIndex.NO_COLLISION],
            multi_metrics[:, MultiMetricIndex.DRIVABLE_AREA],
            pdm_masked_progress=pdm_masked_progress,
        )
        weighted_metrics[:, WeightedMetricIndex.TTC] = collision_metrics.time_to_collision(
            simulated_states, ego_coords, ego_areas, scene
        )
        weighted_metrics[:, WeightedMetricIndex.LANE_KEEPING] = road_metrics.lane_keeping(
            ego_coords, ego_areas, scene
        )
        weighted_metrics[:, WeightedMetricIndex.COMFORT] = comfort_metrics.history_comfort(
            simulated_states, scene, use_past_states=False
        )
        scores = self._aggregate_v1(multi_metrics, weighted_metrics)
        results: List[ScoringResult] = []
        for proposal_idx, score in enumerate(scores):
            results.append(
                ScoringResult(
                    pdm_score=float(score),
                    no_at_fault_collisions=float(
                        multi_metrics[proposal_idx, MultiMetricIndex.NO_COLLISION]
                    ),
                    drivable_area_compliance=float(
                        multi_metrics[proposal_idx, MultiMetricIndex.DRIVABLE_AREA]
                    ),
                    driving_direction_compliance=float(
                        multi_metrics[proposal_idx, MultiMetricIndex.DRIVING_DIRECTION]
                    ),
                    traffic_light_compliance=float(
                        multi_metrics[proposal_idx, MultiMetricIndex.TRAFFIC_LIGHT]
                    ),
                    ego_progress=float(
                        weighted_metrics[proposal_idx, WeightedMetricIndex.PROGRESS]
                    ),
                    time_to_collision=float(
                        weighted_metrics[proposal_idx, WeightedMetricIndex.TTC]
                    ),
                    lane_keeping=float(
                        weighted_metrics[proposal_idx, WeightedMetricIndex.LANE_KEEPING]
                    ),
                    history_comfort=float(
                        weighted_metrics[proposal_idx, WeightedMetricIndex.COMFORT]
                    ),
                )
            )
        return results

    @staticmethod
    def _aggregate_v1(multi_metrics: np.ndarray, weighted_metrics: np.ndarray) -> np.ndarray:
        """V1 aggregation: (NC * DAC) * (5*EP + 5*TTC + 2*HC) / 12"""
        multiplicative_scores = multi_metrics[
            :, [MultiMetricIndex.NO_COLLISION, MultiMetricIndex.DRIVABLE_AREA]
        ].prod(axis=1)
        weights = np.array([5.0, 5.0, 0.0, 2.0], dtype=np.float64)
        weighted_scores = (weighted_metrics * weights[None, :]).sum(axis=1) / weights.sum()
        return multiplicative_scores * weighted_scores
