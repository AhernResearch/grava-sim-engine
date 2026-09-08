from __future__ import annotations

from typing import List

import numpy as np

from ..enums import MultiMetricIndex, WeightedMetricIndex
from ..geometry import state_to_coords, coords_to_polygons
from ..types import SceneContext, ScoringResult
from .base import ScorerBase


class PDMScorerV1(ScorerBase):
    """NavSim v1 PDMS scorer, strictly aligned with recogdrive/navsim.

    Key differences from V2:
    - NC uses observation-based collision type (treats all objects as agents)
    - No TLC, no LK metrics
    - Comfort uses simulated states only (no ego_past_states)
    - Progress: batch max normalization on masked_progress (raw * multi)
    - No human penalty filter
    - Aggregation: (NC * DAC) * (5*EP + 5*TTC + 2*HC) / 12
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._progress_distance_threshold = 5.0

    def score(self, waypoints_xy: np.ndarray, scene: SceneContext) -> ScoringResult:
        return self.score_batch(waypoints_xy[None, ...], scene)[0]

    def score_from_controls(
        self,
        control_signals: np.ndarray,
        scene: SceneContext,
    ) -> ScoringResult:
        return self.score_batch_from_controls(control_signals[None, ...], scene)[0]

    def score_batch(
        self,
        trajectories_xy: np.ndarray,
        scene: SceneContext,
        *,
        include_ego: bool = False,
        independent_progress_fallback: bool = False,
    ) -> List[ScoringResult]:
        batch_waypoints = self._coerce_trajectories(trajectories_xy)
        proposals = self._build_proposals(batch_waypoints, scene, include_ego=include_ego)
        simulated_states = self._simulator.simulate_proposals(
            ego_state=scene.ego_state, proposals=proposals, observation=scene.observation,
        )
        return self._score_from_simulated(
            simulated_states,
            scene,
            len(batch_waypoints),
            independent_progress_fallback=independent_progress_fallback,
        )

    def score_batch_from_controls(
        self,
        control_signals: np.ndarray,
        scene: SceneContext,
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
        self,
        simulated_states: np.ndarray,
        scene: SceneContext,
        batch_size: int,
        *,
        independent_progress_fallback: bool = False,
    ) -> List[ScoringResult]:
        ego_coords = state_to_coords(simulated_states, self._vehicle)
        ego_polygons = coords_to_polygons(ego_coords)
        ego_areas = self._calculate_ego_areas(ego_coords, scene)

        # V1: only NC * DAC (2 multiplicative metrics)
        multi_metrics = np.ones((batch_size, len(MultiMetricIndex)), dtype=np.float64)
        weighted_metrics = np.ones((batch_size, len(WeightedMetricIndex)), dtype=np.float64)

        # NC: use observation-based collision types (recogride behavior)
        multi_metrics[:, MultiMetricIndex.NO_COLLISION] = self._no_at_fault_collision(
            simulated_states, ego_polygons, ego_areas, scene, use_observation_types=True,
        )
        multi_metrics[:, MultiMetricIndex.DRIVABLE_AREA] = self._drivable_area_compliance(ego_areas)
        # DDC computed but NOT used in aggregation (weight=0)
        multi_metrics[:, MultiMetricIndex.DRIVING_DIRECTION] = self._driving_direction_compliance(
            ego_coords, ego_areas, scene,
        )
        # TLC computed but NOT used in aggregation
        multi_metrics[:, MultiMetricIndex.TRAFFIC_LIGHT] = self._traffic_light_compliance(
            ego_polygons, scene,
        )

        progress_raw = self._progress(ego_coords, scene)
        pdm_masked_progress = self._resolve_pdm_masked_progress(scene)

        weighted_metrics[:, WeightedMetricIndex.PROGRESS] = self._normalize_progress_v1(
            progress_raw,
            multi_metrics,
            pdm_masked_progress=pdm_masked_progress,
            independent_fallback=independent_progress_fallback,
        )
        weighted_metrics[:, WeightedMetricIndex.TTC] = self._time_to_collision(
            simulated_states, ego_coords, ego_areas, scene,
        )
        weighted_metrics[:, WeightedMetricIndex.LANE_KEEPING] = self._lane_keeping(
            ego_coords, ego_areas, scene,
        )
        # V1 comfort: only simulated states, no past_states
        weighted_metrics[:, WeightedMetricIndex.COMFORT] = self._history_comfort(
            simulated_states, scene, use_past_states=False,
        )

        scores = self._aggregate_v1(multi_metrics, weighted_metrics)

        results: List[ScoringResult] = []
        for proposal_idx, score in enumerate(scores):
            results.append(ScoringResult(
                scoring_version="v1",
                pdm_score=float(score),
                no_at_fault_collisions=float(multi_metrics[proposal_idx, MultiMetricIndex.NO_COLLISION]),
                drivable_area_compliance=float(multi_metrics[proposal_idx, MultiMetricIndex.DRIVABLE_AREA]),
                driving_direction_compliance=float(multi_metrics[proposal_idx, MultiMetricIndex.DRIVING_DIRECTION]),
                traffic_light_compliance=float(multi_metrics[proposal_idx, MultiMetricIndex.TRAFFIC_LIGHT]),
                ego_progress=float(weighted_metrics[proposal_idx, WeightedMetricIndex.PROGRESS]),
                time_to_collision=float(weighted_metrics[proposal_idx, WeightedMetricIndex.TTC]),
                lane_keeping=float(weighted_metrics[proposal_idx, WeightedMetricIndex.LANE_KEEPING]),
                history_comfort=float(weighted_metrics[proposal_idx, WeightedMetricIndex.COMFORT]),
            ))
        return results

    def _resolve_pdm_masked_progress(self, scene: SceneContext) -> float | None:
        """Resolve the official v1 reference denominator from explicit PDM context.

        Preference order:
            1. scene.pdm_masked_progress (precomputed official reference)
            2. Online PDM simulation from scene.pdm_trajectory
            3. None (caller falls back to batch max)

        GT fields may coexist on the scene for analysis/debug purposes, but they do
        not define official v1 reference semantics here.
        """
        if scene.pdm_masked_progress is not None:
            return scene.pdm_masked_progress

        pdm_result = self._simulate_and_score_pdm(
            scene,
            multi_indices=[MultiMetricIndex.NO_COLLISION, MultiMetricIndex.DRIVABLE_AREA],
            weighted_indices=[],
        )
        if pdm_result is None:
            return None

        pdm_multi = float(
            pdm_result.multi_metrics[MultiMetricIndex.NO_COLLISION]
            * pdm_result.multi_metrics[MultiMetricIndex.DRIVABLE_AREA]
        )
        return float(pdm_result.progress * pdm_multi)

    def _normalize_progress_v1(
        self,
        progress_raw: np.ndarray,
        multi_metrics: np.ndarray,
        *,
        pdm_masked_progress: float | None = None,
        independent_fallback: bool = False,
    ) -> np.ndarray:
        """V1 progress normalization aligned with official PDM reference context.

        Keep raw progress centerline-based, preserve v1 multiplicative metrics, and
        normalize with masked_pred / max(masked_pred, masked_pdm) when a valid PDM
        denominator exists. Only fall back to batch max when PDM reference is unavailable.

        Any GT-derived progress that remains on SceneContext is analysis-only side
        data and must not be interpreted as the official v1 denominator here.
        """
        multi_prod = multi_metrics[
            :, [MultiMetricIndex.NO_COLLISION, MultiMetricIndex.DRIVABLE_AREA]
        ].prod(axis=1)
        masked_progress = progress_raw * multi_prod

        if pdm_masked_progress is not None and pdm_masked_progress > self._progress_distance_threshold:
            denominator = np.maximum(masked_progress, pdm_masked_progress)
            normalized = np.divide(
                masked_progress,
                denominator,
                out=np.zeros_like(masked_progress, dtype=np.float64),
                where=denominator > 0.0,
            )
        elif independent_fallback:
            normalized = np.ones_like(progress_raw, dtype=np.float64)
            normalized[multi_prod == 0.0] = 0.0
        else:
            max_masked = float(masked_progress.max()) if len(masked_progress) > 0 else 0.0
            if max_masked > self._progress_distance_threshold:
                normalized = np.clip(masked_progress / max_masked, 0.0, 1.0)
            else:
                normalized = np.ones_like(progress_raw, dtype=np.float64)
                normalized[multi_prod == 0.0] = 0.0

        return normalized

    @staticmethod
    def _aggregate_v1(multi_metrics: np.ndarray, weighted_metrics: np.ndarray) -> np.ndarray:
        """V1 aggregation: (NC * DAC) * (5*EP + 5*TTC + 2*HC) / 12"""
        multiplicative_scores = multi_metrics[
            :, [MultiMetricIndex.NO_COLLISION, MultiMetricIndex.DRIVABLE_AREA]
        ].prod(axis=1)

        weights = np.array([5.0, 5.0, 0.0, 2.0], dtype=np.float64)  # EP, TTC, LK, HC
        weighted_scores = (weighted_metrics * weights[None, :]).sum(axis=1) / weights.sum()
        return multiplicative_scores * weighted_scores
