from __future__ import annotations

from typing import List

import numpy as np

from ..enums import MultiMetricIndex, WeightedMetricIndex
from ..geometry import state_to_coords, coords_to_polygons
from ..types import SceneContext, ScoringResult
from .base import ScorerBase, PDMScorerConfig, _GTSimResult


class PDMScorerV2(ScorerBase):
    """Extended PDMS scorer with TLC, LK, and human penalty filter.

    Key differences from V1:
    - NC uses track_object_types for agent/static distinction
    - TLC and LK metrics included
    - Comfort uses ego_past_states + simulated states
    - Progress: GT normalization
    - Human penalty filter enabled
    - Aggregation: (NC * DAC * DDC * TLC) * (5*EP + 5*TTC + 2*LK + 2*HC) / 14
    """

    def __init__(self, config: PDMScorerConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self._config = config or PDMScorerConfig.v2()

    def score(self, waypoints_xy: np.ndarray, scene: SceneContext) -> ScoringResult:
        return self.score_batch(waypoints_xy[None, ...], scene)[0]

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
        ego_coords = state_to_coords(simulated_states, self._vehicle)
        ego_polygons = coords_to_polygons(ego_coords)
        ego_areas = self._calculate_ego_areas(ego_coords, scene)

        multi_metrics = np.ones((len(batch_waypoints), len(MultiMetricIndex)), dtype=np.float64)
        weighted_metrics = np.ones((len(batch_waypoints), len(WeightedMetricIndex)), dtype=np.float64)

        # V2: all 4 safety metrics multiplicative
        multi_metrics[:, MultiMetricIndex.NO_COLLISION] = self._no_at_fault_collision(
            simulated_states, ego_polygons, ego_areas, scene,
        )
        multi_metrics[:, MultiMetricIndex.DRIVABLE_AREA] = self._drivable_area_compliance(ego_areas)
        multi_metrics[:, MultiMetricIndex.DRIVING_DIRECTION] = self._driving_direction_compliance(
            ego_coords, ego_areas, scene,
        )
        multi_metrics[:, MultiMetricIndex.TRAFFIC_LIGHT] = self._traffic_light_compliance(
            ego_polygons, scene,
        )

        progress_raw = self._progress(ego_coords, scene)

        # GT simulation: shared by progress normalization + human_penalty_filter
        gt_result = self._simulate_and_score_gt(scene)
        gt_progress = gt_result.progress if gt_result else None

        weighted_metrics[:, WeightedMetricIndex.PROGRESS] = self._normalize_progress_v2(
            progress_raw,
            multi_metrics,
            gt_progress=gt_progress,
            independent_fallback=independent_progress_fallback,
        )
        weighted_metrics[:, WeightedMetricIndex.TTC] = self._time_to_collision(
            simulated_states, ego_coords, ego_areas, scene,
        )
        weighted_metrics[:, WeightedMetricIndex.LANE_KEEPING] = self._lane_keeping(
            ego_coords, ego_areas, scene,
        )
        # V2 comfort: past_states + simulated states
        weighted_metrics[:, WeightedMetricIndex.COMFORT] = self._history_comfort(
            simulated_states, scene, use_past_states=True,
        )

        # Human penalty filter
        human_penalty_names: list[str] | None = None
        if gt_result is not None and self._config.human_penalty_filter:
            has_failure = (multi_metrics < 1.0).any() or (weighted_metrics < 1.0).any()
            if has_failure:
                human_penalty_names = self._apply_human_penalty_from_gt(
                    multi_metrics, weighted_metrics, gt_result,
                )

        scores = self._aggregate_v2(multi_metrics, weighted_metrics)

        results: List[ScoringResult] = []
        for proposal_idx, score in enumerate(scores):
            results.append(ScoringResult(
                scoring_version="v2",
                pdm_score=float(score),
                no_at_fault_collisions=float(multi_metrics[proposal_idx, MultiMetricIndex.NO_COLLISION]),
                drivable_area_compliance=float(multi_metrics[proposal_idx, MultiMetricIndex.DRIVABLE_AREA]),
                driving_direction_compliance=float(multi_metrics[proposal_idx, MultiMetricIndex.DRIVING_DIRECTION]),
                traffic_light_compliance=float(multi_metrics[proposal_idx, MultiMetricIndex.TRAFFIC_LIGHT]),
                ego_progress=float(weighted_metrics[proposal_idx, WeightedMetricIndex.PROGRESS]),
                time_to_collision=float(weighted_metrics[proposal_idx, WeightedMetricIndex.TTC]),
                lane_keeping=float(weighted_metrics[proposal_idx, WeightedMetricIndex.LANE_KEEPING]),
                history_comfort=float(weighted_metrics[proposal_idx, WeightedMetricIndex.COMFORT]),
                human_penalty_applied=human_penalty_names,
            ))
        return results

    def _normalize_progress_v2(
        self,
        progress_raw: np.ndarray,
        multi_metrics: np.ndarray,
        gt_progress: float | None = None,
        *,
        independent_fallback: bool = False,
    ) -> np.ndarray:
        """V2 progress: GT normalization, zero out when multi=0."""
        multi_prod = multi_metrics.prod(axis=1)

        if gt_progress is not None and gt_progress > self._config.progress_distance_threshold:
            normalized = np.clip(progress_raw / gt_progress, 0.0, 1.0)
            normalized[multi_prod == 0.0] = 0.0
            return normalized

        if independent_fallback:
            result = np.ones_like(progress_raw, dtype=np.float64)
            result[multi_prod == 0.0] = 0.0
            return result

        max_progress = float(progress_raw.max()) if len(progress_raw) > 0 else 0.0
        if max_progress > self._config.progress_distance_threshold:
            return np.clip(progress_raw / max_progress, 0.0, 1.0)
        else:
            result = np.ones_like(progress_raw, dtype=np.float64)
            result[multi_prod == 0.0] = 0.0
            return result

    @staticmethod
    def _aggregate_v2(multi_metrics: np.ndarray, weighted_metrics: np.ndarray) -> np.ndarray:
        """V2: (NC * DAC * DDC * TLC) * (5*EP + 5*TTC + 2*LK + 2*HC) / 14"""
        multiplicative_scores = multi_metrics.prod(axis=1)
        weights = np.array([5.0, 5.0, 2.0, 2.0], dtype=np.float64)  # EP, TTC, LK, HC
        weighted_scores = (weighted_metrics * weights[None, :]).sum(axis=1) / weights.sum()
        return multiplicative_scores * weighted_scores
