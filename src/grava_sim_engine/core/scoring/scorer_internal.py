from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from shapely.geometry import Point, Polygon

from ..enums import BBCoordsIndex
from ..geometry import coords_to_polygons, state_to_coords
from ..types import InternalScoringResult, KeyActionObstacle, SceneContext, VehicleParams
from .base import ScorerBase


@dataclass(frozen=True)
class InternalKeyActionScorerConfig:
    follow_margin: float = 0.1
    nudge_trigger_margin: float = 0.1
    min_no_nudge_upper_bound: float = 0.1
    no_nudge_gate_min_front_gap: float = 0.1
    pass_margin: float = 1.0
    s_min: float = 3.0
    d_back: float = 2.0
    interaction_corridor_half_width: float = 4.0
    s_horizon_min: float = 60.0
    key_action_horizon_cap: float = 50.0
    key_action_reachable_margin: float = 8.0
    w_key: float = 0.45
    w_progress: float = 0.35
    w_comfort: float = 0.20


@dataclass(frozen=True)
class _ProjectedObstacle:
    annotation: KeyActionObstacle
    s_back: float
    s_front: float
    polygon: Polygon


def _internal_vehicle_params() -> VehicleParams:
    return VehicleParams(
        half_length=4.765 / 2.0,
        half_width=1.884 / 2.0,
        rear_axle_to_center=1.3555,
        wheel_base=1.392 + 1.438,
    )


class InternalKeyActionScorer(ScorerBase):
    """Hard internal closed-loop scorer based on labeled key-action obstacles."""

    def __init__(self, **kwargs) -> None:
        if kwargs.get("vehicle") is None:
            kwargs["vehicle"] = _internal_vehicle_params()
        super().__init__(**kwargs)

    def score(
        self,
        waypoints_xy: np.ndarray,
        scene: SceneContext,
        config: InternalKeyActionScorerConfig | None = None,
        *,
        include_ego: bool = False,
    ) -> InternalScoringResult:
        waypoints = np.asarray(waypoints_xy, dtype=np.float64)
        return self.score_batch(
            waypoints[None, ...],
            scene,
            config,
            include_ego=include_ego,
        )[0]

    def score_batch(
        self,
        trajectories_xy: np.ndarray,
        scene: SceneContext,
        config: InternalKeyActionScorerConfig | None = None,
        *,
        include_ego: bool = False,
    ) -> List[InternalScoringResult]:
        config = config or InternalKeyActionScorerConfig()
        batch_waypoints = self._coerce_trajectories(np.asarray(trajectories_xy, dtype=np.float64))

        projected_obstacles = self._project_relevant_obstacles(scene, config)
        invalid_reason = self._invalid_reason(projected_obstacles, scene)
        if invalid_reason is not None:
            return [
                InternalScoringResult(
                    sample_valid=False,
                    invalid_reason=invalid_reason,
                    error=invalid_reason,
                )
                for _ in range(len(batch_waypoints))
            ]

        ego_front_now = self._ego_current_front_progress(scene)
        first_no_nudge = self._first_no_nudge(projected_obstacles, ego_front_now, config)
        upper_bound = (
            max(
                first_no_nudge.s_back - config.follow_margin,
                config.min_no_nudge_upper_bound,
            )
            if first_no_nudge is not None
            else None
        )

        gate_limited_key_actions = self._key_actions(projected_obstacles, upper_bound)
        if first_no_nudge is None and not gate_limited_key_actions:
            return [
                InternalScoringResult(
                    sample_valid=False,
                    invalid_reason="no_key_action_obstacles",
                    num_relevant_labeled=len(projected_obstacles),
                    error="no_key_action_obstacles",
                )
                for _ in range(len(batch_waypoints))
            ]

        proposals = self._build_proposals(batch_waypoints, scene, include_ego=include_ego)
        simulated_states = self._simulator.simulate_proposals(
            ego_state=scene.ego_state,
            proposals=proposals,
            observation=scene.observation,
        )
        ego_coords = state_to_coords(simulated_states, self._vehicle)
        ego_polygons = coords_to_polygons(ego_coords)
        ego_areas = self._calculate_ego_areas(ego_coords, scene)

        nc = self._no_at_fault_collision(
            simulated_states,
            ego_polygons,
            ego_areas,
            scene,
            use_observation_types=True,
        )
        dac = self._drivable_area_compliance(ego_areas)
        comfort = self._history_comfort(simulated_states, scene, use_past_states=False)
        ego_front_max, ego_rear_max = self._ego_route_extents(ego_coords, scene)

        progress_norm, progress_norm_source = self._progress_norm(scene, config, upper_bound)
        results: list[InternalScoringResult] = []
        for idx in range(len(batch_waypoints)):
            key_action_reachable_s = self._key_action_reachable_s(
                scene,
                ego_front_max[idx],
                config,
            )
            key_actions = self._key_actions(
                projected_obstacles,
                upper_bound,
                reachable_s=key_action_reachable_s,
            )
            if first_no_nudge is None and not key_actions:
                results.append(
                    InternalScoringResult(
                        sample_valid=False,
                        invalid_reason="no_reachable_key_action_obstacles",
                        num_relevant_labeled=len(projected_obstacles),
                        ego_front_max=float(ego_front_max[idx]),
                        ego_rear_max=float(ego_rear_max[idx]),
                        raw_progress=float(ego_front_max[idx]),
                        progress_norm=float(progress_norm),
                        progress_norm_source=progress_norm_source,
                        error="no_reachable_key_action_obstacles",
                    )
                )
                continue

            overrun = upper_bound is not None and ego_front_max[idx] > upper_bound
            label_one_passed = sum(
                ego_front_max[idx] > obstacle.s_back + config.nudge_trigger_margin
                for obstacle in key_actions
            )
            no_nudge_count = 1 if first_no_nudge is not None else 0
            no_nudge_passed = 1 if no_nudge_count and not overrun else 0
            total_key_actions = len(key_actions) + no_nudge_count
            passed_count = int(label_one_passed) + no_nudge_passed
            key_action_score = passed_count / total_key_actions
            if overrun:
                progress_score = 0.0
            else:
                progress_score = float(np.clip(ego_front_max[idx] / progress_norm, 0.0, 1.0))

            safety_score = float(nc[idx] * dac[idx])
            internal_score = safety_score * (
                config.w_key * key_action_score
                + config.w_progress * progress_score
                + config.w_comfort * float(comfort[idx])
            )
            results.append(
                InternalScoringResult(
                    internal_score=float(internal_score),
                    safety_score=safety_score,
                    comfort_score=float(comfort[idx]),
                    key_action_score=float(key_action_score),
                    progress_score=float(progress_score),
                    no_at_fault_collisions=float(nc[idx]),
                    drivable_area_compliance=float(dac[idx]),
                    sample_valid=True,
                    first_no_nudge_upper_bound=None if upper_bound is None else float(upper_bound),
                    overrun_no_nudge_gate=bool(overrun),
                    num_relevant_labeled=len(projected_obstacles),
                    num_key_actions=total_key_actions,
                    num_key_actions_passed=int(passed_count),
                    ego_front_max=float(ego_front_max[idx]),
                    ego_rear_max=float(ego_rear_max[idx]),
                    raw_progress=float(ego_front_max[idx]),
                    progress_norm=float(progress_norm),
                    progress_norm_source=progress_norm_source,
                )
            )
        return results

    def _project_relevant_obstacles(
        self,
        scene: SceneContext,
        config: InternalKeyActionScorerConfig,
    ) -> list[_ProjectedObstacle]:
        s0 = scene.centerline.project(Point(0.0, 0.0))
        corridor = scene.centerline.linestring.buffer(config.interaction_corridor_half_width)
        horizon = max(
            config.s_horizon_min,
            float(scene.candidate_progress or 0.0),
            float(scene.gt_progress or 0.0),
        )

        relevant: list[_ProjectedObstacle] = []
        for annotation in scene.key_action_obstacles:
            if annotation.label not in (0, 1):
                continue
            coords = np.asarray(annotation.polygon_coords, dtype=np.float64)
            if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
                continue
            polygon = Polygon(coords[:, :2])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or not polygon.intersects(corridor):
                continue
            projected = [
                scene.centerline.project(Point(float(point[0]), float(point[1]))) - s0
                for point in coords[:, :2]
            ]
            s_back = float(min(projected))
            s_front = float(max(projected))
            if s_front <= -config.d_back:
                continue
            if s_back >= horizon:
                continue
            relevant.append(
                _ProjectedObstacle(
                    annotation=annotation,
                    s_back=s_back,
                    s_front=s_front,
                    polygon=polygon,
                )
            )
        return sorted(relevant, key=lambda item: item.s_back)

    @staticmethod
    def _invalid_reason(
        projected_obstacles: list[_ProjectedObstacle],
        scene: SceneContext,
    ) -> str | None:
        if not scene.key_action_obstacles:
            return "missing_key_action_obstacles"
        if not projected_obstacles:
            return "no_relevant_labeled_obstacles"
        return None

    @staticmethod
    def _first_no_nudge(
        projected_obstacles: list[_ProjectedObstacle],
        ego_front_now: float,
        config: InternalKeyActionScorerConfig,
    ) -> _ProjectedObstacle | None:
        min_gate_s_back = ego_front_now + config.no_nudge_gate_min_front_gap
        no_nudges = [
            obstacle
            for obstacle in projected_obstacles
            if obstacle.annotation.label == 0 and obstacle.s_back > min_gate_s_back
        ]
        return min(no_nudges, key=lambda item: item.s_back) if no_nudges else None

    @staticmethod
    def _key_actions(
        projected_obstacles: list[_ProjectedObstacle],
        upper_bound: float | None,
        reachable_s: float | None = None,
    ) -> list[_ProjectedObstacle]:
        def is_selected(obstacle: _ProjectedObstacle) -> bool:
            if obstacle.annotation.label != 1:
                return False
            if upper_bound is not None and obstacle.s_back >= upper_bound:
                return False
            if reachable_s is not None and obstacle.s_back >= reachable_s:
                return False
            return True

        return [
            obstacle for obstacle in projected_obstacles
            if is_selected(obstacle)
        ]

    @staticmethod
    def _key_action_reachable_s(
        scene: SceneContext,
        ego_front_max: float,
        config: InternalKeyActionScorerConfig,
    ) -> float:
        reachable_base = max(
            float(scene.candidate_progress or 0.0),
            float(scene.gt_progress or 0.0),
            float(ego_front_max),
        )
        return min(
            float(config.key_action_horizon_cap),
            reachable_base + float(config.key_action_reachable_margin),
        )

    @staticmethod
    def _ego_route_extents(
        ego_coords: np.ndarray,
        scene: SceneContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        s0 = scene.centerline.project(Point(0.0, 0.0))
        front_indices = [BBCoordsIndex.FRONT_LEFT, BBCoordsIndex.FRONT_RIGHT]
        rear_indices = [BBCoordsIndex.REAR_LEFT, BBCoordsIndex.REAR_RIGHT]
        batch_size = ego_coords.shape[0]
        front_max = np.zeros(batch_size, dtype=np.float64)
        rear_max = np.zeros(batch_size, dtype=np.float64)
        for batch_idx in range(batch_size):
            front_values: list[float] = []
            rear_values: list[float] = []
            for time_idx in range(ego_coords.shape[1]):
                for corner_idx in front_indices:
                    point = ego_coords[batch_idx, time_idx, corner_idx]
                    front_values.append(scene.centerline.project(Point(*point)) - s0)
                for corner_idx in rear_indices:
                    point = ego_coords[batch_idx, time_idx, corner_idx]
                    rear_values.append(scene.centerline.project(Point(*point)) - s0)
            front_max[batch_idx] = max(front_values) if front_values else 0.0
            rear_max[batch_idx] = max(rear_values) if rear_values else 0.0
        return front_max, rear_max

    def _ego_current_front_progress(self, scene: SceneContext) -> float:
        s0 = scene.centerline.project(Point(0.0, 0.0))
        ego_coords = state_to_coords(scene.ego_state[None, :], self._vehicle)
        front_indices = [BBCoordsIndex.FRONT_LEFT, BBCoordsIndex.FRONT_RIGHT]
        return max(
            scene.centerline.project(Point(*ego_coords[0, corner_idx])) - s0
            for corner_idx in front_indices
        )

    @staticmethod
    def _progress_norm(
        scene: SceneContext,
        config: InternalKeyActionScorerConfig,
        upper_bound: float | None,
    ) -> tuple[float, str]:
        candidates: list[tuple[str, float]] = []
        if scene.candidate_progress is not None:
            candidates.append(("candidate", float(scene.candidate_progress)))
        if scene.gt_progress is not None:
            candidates.append(("logged", float(scene.gt_progress)))

        if candidates:
            best_source, best_value = max(candidates, key=lambda item: item[1])
            norm = max(best_value, config.s_min)
            source = best_source if best_value >= config.s_min else "s_min"
            if len(candidates) == 2 and best_value >= config.s_min:
                source = f"candidate+logged:{best_source}"
        else:
            norm = config.s_min
            source = "s_min"

        if upper_bound is not None:
            norm = min(norm, upper_bound)
            source = f"{source}|upper_bound"
        return max(norm, 1e-6), source
