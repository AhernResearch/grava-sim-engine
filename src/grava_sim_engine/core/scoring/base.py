from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import numpy.typing as npt
import shapely
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, nearest_points

from ..comfort import ego_comfort_violation, ego_is_comfortable
from ..enums import (
    BBCoordsIndex,
    CollisionType,
    DRIVABLE_LAYERS,
    EgoAreaIndex,
    MultiMetricIndex,
    SemanticMapLayer,
    StateIndex,
    WeightedMetricIndex,
)
from ..geometry import coords_to_polygons, normalize_angle, state_to_coords
from ..occupancy import DrivableMap, OccupancyMap, _LAYER_NAME_TO_ENUM, _normalize_layer_name
from ..simulator import PDMSimulator
from ..types import SceneContext, VehicleParams


RED_LIGHT_TOKEN_PREFIX = "red_light"

# Official NavSim uses 5e-2 for collision stopped classification
_COLLISION_STOPPED_THRESHOLD = 5e-2


@dataclass(frozen=True)
class PDMScorerConfig:
    scoring_version: str = "v1"
    progress_weight: float = 5.0
    ttc_weight: float = 5.0
    lane_keeping_weight: float = 2.0
    comfort_weight: float = 2.0
    driving_direction_horizon: float = 1.0
    driving_direction_compliance_threshold: float = 2.0
    driving_direction_violation_threshold: float = 6.0
    stopped_speed_threshold: float = 5e-03
    future_collision_horizon: float = 1.0
    progress_distance_threshold: float = 5.0
    lane_keeping_deviation: float = 0.5
    lane_keeping_horizon: float = 2.0
    human_penalty_filter: bool = True

    @classmethod
    def v1(cls) -> "PDMScorerConfig":
        """NavSim v1 PDMS: (NC×DAC) × (5EP+5TTC+2C)/12"""
        return cls(scoring_version="v1", lane_keeping_weight=0.0, human_penalty_filter=False)

    @classmethod
    def v2(cls) -> "PDMScorerConfig":
        """NavSim v2 EPDMS-like: (NC×DAC×DDC×TLC) × (5EP+5TTC+2LK+2HC)/14"""
        return cls(scoring_version="v2")

    @property
    def weighted_metrics_array(self) -> npt.NDArray[np.float64]:
        weights = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
        weights[WeightedMetricIndex.PROGRESS] = self.progress_weight
        weights[WeightedMetricIndex.TTC] = self.ttc_weight
        weights[WeightedMetricIndex.LANE_KEEPING] = self.lane_keeping_weight
        weights[WeightedMetricIndex.COMFORT] = self.comfort_weight
        return weights


@dataclass(frozen=True)
class RLScorerConfig:
    """RL reward scoring configuration.

    Safety metrics support 'continuous' (distance-based) or 'discrete' (NavSim-compatible) mode.
    Performance metrics are always continuous.
    Aggregation: soft safety gate × weighted performance average.
    """

    ep_weight: float = 5.0
    ttc_weight: float = 5.0
    hc_weight: float = 2.0
    lk_weight: float = 0.0

    safety_gate_alpha: float = 0.5
    safety_mode: str = "continuous"

    nc_weight: float = 5.0
    dac_weight: float = 3.0
    ddc_weight: float = 0.0
    tlc_weight: float = 0.0

    collision_distance_scale: float = 2.0
    dac_margin: float = 2.0
    tlc_margin: float = 1.0
    obstacle_clearance_margin: float = 5.0  # STRtree query buffer & distance cap (meters)
    boundary_clearance_margin: float = 2.0  # distance cap for boundary clearance (meters)

    progress_distance_threshold: float = 5.0
    ttc_horizon: float = 4.0  # match 4s simulation horizon
    lane_keeping_deviation: float = 0.5
    lane_keeping_max_deviation: float = 2.0
    lane_keeping_horizon: float = 2.0
    driving_direction_horizon: float = 1.0  # rolling window in seconds (matches discrete DDC)
    driving_direction_compliance_threshold: float = 2.0
    driving_direction_violation_threshold: float = 6.0
    stopped_speed_threshold: float = 5e-03
    future_collision_horizon: float = 1.0

    @classmethod
    def v1(cls) -> "RLScorerConfig":
        """Continuous RL aligned with NavSim v1 active metric set."""
        return cls(ddc_weight=0.0, tlc_weight=0.0, lk_weight=0.0)

    @property
    def weights_array(self) -> npt.NDArray[np.float64]:
        return np.array(
            [self.nc_weight, self.dac_weight, self.ddc_weight, self.tlc_weight,
             self.ep_weight, self.ttc_weight, self.lk_weight, self.hc_weight],
            dtype=np.float64,
        )

    @property
    def performance_weights(self) -> npt.NDArray[np.float64]:
        return np.array(
            [self.ep_weight, self.ttc_weight, self.lk_weight, self.hc_weight],
            dtype=np.float64,
        )


@dataclass
class _GTSimResult:
    """Cached GT simulation result, shared by progress normalization and human_penalty_filter."""
    progress: float
    multi_metrics: np.ndarray    # (len(MultiMetricIndex),)
    weighted_metrics: np.ndarray  # (len(WeightedMetricIndex),)


class ScorerBase:
    """Shared infrastructure for all scorer versions.

    Provides coordinate transforms, collision classification, spatial queries,
    and simulation pipeline. Subclasses implement version-specific scoring logic.
    """

    REQUIRED_NUM_WAYPOINTS = 8

    def __init__(
        self,
        vehicle: VehicleParams | None = None,
        simulator: PDMSimulator | None = None,
        discretization_time: float = 0.1,
    ) -> None:
        self._vehicle = vehicle or VehicleParams()
        self._simulator = simulator or PDMSimulator(
            discretization_time=discretization_time,
            vehicle=self._vehicle,
        )

    # ── Trajectory preprocessing ──────────────────────────────────────

    @staticmethod
    def _coerce_trajectories(trajectories_xy: np.ndarray) -> np.ndarray:
        trajectory_array = np.asarray(trajectories_xy, dtype=np.float64)
        if trajectory_array.ndim == 2:
            if trajectory_array.shape[1] not in (2, 3):
                raise ValueError(f"waypoints must have shape [T,2|3], got {trajectory_array.shape}")
            trajectory_array = trajectory_array[None, ...]
        elif trajectory_array.ndim != 3 or trajectory_array.shape[-1] not in (2, 3):
            raise ValueError(
                f"trajectories must have shape [B,T,2|3] or [T,2|3], got {trajectory_array.shape}"
            )

        T = trajectory_array.shape[1]
        if T != ScorerBase.REQUIRED_NUM_WAYPOINTS:
            raise ValueError(
                f"Input must be exactly {ScorerBase.REQUIRED_NUM_WAYPOINTS} waypoints "
                f"at 0.5s intervals (4s horizon), got {T} waypoints. "
                f"Expected shape: [B, {ScorerBase.REQUIRED_NUM_WAYPOINTS}, 2|3]"
            )
        return trajectory_array

    @staticmethod
    def _derive_relative_headings(waypoints_xy: np.ndarray) -> np.ndarray:
        """Derive headings from xy waypoints using cubic spline tangent direction.

        Simple arctan2(dy, dx) on sparse 0.5s waypoints gives the chord direction
        between consecutive points, not the tangent at each point.  On curves this
        introduces significant heading error that propagates through LQR simulation
        and causes false drivable-area violations.

        Cubic spline fit + first derivative gives the tangent heading, which is
        consistent with NavSim/RecogDrive where the agent directly outputs heading.
        """
        batch_size, num_points, _ = waypoints_xy.shape
        t = np.arange(num_points, dtype=np.float64)
        headings = np.zeros((batch_size, num_points), dtype=np.float64)

        for b in range(batch_size):
            x, y = waypoints_xy[b, :, 0], waypoints_xy[b, :, 1]

            deltas = np.diff(waypoints_xy[b, :, :2], axis=0)
            segment_lengths = np.linalg.norm(deltas, axis=1)
            total_dist = np.sum(segment_lengths)
            if total_dist < 1e-6:
                headings[b] = 0.0
                continue

            # Repeated/stationary waypoints make cubic-spline tangents oscillate
            # wildly. Propagate the nearest valid motion heading through those
            # stationary spans instead of inventing a turn.
            moving = segment_lengths > 1e-3
            if not np.all(moving):
                segment_headings = np.zeros(num_points - 1, dtype=np.float64)
                segment_headings[moving] = np.arctan2(deltas[moving, 1], deltas[moving, 0])
                first_heading = float(segment_headings[np.flatnonzero(moving)[0]])
                current_heading = first_heading
                headings[b, 0] = current_heading
                for point_idx in range(1, num_points):
                    if moving[point_idx - 1]:
                        current_heading = float(segment_headings[point_idx - 1])
                    headings[b, point_idx] = current_heading
                continue

            cs_x = CubicSpline(t, x)
            cs_y = CubicSpline(t, y)
            # heading = tangent direction = arctan2(dy/dt, dx/dt)
            headings[b] = np.arctan2(cs_y(t, 1), cs_x(t, 1))

        return headings

    @staticmethod
    def _ego_to_global(waypoints_xy: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
        """Convert ego-relative waypoints to global SE2.

        Expects ego-relative waypoints in NUPLAN frame:
        - x: forward
        - y: left
        """
        batch_size, horizon, _ = waypoints_xy.shape
        ego_x = float(ego_state[StateIndex.X])
        ego_y = float(ego_state[StateIndex.Y])
        ego_heading = float(ego_state[StateIndex.HEADING])
        cos_h = np.cos(ego_heading)
        sin_h = np.sin(ego_heading)

        relative_xy = waypoints_xy[..., :2]
        global_xy = np.zeros((batch_size, horizon, 2), dtype=np.float64)
        global_xy[..., 0] = relative_xy[..., 0] * cos_h - relative_xy[..., 1] * sin_h + ego_x
        global_xy[..., 1] = relative_xy[..., 0] * sin_h + relative_xy[..., 1] * cos_h + ego_y

        relative_heading = (
            waypoints_xy[..., 2]
            if waypoints_xy.shape[-1] == 3
            else ScorerBase._derive_relative_headings(relative_xy)
        )
        headings = normalize_angle(relative_heading + ego_heading)
        return np.concatenate([global_xy, headings[..., None]], axis=-1)

    def _build_proposals(
        self,
        waypoints: np.ndarray,
        scene: SceneContext,
        input_interval: float = 0.5,
        *,
        include_ego: bool = False,
    ) -> np.ndarray:
        """Convert ego-relative waypoints to global proposals with interpolation.

        Args:
            include_ego: Whether the input waypoints already contain ego pose
                as the first point. If False (default), ego pose is prepended.
        Returns:
            (B, 41, 3) global proposals including ego at t=0.
        """
        ego_state = scene.ego_state
        global_coarse = self._ego_to_global(waypoints, ego_state)
        need_prepend = not include_ego

        sim_dt = float(scene.observation.interval_time)
        if sim_dt <= 0 or abs(input_interval - sim_dt) < 1e-6:
            if need_prepend:
                ego_pose = np.array(
                    [ego_state[StateIndex.X], ego_state[StateIndex.Y], ego_state[StateIndex.HEADING]],
                    dtype=np.float64,
                )
                batch_size = global_coarse.shape[0]
                return np.concatenate(
                    [np.broadcast_to(ego_pose, (batch_size, 1, 3)), global_coarse], axis=1,
                )
            return global_coarse

        ratio = round(input_interval / sim_dt)
        if ratio <= 1:
            if need_prepend:
                ego_pose = np.array(
                    [ego_state[StateIndex.X], ego_state[StateIndex.Y], ego_state[StateIndex.HEADING]],
                    dtype=np.float64,
                )
                batch_size = global_coarse.shape[0]
                return np.concatenate(
                    [np.broadcast_to(ego_pose, (batch_size, 1, 3)), global_coarse], axis=1,
                )
            return global_coarse

        batch_size, num_coarse, _ = global_coarse.shape
        ego_pose = np.array(
            [ego_state[StateIndex.X], ego_state[StateIndex.Y], ego_state[StateIndex.HEADING]],
            dtype=np.float64,
        )
        extended = np.concatenate(
            [np.broadcast_to(ego_pose, (batch_size, 1, 3)), global_coarse], axis=1,
        )

        num_extended = num_coarse + 1
        num_fine = (num_extended - 1) * ratio + 1

        # Linear interpolation matching NavSim's InterpolatedTrajectory behavior
        t_coarse = np.arange(num_extended, dtype=np.float64)
        t_fine = np.linspace(0, num_extended - 1, num_fine)

        # Unwrap heading for smooth interpolation across -pi/pi boundary
        extended_heading = np.unwrap(extended[..., 2], axis=1)

        proposals = np.zeros((batch_size, num_fine, 3), dtype=np.float64)
        for b in range(batch_size):
            proposals[b, :, 0] = np.interp(t_fine, t_coarse, extended[b, :, 0])
            proposals[b, :, 1] = np.interp(t_fine, t_coarse, extended[b, :, 1])
            proposals[b, :, 2] = normalize_angle(
                np.interp(t_fine, t_coarse, extended_heading[b]),
            )
        return proposals  # (B, 41, 3) always includes ego at t=0

    def _waypoints_to_proposals(self, waypoints_xy: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
        return self._ego_to_global(waypoints_xy, ego_state)

    # ── Ego area computation ──────────────────────────────────────────

    def _calculate_ego_areas(self, ego_coords: np.ndarray, scene: SceneContext) -> np.ndarray:
        batch_size, horizon, _, _ = ego_coords.shape
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        corners = ego_coords[:, :, :4, :]

        ego_areas = np.zeros((batch_size, horizon, len(EgoAreaIndex)), dtype=bool)

        lane_membership = self._points_in_map_tokens(
            corners, scene.drivable_area_map,
            {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR},
        )
        if len(lane_membership) > 0:
            lane_corner_counts = lane_membership.sum(axis=-1)
            multiple_lane_mask = (lane_corner_counts > 0).sum(axis=0) > 1
            single_lane_mask = np.any(lane_corner_counts == 4, axis=0)
            ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES] = multiple_lane_mask & ~single_lane_mask

        center_membership = scene.drivable_area_map.points_in_polygons(centers)
        corner_membership = scene.drivable_area_map.points_in_polygons(corners)
        drivable_layers_all = list(DRIVABLE_LAYERS)
        corner_in_drivable = corner_membership[..., drivable_layers_all[0]]
        for layer in drivable_layers_all[1:]:
            corner_in_drivable = corner_in_drivable | corner_membership[..., layer]
        ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA] = ~corner_in_drivable.all(axis=-1)

        route_layers = self._points_in_route_lanes(centers, scene.drivable_area_map, scene.route_lane_ids)
        ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC] = ~route_layers
        ego_areas[:, :, EgoAreaIndex.IN_INTERSECTION] = center_membership[..., SemanticMapLayer.INTERSECTION]

        return ego_areas

    # ── Progress ──────────────────────────────────────────────────────

    @staticmethod
    def _progress(ego_coords: np.ndarray, scene: SceneContext) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        progress = np.zeros(len(ego_coords), dtype=np.float64)
        for proposal_idx in range(len(ego_coords)):
            start_progress = scene.centerline.project(Point(*centers[proposal_idx, 0]))
            end_progress = scene.centerline.project(Point(*centers[proposal_idx, -1]))
            progress[proposal_idx] = max(0.0, end_progress - start_progress)
        return progress

    # ── Collision classification ──────────────────────────────────────

    def _classify_collision_type(
        self,
        ego_state: np.ndarray,
        ego_polygon: BaseGeometry,
        track_polygon: BaseGeometry,
        token: str,
        scene: SceneContext,
        time_idx: int,
    ) -> CollisionType:
        ego_speed = np.hypot(
            ego_state[StateIndex.VELOCITY_X], ego_state[StateIndex.VELOCITY_Y],
        )
        if ego_speed <= _COLLISION_STOPPED_THRESHOLD:
            return CollisionType.STOPPED_EGO_OPEN

        track_speed = scene.track_speeds.get(token)
        if track_speed is None:
            track_speed = self._estimate_track_speed(scene, token, time_idx)
        if track_speed <= _COLLISION_STOPPED_THRESHOLD:
            return CollisionType.STOPPED_TRACK_OPEN

        if self._is_track_behind_ego(ego_state, track_polygon):
            return CollisionType.ACTIVE_REAR_BUMPER

        front_bumper = LineString([
            ego_polygon.exterior.coords[0], ego_polygon.exterior.coords[3],
        ])
        if front_bumper.intersects(track_polygon):
            return CollisionType.ACTIVE_FRONT_BUMPER

        return CollisionType.ACTIVE_LATERAL

    def _collision_penalty(
        self,
        collision_type: CollisionType,
        ego_area: np.ndarray,
        token: str,
        scene: SceneContext,
    ) -> float:
        """At-fault collision penalty.

        Uses scene.track_object_types for object type lookup (v2 behavior).
        V1 override: _collision_penalty_from_observation.
        """
        if collision_type in (CollisionType.ACTIVE_REAR_BUMPER, CollisionType.STOPPED_EGO_OPEN):
            return 1.0

        is_agent = scene.track_object_types.get(token, "agent") != "static"
        at_fault_score = 0.0 if is_agent else 0.5

        if collision_type == CollisionType.ACTIVE_FRONT_BUMPER:
            return at_fault_score
        if collision_type == CollisionType.STOPPED_TRACK_OPEN:
            return at_fault_score
        if ego_area[EgoAreaIndex.MULTIPLE_LANES] or ego_area[EgoAreaIndex.NON_DRIVABLE_AREA]:
            return at_fault_score
        return 1.0

    @staticmethod
    def _collision_penalty_from_observation(
        collision_type: CollisionType,
        ego_area: np.ndarray,
        token: str,
        scene: SceneContext,
    ) -> float:
        """V1 collision penalty: reads object type from scene.track_object_types.

        Equivalent to recogdrive's observation.unique_objects[token].tracked_object_type.
        In our architecture, track_object_types is extracted from the same source.
        agent → 0.0, static → 0.5 (matches recogdrive).
        """
        if collision_type in (CollisionType.ACTIVE_REAR_BUMPER, CollisionType.STOPPED_EGO_OPEN):
            return 1.0

        # Matches recogdrive: tracked_object_type in AGENT_TYPES → 0.0, else 0.5
        is_agent = scene.track_object_types.get(token, "agent") != "static"
        at_fault_score = 0.0 if is_agent else 0.5

        if collision_type == CollisionType.ACTIVE_FRONT_BUMPER:
            return at_fault_score
        if collision_type == CollisionType.STOPPED_TRACK_OPEN:
            return at_fault_score
        if ego_area[EgoAreaIndex.MULTIPLE_LANES] or ego_area[EgoAreaIndex.NON_DRIVABLE_AREA]:
            return at_fault_score
        return 1.0

    # ── PDMS discrete metrics ─────────────────────────────────────────

    def _no_at_fault_collision(
        self,
        simulated_states: np.ndarray,
        ego_polygons: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        *,
        use_observation_types: bool = False,
    ) -> np.ndarray:
        scores = np.ones(len(simulated_states), dtype=np.float64)
        collided_track_ids = [set(scene.collided_track_ids) for _ in range(len(simulated_states))]

        penalty_fn = self._collision_penalty_from_observation if use_observation_types else self._collision_penalty

        for time_idx in range(simulated_states.shape[1]):
            occupancy_map = self._get_occupancy_map(scene, time_idx)
            if occupancy_map is None:
                continue

            for proposal_idx in range(len(simulated_states)):
                if scores[proposal_idx] == 0.0:
                    continue

                ego_polygon = ego_polygons[proposal_idx, time_idx]
                if not occupancy_map.intersects(ego_polygon)[0]:
                    continue

                for token in occupancy_map.get_colliding_tokens(ego_polygon):
                    if RED_LIGHT_TOKEN_PREFIX in token or token in collided_track_ids[proposal_idx]:
                        continue

                    track_polygon = occupancy_map[token]
                    collision_type = self._classify_collision_type(
                        simulated_states[proposal_idx, time_idx], ego_polygon,
                        track_polygon, token, scene, time_idx,
                    )
                    at_fault_score = penalty_fn(
                        collision_type, ego_areas[proposal_idx, time_idx], token, scene,
                    )
                    scores[proposal_idx] = min(scores[proposal_idx], at_fault_score)
                    if at_fault_score >= 1.0:
                        collided_track_ids[proposal_idx].add(token)

        return scores

    @staticmethod
    def _drivable_area_compliance(ego_areas: np.ndarray) -> np.ndarray:
        off_road = ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].any(axis=1)
        scores = np.ones(len(ego_areas), dtype=np.float64)
        scores[off_road] = 0.0
        return scores

    def _driving_direction_compliance(
        self,
        ego_coords: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        *,
        horizon: float = 1.0,
        compliance_threshold: float = 2.0,
        violation_threshold: float = 6.0,
    ) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        oncoming_progress = np.zeros(centers.shape[:2], dtype=np.float64)
        oncoming_progress[:, 1:] = np.linalg.norm(centers[:, 1:] - centers[:, :-1], axis=-1)

        not_oncoming = ~ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        oncoming_progress[not_oncoming] = 0.0

        horizon_steps = max(int(round(horizon / self._dt(scene))), 1)
        rolling_progress = np.zeros_like(oncoming_progress)
        for time_idx in range(oncoming_progress.shape[1]):
            start_idx = max(0, time_idx - horizon_steps)
            rolling_progress[:, time_idx] = oncoming_progress[:, start_idx: time_idx + 1].sum(axis=1)

        max_progress = rolling_progress.max(axis=1)
        scores = np.ones(len(ego_coords), dtype=np.float64)
        medium_mask = (max_progress >= compliance_threshold) & (max_progress < violation_threshold)
        severe_mask = max_progress >= violation_threshold
        scores[medium_mask] = 0.5
        scores[severe_mask] = 0.0
        return scores

    def _traffic_light_compliance(self, ego_polygons: np.ndarray, scene: SceneContext) -> np.ndarray:
        scores = np.ones(len(ego_polygons), dtype=np.float64)
        for time_idx in range(ego_polygons.shape[1]):
            red_light_map = self._get_red_light_map(scene, time_idx)
            if red_light_map is None:
                continue
            for proposal_idx in range(len(ego_polygons)):
                if scores[proposal_idx] == 0.0:
                    continue
                ego_polygon = ego_polygons[proposal_idx, time_idx]
                if not red_light_map.intersects(ego_polygon)[0]:
                    continue
                for token in red_light_map.get_colliding_tokens(ego_polygon):
                    if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                        scores[proposal_idx] = 0.0
                        break
        return scores

    def _time_to_collision(
        self,
        simulated_states: np.ndarray,
        ego_coords: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        future_collision_horizon: float = 1.0,
        stopped_speed_threshold: float = 5e-3,
    ) -> np.ndarray:
        dt = self._dt(scene)
        total_forward_steps = max(int(round(future_collision_horizon / dt)), 1)
        sample_stride = max(int(round(0.33 / dt)), 1)
        future_offsets = np.arange(0, total_forward_steps, sample_stride, dtype=int)
        if len(future_offsets) == 0:
            future_offsets = np.asarray([0], dtype=int)

        speeds = np.hypot(
            simulated_states[..., StateIndex.VELOCITY_X],
            simulated_states[..., StateIndex.VELOCITY_Y],
        )
        dxy_per_second = np.stack(
            [
                np.cos(simulated_states[..., StateIndex.HEADING]) * speeds,
                np.sin(simulated_states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        projected_coords = np.repeat(ego_coords[:, :, None, :, :], len(future_offsets), axis=2)
        for offset_idx, future_offset in enumerate(future_offsets):
            projected_coords[:, :, offset_idx] += (
                dxy_per_second[:, :, None, :] * (float(future_offset) * dt)
            )
        projected_polygons = coords_to_polygons(projected_coords)

        scores = np.ones(len(simulated_states), dtype=np.float64)
        collided_track_ids = [set(scene.collided_track_ids) for _ in range(len(simulated_states))]
        num_time_steps = simulated_states.shape[1]

        for time_idx in range(num_time_steps):
            for offset_idx, future_offset in enumerate(future_offsets):
                current_time_idx = time_idx + future_offset
                occupancy_map = self._get_occupancy_map(scene, current_time_idx)
                if occupancy_map is None:
                    continue
                for proposal_idx in range(len(simulated_states)):
                    if scores[proposal_idx] == 0.0:
                        continue
                    if speeds[proposal_idx, time_idx] < stopped_speed_threshold:
                        continue
                    ego_polygon = projected_polygons[proposal_idx, time_idx, offset_idx]
                    if not occupancy_map.intersects(ego_polygon)[0]:
                        continue
                    for token in occupancy_map.get_colliding_tokens(ego_polygon):
                        if RED_LIGHT_TOKEN_PREFIX in token or token in collided_track_ids[proposal_idx]:
                            continue
                        ego_in_intersection = scene.drivable_area_map.is_in_layer(
                            simulated_states[proposal_idx, time_idx, [StateIndex.X, StateIndex.Y]],
                            SemanticMapLayer.INTERSECTION,
                        )
                        if self._is_ttc_violation(
                            simulated_states[proposal_idx, time_idx],
                            occupancy_map[token],
                            ego_areas[proposal_idx, time_idx],
                            ego_in_intersection,
                        ):
                            scores[proposal_idx] = 0.0
                            collided_track_ids[proposal_idx].add(token)
                        else:
                            collided_track_ids[proposal_idx].add(token)
        return scores

    def _lane_keeping(
        self,
        ego_coords: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        *,
        deviation: float = 0.5,
        horizon: float = 2.0,
    ) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        horizon_steps = max(int(round(horizon / self._dt(scene))), 1)
        scores = np.ones(len(ego_coords), dtype=np.float64)

        for proposal_idx in range(len(ego_coords)):
            consecutive_exceeds = 0
            violated = False
            for time_idx in range(ego_coords.shape[1]):
                if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                    continue
                dev = Point(*centers[proposal_idx, time_idx]).distance(scene.centerline.linestring)
                if dev > deviation:
                    consecutive_exceeds += 1
                    if consecutive_exceeds >= horizon_steps:
                        violated = True
                        break
                else:
                    consecutive_exceeds = 0
            if violated:
                scores[proposal_idx] = 0.0
        return scores

    def _history_comfort(
        self,
        simulated_states: np.ndarray,
        scene: SceneContext,
        *,
        use_past_states: bool = True,
    ) -> np.ndarray:
        """Discrete comfort: 1.0 if comfortable, 0.0 if not.

        Args:
            use_past_states: V2 prepends ego_past_states. V1 uses simulated states only.
        """
        scores = np.ones(len(simulated_states), dtype=np.float64)
        if use_past_states:
            past_states = np.asarray(scene.ego_past_states, dtype=np.float64)
            if len(past_states) == 0:
                return scores

        dt = self._dt(scene)
        for proposal_idx in range(len(simulated_states)):
            if use_past_states:
                padded = np.concatenate([past_states, simulated_states[proposal_idx]], axis=0)
            else:
                padded = simulated_states[proposal_idx]
            time_points_s = np.arange(len(padded), dtype=np.float64) * dt
            scores[proposal_idx] = 1.0 if ego_is_comfortable(padded, time_points_s) else 0.0
        return scores

    # ── Reference simulation ──────────────────────────────────────────

    def _simulate_and_score_reference(
        self,
        reference_trajectory: np.ndarray | None,
        scene: SceneContext,
        *,
        multi_indices: Iterable[int | MultiMetricIndex] | None = None,
        weighted_indices: Iterable[int | WeightedMetricIndex] | None = None,
    ) -> _GTSimResult | None:
        """Simulate one reference trajectory and compute shared metrics.

        This helper is intentionally reference-agnostic: callers decide whether the
        trajectory is GT (analysis/debug side channel) or PDM (official v1 reference
        context). The simulation logic is shared, but the semantics are owned by the
        caller-specific wrapper.
        """
        if reference_trajectory is None:
            return None

        if multi_indices is None:
            multi_set = set(MultiMetricIndex)
        else:
            multi_set = {MultiMetricIndex(int(idx)) for idx in multi_indices}

        if weighted_indices is None:
            weighted_set = set(WeightedMetricIndex)
        else:
            weighted_set = {WeightedMetricIndex(int(idx)) for idx in weighted_indices}

        trajectory = np.asarray(reference_trajectory, dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[-1] not in (2, 3) or trajectory.shape[0] == 0:
            return None

        reference_waypoints = trajectory[None, ...]
        try:
            if reference_waypoints.shape[1] == self.REQUIRED_NUM_WAYPOINTS:
                proposals = self._build_proposals(reference_waypoints, scene)
            else:
                proposals = self._waypoints_to_proposals(reference_waypoints, scene.ego_state)

            simulated = self._simulator.simulate_proposals(
                ego_state=scene.ego_state, proposals=proposals, observation=scene.observation,
            )
            coords = state_to_coords(simulated, self._vehicle)
            polygons = coords_to_polygons(coords)
            areas = self._calculate_ego_areas(coords, scene)

            progress = float(self._progress(coords, scene)[0])

            multi = np.ones(len(MultiMetricIndex), dtype=np.float64)
            if MultiMetricIndex.NO_COLLISION in multi_set:
                multi[MultiMetricIndex.NO_COLLISION] = self._no_at_fault_collision(
                    simulated, polygons, areas, scene,
                )[0]
            if MultiMetricIndex.DRIVABLE_AREA in multi_set:
                multi[MultiMetricIndex.DRIVABLE_AREA] = self._drivable_area_compliance(areas)[0]
            if MultiMetricIndex.DRIVING_DIRECTION in multi_set:
                multi[MultiMetricIndex.DRIVING_DIRECTION] = self._driving_direction_compliance(
                    coords, areas, scene,
                )[0]
            if MultiMetricIndex.TRAFFIC_LIGHT in multi_set:
                multi[MultiMetricIndex.TRAFFIC_LIGHT] = self._traffic_light_compliance(
                    polygons, scene,
                )[0]

            weighted = np.ones(len(WeightedMetricIndex), dtype=np.float64)
            if WeightedMetricIndex.TTC in weighted_set:
                weighted[WeightedMetricIndex.TTC] = self._time_to_collision(
                    simulated, coords, areas, scene,
                )[0]
            if WeightedMetricIndex.LANE_KEEPING in weighted_set:
                weighted[WeightedMetricIndex.LANE_KEEPING] = self._lane_keeping(
                    coords, areas, scene,
                )[0]
            if WeightedMetricIndex.COMFORT in weighted_set:
                weighted[WeightedMetricIndex.COMFORT] = self._history_comfort(
                    simulated, scene,
                )[0]

        except Exception as exc:
            raise RuntimeError(
                f"Failed to simulate reference trajectory for scene {scene.log_name}/{scene.scene_token}"
            ) from exc

        return _GTSimResult(progress=progress, multi_metrics=multi, weighted_metrics=weighted)

    def _simulate_and_score_gt(
        self,
        scene: SceneContext,
        *,
        multi_indices: list[int] | None = None,
        weighted_indices: list[int] | None = None,
    ) -> _GTSimResult | None:
        """Simulate GT once for analysis-only side-channel metrics.

        This must not be read as the official v1 scoring reference path. GT remains
        available so diagnostics, open-loop comparisons, and optional debug tooling
        can coexist with the explicit PDM reference context.
        """
        return self._simulate_and_score_reference(
            scene.gt_trajectory,
            scene,
            multi_indices=multi_indices,
            weighted_indices=weighted_indices,
        )

    def _simulate_and_score_pdm(
        self,
        scene: SceneContext,
        *,
        multi_indices: Iterable[int | MultiMetricIndex] | None = None,
        weighted_indices: Iterable[int | WeightedMetricIndex] | None = None,
    ) -> _GTSimResult | None:
        """Simulate the explicit PDM trajectory used as official v1 reference context."""
        return self._simulate_and_score_reference(
            scene.pdm_trajectory,
            scene,
            multi_indices=multi_indices,
            weighted_indices=weighted_indices,
        )

    # ── Human penalty filter ──────────────────────────────────────────

    _MULTI_METRIC_NAMES = {
        MultiMetricIndex.NO_COLLISION: "no_at_fault_collisions",
        MultiMetricIndex.DRIVABLE_AREA: "drivable_area_compliance",
        MultiMetricIndex.DRIVING_DIRECTION: "driving_direction_compliance",
        MultiMetricIndex.TRAFFIC_LIGHT: "traffic_light_compliance",
    }
    _WEIGHTED_METRIC_NAMES = {
        WeightedMetricIndex.PROGRESS: "ego_progress",
        WeightedMetricIndex.TTC: "time_to_collision",
        WeightedMetricIndex.LANE_KEEPING: "lane_keeping",
        WeightedMetricIndex.COMFORT: "history_comfort",
    }

    def _apply_human_penalty_from_gt(
        self,
        multi_metrics: np.ndarray,
        weighted_metrics: np.ndarray,
        gt_result: _GTSimResult,
    ) -> list[str]:
        overridden: list[str] = []
        for idx, name in self._MULTI_METRIC_NAMES.items():
            if gt_result.multi_metrics[idx] == 0.0:
                multi_metrics[:, idx] = 1.0
                overridden.append(name)
        for idx, name in self._WEIGHTED_METRIC_NAMES.items():
            if idx == WeightedMetricIndex.PROGRESS:
                continue
            if gt_result.weighted_metrics[idx] == 0.0:
                weighted_metrics[:, idx] = 1.0
                overridden.append(name)
        return overridden

    # ── RL continuous: safety layer ───────────────────────────────────

    @staticmethod
    def _penetration_direction(ego_heading: float, ego_centroid, intersection_centroid) -> str:
        """Determine penetration direction: intersection centroid relative to ego center in ego frame."""
        dx = intersection_centroid.x - ego_centroid.x
        dy = intersection_centroid.y - ego_centroid.y
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        local_x = dx * cos_h - dy * sin_h  # longitudinal (forward)
        local_y = dx * sin_h + dy * cos_h  # lateral (left+)

        angle = np.arctan2(local_y, local_x)
        if -np.pi / 4 <= angle <= np.pi / 4:
            return "front"
        elif np.pi / 4 < angle <= 3 * np.pi / 4:
            return "left"
        elif -3 * np.pi / 4 <= angle < -np.pi / 4:
            return "right"
        else:
            return "rear"

    def _collision_metrics(
        self,
        simulated_states: np.ndarray,
        ego_polygons: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig,
    ) -> dict:
        """NC continuous score + raw collision geometry + per-step collision info.

        Returns dict with:
            "nc": np.ndarray (batch_size,) — existing NC continuous [0,1]
            "max_collision_overlap": np.ndarray (batch_size,) — worst-frame overlap_area/ego_area [0,1]
            "max_collision_penetration_distance": np.ndarray (batch_size,) — sqrt(overlap_area), linear-scale proxy (meters)
            "per_step_collisions": list[list[dict|None]] — per (batch, step) collision info
        """
        del rl_config

        batch_size = len(ego_polygons)
        num_steps = ego_polygons.shape[1]

        collision_records: list[dict[str, tuple[float, float]]] = [{} for _ in range(batch_size)]
        forgiven: list[set[str]] = [set(scene.collided_track_ids) for _ in range(batch_size)]
        max_overlaps = np.zeros(batch_size, dtype=np.float64)
        max_penetration_dists = np.zeros(batch_size, dtype=np.float64)
        per_step_collisions: list[list[dict | None]] = [[None] * num_steps for _ in range(batch_size)]

        for time_idx in range(num_steps):
            occupancy_map = self._get_occupancy_map(scene, time_idx)
            if occupancy_map is None:
                continue
            for pi in range(batch_size):
                ego_poly = ego_polygons[pi, time_idx]
                if not occupancy_map.intersects(ego_poly)[0]:
                    continue
                ego_area = float(ego_poly.area)
                if ego_area < 1e-9:
                    continue

                for token in occupancy_map.get_colliding_tokens(ego_poly):
                    if RED_LIGHT_TOKEN_PREFIX in token or token in forgiven[pi]:
                        continue
                    track_polygon = occupancy_map[token]

                    if token not in collision_records[pi]:
                        collision_type = self._classify_collision_type(
                            simulated_states[pi, time_idx], ego_poly,
                            track_polygon, token, scene, time_idx,
                        )
                        if collision_type in (CollisionType.ACTIVE_REAR_BUMPER, CollisionType.STOPPED_EGO_OPEN):
                            forgiven[pi].add(token)
                            continue
                        at_fault_floor = self._collision_penalty(
                            collision_type, ego_areas[pi, time_idx], token, scene,
                        )
                        if at_fault_floor >= 1.0:
                            forgiven[pi].add(token)
                            continue
                        collision_records[pi][token] = (at_fault_floor, 1.0)

                    intersection_geom = ego_poly.intersection(track_polygon)
                    overlap = intersection_geom.area
                    if overlap < 1e-12:
                        # Degenerate touch (LineString/Point contact) — skip
                        continue
                    severity = min(overlap / ego_area, 1.0)
                    max_overlaps[pi] = max(max_overlaps[pi], severity)
                    penetration = float(np.sqrt(overlap))
                    max_penetration_dists[pi] = max(max_penetration_dists[pi], penetration)
                    floor, cum_prod = collision_records[pi][token]
                    collision_records[pi][token] = (floor, cum_prod * (1.0 - severity))

                    # Record per-step collision (keep worst at this step)
                    if per_step_collisions[pi][time_idx] is None or penetration > per_step_collisions[pi][time_idx]["penetration"]:
                        ego_heading = float(simulated_states[pi, time_idx, StateIndex.HEADING])
                        direction = self._penetration_direction(
                            ego_heading, ego_poly.centroid, intersection_geom.centroid,
                        )
                        per_step_collisions[pi][time_idx] = {
                            "direction": direction,
                            "penetration": round(penetration, 2),
                        }

        scores = np.ones(batch_size, dtype=np.float64)
        for pi in range(batch_size):
            for token, (at_fault_floor, cum_prod) in collision_records[pi].items():
                penalty = at_fault_floor + (1.0 - at_fault_floor) * cum_prod
                scores[pi] = min(scores[pi], penalty)
        return {
            "nc": scores,
            "max_collision_overlap": max_overlaps,
            "max_collision_penetration_distance": max_penetration_dists,
            "per_step_collisions": per_step_collisions,
        }

    _DAC_STRUCTURAL_LAYERS = frozenset({"roadblock", "intersection"})

    def _dac_continuous(
        self, ego_polygons: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        del rl_config
        dm = scene.drivable_area_map
        eligible = frozenset(
            i for i, (tok, typ) in enumerate(zip(dm.tokens, dm.types))
            if tok in scene.route_lane_ids or typ in self._DAC_STRUCTURAL_LAYERS
        )

        batch_size, num_steps = ego_polygons.shape[:2]
        scores = np.ones(batch_size, dtype=np.float64)

        if dm._tree is None:
            return np.zeros(batch_size, dtype=np.float64)

        # Fallback: if no eligible polygons (empty route), use full drivable union
        use_drivable_fallback = len(eligible) == 0
        drivable_union = dm.drivable_union if use_drivable_fallback else None

        for proposal_idx in range(batch_size):
            for time_idx in range(num_steps):
                ego_poly = ego_polygons[proposal_idx, time_idx]
                ego_area = float(ego_poly.area)
                if ego_area < 1e-9:
                    continue

                if use_drivable_fallback:
                    if drivable_union is None or drivable_union.is_empty:
                        scores[proposal_idx] = 0.0
                        break
                    inside = float(ego_poly.intersection(drivable_union).area)
                else:
                    nearby = dm._tree.query(ego_poly, predicate="intersects")
                    local = [j for j in nearby.tolist() if j in eligible]
                    if not local:
                        scores[proposal_idx] = 0.0
                        break
                    local_union = unary_union([dm._polygons[j] for j in local])
                    inside = float(ego_poly.intersection(local_union).area)

                coverage = min(inside / ego_area, 1.0)
                scores[proposal_idx] = min(scores[proposal_idx], coverage)
                if scores[proposal_idx] == 0.0:
                    break
        return scores

    def _obstacle_distance_series(
        self, ego_polygons: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Per-step nearest obstacle distance for each proposal. Shape: (B, T)."""
        batch_size, num_steps = ego_polygons.shape[:2]
        margin = rl_config.obstacle_clearance_margin
        dist_series = np.full((batch_size, num_steps), margin, dtype=np.float64)

        for time_idx in range(num_steps):
            occupancy_map = self._get_occupancy_map(scene, time_idx)
            if occupancy_map is None or occupancy_map._tree is None or len(occupancy_map) == 0:
                continue
            for pi in range(batch_size):
                ego_poly = ego_polygons[pi, time_idx]
                nearby_idx = occupancy_map._tree.query(ego_poly.buffer(margin), predicate="intersects")
                if len(nearby_idx) == 0:
                    continue
                valid = np.array([
                    RED_LIGHT_TOKEN_PREFIX not in occupancy_map._tokens[j] for j in nearby_idx
                ])
                if not valid.any():
                    continue
                nearby_polys = occupancy_map._polygons[nearby_idx[valid]]
                dists = shapely.distance(
                    np.full(len(nearby_polys), ego_poly, dtype=object), nearby_polys,
                )
                if len(dists) > 0:
                    dist_series[pi, time_idx] = float(np.clip(np.min(dists), 0.0, margin))
        return dist_series

    def _min_obstacle_distance(
        self, ego_polygons: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Min distance from ego to nearest obstacle across all frames. Returns meters."""
        return self._obstacle_distance_series(ego_polygons, scene, rl_config).min(axis=1)

    def _mean_obstacle_distance_within(
        self, ego_polygons: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Mean distance to obstacles within margin, averaged across frames.

        Frames where no obstacle is within range (dist == margin cap) are excluded.
        Fallback to margin if no obstacles within range at any frame.
        """
        batch_size, _ = ego_polygons.shape[:2]
        margin = rl_config.obstacle_clearance_margin
        dist_series = self._obstacle_distance_series(ego_polygons, scene, rl_config)
        # Strict < margin: exclude frames with no obstacle in range (capped at margin)
        masked = np.where(dist_series < margin, dist_series, np.nan)
        means = np.nanmean(masked, axis=1)
        means = np.where(np.isnan(means), margin, means)
        return means.astype(np.float64)

    def _half_lane_width(
        self, scene: SceneContext,
    ) -> float:
        """Perpendicular distance from centerline to ego's lane polygon boundary.

        Uses a ray perpendicular to the centerline tangent at ego position to find
        the intersection with the lane polygon boundary. Falls back to shortest
        distance if the perpendicular ray does not intersect.
        """
        ego_x = scene.ego_state[StateIndex.X]
        ego_y = scene.ego_state[StateIndex.Y]
        ego_pt = Point(ego_x, ego_y)
        cl_ls = scene.centerline.linestring
        proj_dist = cl_ls.project(ego_pt)
        cl_pt = cl_ls.interpolate(proj_dist)

        # Compute tangent direction via finite difference
        eps = 0.1
        length = cl_ls.length
        p_before = cl_ls.interpolate(max(proj_dist - eps, 0.0))
        p_after = cl_ls.interpolate(min(proj_dist + eps, length))
        tx = p_after.x - p_before.x
        ty = p_after.y - p_before.y
        tn = max(np.hypot(tx, ty), 1e-9)
        # Normal direction (perpendicular to tangent)
        nx, ny = -ty / tn, tx / tn

        lane_layers = {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR}
        dm = scene.drivable_area_map
        for token, layer_name in zip(dm.tokens, dm.types):
            if _layer_name_to_enum(layer_name) not in lane_layers:
                continue
            poly = dm[token]
            if not poly.contains(ego_pt):
                continue
            # Try perpendicular ray intersection with polygon boundary
            ray_len = 20.0  # generous ray length
            ray_line = LineString([
                (cl_pt.x - nx * ray_len, cl_pt.y - ny * ray_len),
                (cl_pt.x + nx * ray_len, cl_pt.y + ny * ray_len),
            ])
            boundary_intersection = ray_line.intersection(poly.boundary)
            if not boundary_intersection.is_empty:
                # Distance from centerline point to nearest boundary intersection
                half_w = cl_pt.distance(boundary_intersection)
                if half_w > 0.1:
                    return max(half_w, 0.5)
            # Fallback: shortest distance (less accurate on curves)
            return max(cl_pt.distance(poly.boundary), 0.5)

        # Fallback: centerline to full drivable boundary
        drivable = dm.drivable_union
        if drivable.is_empty:
            return 2.0
        return max(cl_pt.distance(drivable.boundary), 0.5)

    def _boundary_distance_series_raw(
        self, ego_coords: np.ndarray, scene: SceneContext,
    ) -> np.ndarray:
        """Per-step signed min distance from ego corners to drivable boundary. Shape: (B, T).

        Positive = inside drivable area (clearance to boundary).
        Negative = outside drivable area (penetration depth).
        Zero = exactly on boundary.
        """
        corners = ego_coords[:, :, :4, :]  # (B, T, 4, 2)
        B, T = corners.shape[:2]

        drivable = scene.drivable_area_map.drivable_union
        if drivable.is_empty:
            return np.zeros((B, T), dtype=np.float64)
        boundary = drivable.boundary

        flat_points = shapely.points(corners.reshape(-1, 2))
        flat_inside = shapely.contains(drivable, flat_points)
        flat_dists = shapely.distance(flat_points, boundary)
        # Signed: positive inside, negative outside
        flat_dists[~flat_inside] = -flat_dists[~flat_inside]
        dists = flat_dists.reshape(B, T, 4).min(axis=2)
        return dists

    def _boundary_distance_series(
        self, ego_coords: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Per-step signed min distance from ego corners to drivable boundary. Shape: (B, T).

        Positive = inside drivable area (clearance), negative = outside (penetration).
        Uses the signed raw version for full information.
        """
        return self._boundary_distance_series_raw(ego_coords, scene)

    def _min_boundary_distance(
        self, ego_coords: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Min signed distance from ego corner to drivable area boundary.

        Positive = always inside drivable area (min clearance).
        Negative = went off-road (max penetration depth).
        """
        return self._boundary_distance_series_raw(ego_coords, scene).min(axis=1)

    def _ddc_continuous(
        self, ego_coords: np.ndarray, ego_areas: np.ndarray,
        scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        oncoming_progress = np.zeros(centers.shape[:2], dtype=np.float64)
        oncoming_progress[:, 1:] = np.linalg.norm(centers[:, 1:] - centers[:, :-1], axis=-1)

        not_oncoming = ~ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        oncoming_progress[not_oncoming] = 0.0

        horizon_steps = max(
            int(round(rl_config.driving_direction_horizon / self._dt(scene))), 1,
        )
        rolling_progress = np.zeros_like(oncoming_progress)
        for time_idx in range(oncoming_progress.shape[1]):
            start_idx = max(0, time_idx - horizon_steps)
            rolling_progress[:, time_idx] = oncoming_progress[:, start_idx: time_idx + 1].sum(axis=1)

        max_progress = rolling_progress.max(axis=1)
        lo = rl_config.driving_direction_compliance_threshold
        hi = rl_config.driving_direction_violation_threshold
        if hi <= lo:
            return np.where(max_progress <= lo, 1.0, 0.0).astype(np.float64)
        return np.clip(1.0 - (max_progress - lo) / (hi - lo), 0.0, 1.0)

    def _tlc_continuous(
        self, ego_polygons: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        batch_size = len(ego_polygons)
        min_distances = np.full(batch_size, np.inf, dtype=np.float64)
        margin = rl_config.tlc_margin

        for time_idx in range(ego_polygons.shape[1]):
            red_light_map = self._get_red_light_map(scene, time_idx)
            if red_light_map is None:
                continue
            for proposal_idx in range(batch_size):
                if min_distances[proposal_idx] == 0.0:
                    continue
                ego_poly = ego_polygons[proposal_idx, time_idx]
                if red_light_map.intersects(ego_poly)[0]:
                    for token in red_light_map.get_colliding_tokens(ego_poly):
                        if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                            min_distances[proposal_idx] = 0.0
                            break
                else:
                    buffered = ego_poly.buffer(margin)
                    if red_light_map.intersects(buffered)[0]:
                        for token in red_light_map.get_colliding_tokens(buffered):
                            if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                                dist = ego_poly.distance(red_light_map[token])
                                min_distances[proposal_idx] = min(min_distances[proposal_idx], dist)

        min_distances = np.where(np.isinf(min_distances), margin * 2.0, min_distances)
        return np.clip(min_distances / margin, 0.0, 1.0)

    # ── RL continuous: performance layer ──────────────────────────────

    def _ep_continuous(
        self, ego_coords: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
        *,
        reference_masked_progress: float | None = None,
    ) -> np.ndarray:
        """Continuous ego progress normalized by a caller-provided denominator.

        Fallback chain:
            1. reference_masked_progress param (typically official PDM masked progress)
            2. progress_distance_threshold (5m, no reference available)

        This helper no longer assigns any implicit official-reference meaning to GT.
        If a caller wants to analyze GT-normalized progress for diagnostics, it must
        resolve and pass that denominator explicitly as an analysis choice.
        """
        raw_progress = self._progress(ego_coords, scene)
        denominator = reference_masked_progress
        if denominator is not None and denominator > rl_config.progress_distance_threshold:
            return np.clip(raw_progress / denominator, 0.0, 1.0)
        return np.clip(raw_progress / rl_config.progress_distance_threshold, 0.0, 1.0)

    def _ttc_continuous(
        self, simulated_states: np.ndarray, ego_coords: np.ndarray,
        ego_areas: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        dt = self._dt(scene)
        total_forward_steps = max(int(round(rl_config.future_collision_horizon / dt)), 1)
        sample_stride = max(int(round(0.33 / dt)), 1)
        future_offsets = np.arange(0, total_forward_steps, sample_stride, dtype=int)
        if len(future_offsets) == 0:
            future_offsets = np.asarray([0], dtype=int)

        speeds = np.hypot(
            simulated_states[..., StateIndex.VELOCITY_X],
            simulated_states[..., StateIndex.VELOCITY_Y],
        )
        dxy_per_second = np.stack(
            [
                np.cos(simulated_states[..., StateIndex.HEADING]) * speeds,
                np.sin(simulated_states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        projected_coords = np.repeat(ego_coords[:, :, None, :, :], len(future_offsets), axis=2)
        for offset_idx, future_offset in enumerate(future_offsets):
            projected_coords[:, :, offset_idx] += (
                dxy_per_second[:, :, None, :] * (float(future_offset) * dt)
            )
        projected_polygons = coords_to_polygons(projected_coords)

        batch_size = len(simulated_states)
        first_violation_time = np.full(batch_size, np.inf, dtype=np.float64)
        collided_track_ids = [set(scene.collided_track_ids) for _ in range(batch_size)]
        num_time_steps = simulated_states.shape[1]

        for time_idx in range(num_time_steps):
            for offset_idx, future_offset in enumerate(future_offsets):
                current_time_idx = time_idx + future_offset
                occupancy_map = self._get_occupancy_map(scene, current_time_idx)
                if occupancy_map is None:
                    continue
                for proposal_idx in range(batch_size):
                    if first_violation_time[proposal_idx] < np.inf:
                        continue
                    if speeds[proposal_idx, time_idx] < rl_config.stopped_speed_threshold:
                        continue
                    ego_polygon = projected_polygons[proposal_idx, time_idx, offset_idx]
                    if not occupancy_map.intersects(ego_polygon)[0]:
                        continue
                    for token in occupancy_map.get_colliding_tokens(ego_polygon):
                        if RED_LIGHT_TOKEN_PREFIX in token or token in collided_track_ids[proposal_idx]:
                            continue
                        ego_in_intersection = scene.drivable_area_map.is_in_layer(
                            simulated_states[proposal_idx, time_idx, [StateIndex.X, StateIndex.Y]],
                            SemanticMapLayer.INTERSECTION,
                        )
                        if self._is_ttc_violation(
                            simulated_states[proposal_idx, time_idx],
                            occupancy_map[token],
                            ego_areas[proposal_idx, time_idx],
                            ego_in_intersection,
                        ):
                            first_violation_time[proposal_idx] = float(future_offset) * dt
                        else:
                            collided_track_ids[proposal_idx].add(token)

        ttc_seconds = np.where(np.isinf(first_violation_time), rl_config.ttc_horizon, first_violation_time)
        return np.clip(ttc_seconds / rl_config.ttc_horizon, 0.0, 1.0)

    def _lk_continuous(
        self, ego_coords: np.ndarray, ego_areas: np.ndarray,
        scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        batch_size = len(ego_coords)
        scores = np.ones(batch_size, dtype=np.float64)

        for proposal_idx in range(batch_size):
            deviations = []
            for time_idx in range(ego_coords.shape[1]):
                if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                    continue
                dev = Point(*centers[proposal_idx, time_idx]).distance(scene.centerline.linestring)
                deviations.append(dev)
            if not deviations:
                continue
            mean_dev = float(np.mean(deviations))
            lo = rl_config.lane_keeping_deviation
            hi = rl_config.lane_keeping_max_deviation
            if mean_dev <= lo:
                scores[proposal_idx] = 1.0
            elif mean_dev >= hi:
                scores[proposal_idx] = 0.0
            else:
                scores[proposal_idx] = 1.0 - (mean_dev - lo) / (hi - lo)
        return scores

    def _signed_lateral_offset_at(self, center_xy, centerline_ls) -> float:
        """Compute signed lateral offset. Positive = right of centerline."""
        ego_pt = Point(*center_xy)
        proj_dist = centerline_ls.project(ego_pt)
        cl_pt = centerline_ls.interpolate(proj_dist)
        dx = ego_pt.x - cl_pt.x
        dy = ego_pt.y - cl_pt.y
        # Tangent via finite difference
        eps = 0.1
        length = centerline_ls.length
        p_before = centerline_ls.interpolate(max(proj_dist - eps, 0.0))
        p_after = centerline_ls.interpolate(min(proj_dist + eps, length))
        tangent_x = p_after.x - p_before.x
        tangent_y = p_after.y - p_before.y
        # Cross product of tangent × (ego - cl_pt): positive when ego is left of centerline.
        # Negated so that the returned value is positive = right of centerline.
        cross = tangent_x * dy - tangent_y * dx
        norm = max(np.hypot(tangent_x, tangent_y), 1e-9)
        return -cross / norm

    def _lateral_offset_signed(
        self, ego_coords: np.ndarray, ego_areas: np.ndarray, scene: SceneContext,
    ) -> np.ndarray:
        """Mean signed lateral offset from centerline per proposal. Shape: (batch,)"""
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        batch_size = len(ego_coords)
        result = np.zeros(batch_size, dtype=np.float64)
        cl_ls = scene.centerline.linestring

        for proposal_idx in range(batch_size):
            offsets = []
            for time_idx in range(ego_coords.shape[1]):
                if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                    continue
                offsets.append(
                    self._signed_lateral_offset_at(centers[proposal_idx, time_idx], cl_ls)
                )
            if offsets:
                result[proposal_idx] = float(np.mean(offsets))
        return result

    def _lateral_offset_change(
        self, ego_coords: np.ndarray, ego_areas: np.ndarray, scene: SceneContext,
    ) -> np.ndarray:
        """Signed offset at last non-intersection timestep minus first. Shape: (batch,)"""
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        batch_size = len(ego_coords)
        result = np.zeros(batch_size, dtype=np.float64)
        cl_ls = scene.centerline.linestring

        for proposal_idx in range(batch_size):
            first_offset = None
            last_offset = None
            for time_idx in range(ego_coords.shape[1]):
                if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                    continue
                offset = self._signed_lateral_offset_at(centers[proposal_idx, time_idx], cl_ls)
                if first_offset is None:
                    first_offset = offset
                last_offset = offset
            if first_offset is not None and last_offset is not None:
                result[proposal_idx] = last_offset - first_offset
        return result

    def _centerline_geometry(
        self, ego_coords: np.ndarray, ego_areas: np.ndarray, scene: SceneContext,
    ) -> dict:
        """Centerline-relative geometry diagnostics.

        Note: start_signed/end_signed are the signed lateral offsets at the
        first/last NON-INTERSECTION timestep, not the trajectory start/end.
        Intersection steps are excluded because centerline distance is ill-defined there.
        """
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        batch_size, num_steps = centers.shape[:2]
        cl_ls = scene.centerline.linestring
        start_signed = np.zeros(batch_size, dtype=np.float64)
        end_signed = np.zeros(batch_size, dtype=np.float64)
        mean_dist = np.zeros(batch_size, dtype=np.float64)
        max_dist = np.zeros(batch_size, dtype=np.float64)
        local_points = self._sample_local_centerline_points(scene)

        for proposal_idx in range(batch_size):
            signed_offsets = []
            abs_dists = []
            for time_idx in range(num_steps):
                if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                    continue
                center_xy = centers[proposal_idx, time_idx]
                signed = self._signed_lateral_offset_at(center_xy, cl_ls)
                signed_offsets.append(signed)
                abs_dists.append(Point(*center_xy).distance(cl_ls))
            if signed_offsets:
                start_signed[proposal_idx] = float(signed_offsets[0])
                end_signed[proposal_idx] = float(signed_offsets[-1])
                mean_dist[proposal_idx] = float(np.mean(abs_dists))
                max_dist[proposal_idx] = float(np.max(abs_dists))

        return {
            "start_signed": start_signed,
            "end_signed": end_signed,
            "mean_distance": mean_dist,
            "max_distance": max_dist,
            "local_centerline_points": local_points,
        }

    def _sample_local_centerline_points(
        self, scene: SceneContext, num_points: int = 6, horizon_m: float = 30.0,
    ) -> list[list[float]]:
        cl = scene.centerline
        ego_x = float(scene.ego_state[StateIndex.X])
        ego_y = float(scene.ego_state[StateIndex.Y])
        ego_h = float(scene.ego_state[StateIndex.HEADING])
        ego_pt = Point(ego_x, ego_y)
        start_progress = cl.project(ego_pt)
        sample_ds = np.linspace(start_progress, min(start_progress + horizon_m, cl.length), num_points)
        cos_h = np.cos(-ego_h)
        sin_h = np.sin(-ego_h)
        points = []
        for d in sample_ds:
            p = cl.interpolate(float(d))
            dx = float(p.x) - ego_x
            dy = float(p.y) - ego_y
            local_x = dx * cos_h - dy * sin_h
            local_y = dx * sin_h + dy * cos_h
            points.append([local_x, local_y])
        return points

    def _boundary_geometry(
        self,
        ego_coords: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig,
        *,
        dists: np.ndarray | None = None,
    ) -> dict:
        if dists is None:
            dists = self._boundary_distance_series_raw(ego_coords, scene)
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        sides: list[str | None] = []
        nearest_sides: list[str | None] = []
        nearest_dists: list[float] = []
        cl_ls = scene.centerline.linestring
        drivable = scene.drivable_area_map.drivable_union
        boundary = drivable.boundary if not drivable.is_empty else None

        # Current ego position (global)
        ego_pos = Point(*scene.ego_state[:2])

        # Per-step boundary side (only when off-road, i.e., signed dist < 0)
        B, T = dists.shape
        sides_series: list[list[str | None]] = [[None] * T for _ in range(B)]
        for pi in range(B):
            for ti in range(T):
                if dists[pi, ti] < 0.0:
                    signed = self._signed_lateral_offset_at(centers[pi, ti], cl_ls)
                    sides_series[pi][ti] = "right" if signed > 0 else "left"

        for proposal_idx in range(len(ego_coords)):
            # Original side: ego lateral offset at argmin step
            closest_idx = int(np.argmin(dists[proposal_idx]))
            center_xy = centers[proposal_idx, closest_idx]
            signed = self._signed_lateral_offset_at(center_xy, cl_ls)
            if signed > 0:
                sides.append("right")
            elif signed < 0:
                sides.append("left")
            else:
                sides.append(None)

            # Nearest boundary at current ego position (use ego corners for distance)
            if boundary is not None:
                nearest_pt = nearest_points(ego_pos, boundary)[1]
                # Compute current ego corners from ego_state
                ego_state_arr = scene.ego_state.reshape(1, 1, -1)  # (1, 1, state_dim)
                ego_now_coords = state_to_coords(ego_state_arr, self._vehicle)  # (1, 1, 5, 2)
                ego_now_corners = ego_now_coords[0, 0, :4, :]  # (4, 2)
                corner_pts = shapely.points(ego_now_corners)
                corner_dists = shapely.distance(corner_pts, boundary)
                nearest_dists.append(float(corner_dists.min()))
                nearest_signed = self._signed_lateral_offset_at(
                    np.array([nearest_pt.x, nearest_pt.y]), cl_ls
                )
                if nearest_signed > 0:
                    nearest_sides.append("right")
                elif nearest_signed < 0:
                    nearest_sides.append("left")
                else:
                    nearest_sides.append(None)
            else:
                nearest_sides.append(None)
                nearest_dists.append(0.0)

        return {
            "distances": dists,
            "start": dists[:, 0],
            "end": dists[:, -1],
            "min": dists.min(axis=1),
            "mean": dists.mean(axis=1),
            "side": sides,
            "nearest_side": nearest_sides,
            "nearest_distance": nearest_dists,
            "sides_series": sides_series,
        }

    def _obstacle_geometry(
        self, ego_polygons: np.ndarray, ego_coords: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> dict:
        dists = self._obstacle_distance_series(ego_polygons, scene, rl_config)
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        cl_ls = scene.centerline.linestring
        sides: list[str | None] = []
        closest_steps: list[int | None] = []
        for proposal_idx in range(len(ego_polygons)):
            closest_idx = int(np.argmin(dists[proposal_idx]))
            closest_steps.append(closest_idx)
            center_xy = centers[proposal_idx, closest_idx]
            signed = self._signed_lateral_offset_at(center_xy, cl_ls)
            if signed > 0:
                sides.append("right")
            elif signed < 0:
                sides.append("left")
            else:
                sides.append(None)
        return {
            "distances": dists,
            "start": dists[:, 0],
            "end": dists[:, -1],
            "min": dists.min(axis=1),
            "side": sides,
            "closest_step": closest_steps,
        }

    def _topology_occupancy(self, ego_areas: np.ndarray) -> dict:
        in_intersection = ego_areas[:, :, EgoAreaIndex.IN_INTERSECTION].astype(bool)
        oncoming = ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC].astype(bool)
        non_drivable = ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].astype(bool)
        multiple_lanes = ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES].astype(bool)
        # Guard against zero-length trajectories (axis=1 mean of empty → NaN)
        n_steps = ego_areas.shape[1]
        if n_steps == 0:
            batch_size = ego_areas.shape[0]
            zeros = np.zeros(batch_size, dtype=np.float64)
            return {
                "in_intersection_flags": in_intersection,
                "oncoming_flags": oncoming,
                "non_drivable_flags": non_drivable,
                "multiple_lanes_flags": multiple_lanes,
                "in_intersection_fraction": zeros,
                "oncoming_fraction": zeros,
                "non_drivable_fraction": zeros,
                "multiple_lanes_fraction": zeros,
            }
        return {
            "in_intersection_flags": in_intersection,
            "oncoming_flags": oncoming,
            "non_drivable_flags": non_drivable,
            "multiple_lanes_flags": multiple_lanes,
            "in_intersection_fraction": in_intersection.mean(axis=1),
            "oncoming_fraction": oncoming.mean(axis=1),
            "non_drivable_fraction": non_drivable.mean(axis=1),
            "multiple_lanes_fraction": multiple_lanes.mean(axis=1),
        }

    def _hc_continuous(
        self, simulated_states: np.ndarray, scene: SceneContext,
    ) -> np.ndarray:
        """Continuous comfort metric using max violation ratio.

        Prepends ego_past_states (if available) to improve acceleration/jerk
        estimation at the start of the simulated trajectory.
        """
        scores = np.ones(len(simulated_states), dtype=np.float64)
        dt = self._dt(scene)
        past = scene.ego_past_states
        for proposal_idx in range(len(simulated_states)):
            states = simulated_states[proposal_idx]
            if past is not None and len(past) > 0:
                states = np.concatenate([past, states], axis=0)
            time_points_s = np.arange(len(states), dtype=np.float64) * dt
            scores[proposal_idx] = ego_comfort_violation(states, time_points_s)
        return scores

    # ── Geometric / time utilities ────────────────────────────────────

    def _is_ttc_violation(
        self, ego_state: np.ndarray, track_polygon: BaseGeometry,
        ego_area: np.ndarray, ego_in_intersection: bool,
    ) -> bool:
        agent_xy = np.asarray(track_polygon.centroid.coords[0], dtype=np.float64)
        relative_angle = self._get_agent_relative_angle(ego_state, agent_xy)
        is_ahead = relative_angle < np.deg2rad(30.0)
        is_behind = relative_angle > np.deg2rad(150.0)
        return bool(
            is_ahead
            or (
                (
                    ego_area[EgoAreaIndex.MULTIPLE_LANES]
                    or ego_area[EgoAreaIndex.NON_DRIVABLE_AREA]
                    or ego_in_intersection
                )
                and not is_behind
            )
        )

    @staticmethod
    def _get_agent_relative_angle(ego_state: np.ndarray, agent_xy: np.ndarray) -> float:
        agent_vector = agent_xy - ego_state[[StateIndex.X, StateIndex.Y]]
        norm = np.linalg.norm(agent_vector)
        if norm < 1e-12:
            return 0.0
        ego_heading = float(ego_state[StateIndex.HEADING])
        ego_vector = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        dot_product = np.clip(np.dot(ego_vector, agent_vector / norm), -1.0, 1.0)
        return float(np.arccos(dot_product))

    def _estimate_track_speed(self, scene: SceneContext, token: str, time_idx: int) -> float:
        local_idx = self._local_time_idx(scene, time_idx)
        base_map = scene.observation.get_occupancy_map(local_idx)
        if base_map is None or token not in base_map.token_to_idx:
            return 0.0
        base_centroid = np.asarray(base_map[token].centroid.coords[0], dtype=np.float64)
        dt = self._dt(scene)
        num_steps = len(scene.observation.global_to_local_idcs)
        for offset in (1, -1):
            neighbor_time = time_idx + offset
            if not 0 <= neighbor_time < num_steps:
                continue
            neighbor_idx = self._local_time_idx(scene, neighbor_time)
            neighbor_map = scene.observation.get_occupancy_map(neighbor_idx)
            if neighbor_map is None or token not in neighbor_map.token_to_idx:
                continue
            neighbor_centroid = np.asarray(neighbor_map[token].centroid.coords[0], dtype=np.float64)
            return float(np.linalg.norm(neighbor_centroid - base_centroid) / dt)
        return 0.0

    @staticmethod
    def _points_in_map_tokens(
        points: np.ndarray,
        drivable_area_map: DrivableMap,
        include_layers: Iterable[SemanticMapLayer],
    ) -> np.ndarray:
        include_values = set(include_layers)
        selected_indices = [
            idx for idx, layer_name in enumerate(drivable_area_map.types)
            if _layer_name_to_enum(layer_name) in include_values
        ]
        if not selected_indices:
            return np.zeros((0, *points.shape[:-1]), dtype=bool)
        flat_points = points.reshape(-1, 2)
        membership = np.zeros((len(selected_indices), len(flat_points)), dtype=bool)
        x_coords = flat_points[:, 0]
        y_coords = flat_points[:, 1]
        for row_idx, polygon_idx in enumerate(selected_indices):
            membership[row_idx] = shapely.contains_xy(
                drivable_area_map[drivable_area_map.tokens[polygon_idx]],
                x_coords, y_coords,
            )
        return membership.reshape((len(selected_indices), *points.shape[:-1]))

    def _points_in_route_lanes(
        self, points: np.ndarray, drivable_area_map: DrivableMap,
        route_lane_ids: Sequence[str],
    ) -> np.ndarray:
        route_lane_id_set = set(route_lane_ids)
        if not route_lane_id_set:
            return np.zeros(points.shape[:-1], dtype=bool)
        selected_indices = [
            idx for idx, (token, layer_name) in enumerate(
                zip(drivable_area_map.tokens, drivable_area_map.types)
            )
            if token in route_lane_id_set
            and _layer_name_to_enum(layer_name) in {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR}
        ]
        if not selected_indices:
            return np.zeros(points.shape[:-1], dtype=bool)
        flat_points = points.reshape(-1, 2)
        x_coords = flat_points[:, 0]
        y_coords = flat_points[:, 1]
        membership = np.zeros((len(selected_indices), len(flat_points)), dtype=bool)
        for row_idx, polygon_idx in enumerate(selected_indices):
            membership[row_idx] = shapely.contains_xy(
                drivable_area_map[drivable_area_map.tokens[polygon_idx]],
                x_coords, y_coords,
            )
        return membership.any(axis=0).reshape(points.shape[:-1])

    @staticmethod
    def _relative_point_in_ego_frame(ego_state: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
        dx = point_xy[0] - ego_state[StateIndex.X]
        dy = point_xy[1] - ego_state[StateIndex.Y]
        theta = float(ego_state[StateIndex.HEADING])
        return np.asarray(
            [dx * np.cos(theta) + dy * np.sin(theta), -dx * np.sin(theta) + dy * np.cos(theta)],
            dtype=np.float64,
        )

    def _is_track_behind_ego(self, ego_state: np.ndarray, track_polygon: BaseGeometry) -> bool:
        """Match NavSim's is_agent_behind: relative angle > 150°."""
        agent_xy = np.asarray(track_polygon.centroid.coords[0], dtype=np.float64)
        return bool(self._get_agent_relative_angle(ego_state, agent_xy) > np.deg2rad(150.0))

    @staticmethod
    def _dt(scene: SceneContext) -> float:
        return float(scene.observation.interval_time)

    @staticmethod
    def _local_time_idx(scene: SceneContext, time_idx: int) -> int:
        global_to_local = scene.observation.global_to_local_idcs
        clamped = min(max(time_idx, 0), len(global_to_local) - 1)
        return int(global_to_local[clamped])

    def _get_occupancy_map(self, scene: SceneContext, time_idx: int) -> OccupancyMap | None:
        return scene.observation.get_occupancy_map(self._local_time_idx(scene, time_idx))

    def _get_red_light_map(self, scene: SceneContext, time_idx: int) -> OccupancyMap | None:
        return scene.observation.get_red_light_map(self._local_time_idx(scene, time_idx))


def _layer_name_to_enum(layer_name: str) -> SemanticMapLayer | None:
    return _LAYER_NAME_TO_ENUM.get(_normalize_layer_name(layer_name))
