"""Collision metrics shared by the driving scorers."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from ..enums import CollisionType, EgoAreaIndex, SemanticMapLayer, StateIndex
from ..geometry import coords_to_polygons
from ..types import SceneContext
from . import scene_geometry
from .config import RLScorerConfig

RED_LIGHT_TOKEN_PREFIX = "red_light"
_COLLISION_STOPPED_THRESHOLD = 0.05


def classify_collision_type(
    ego_state: np.ndarray,
    ego_polygon: BaseGeometry,
    track_polygon: BaseGeometry,
    token: str,
    scene: SceneContext,
    time_idx: int,
) -> CollisionType:
    ego_speed = np.hypot(ego_state[StateIndex.VELOCITY_X], ego_state[StateIndex.VELOCITY_Y])
    if ego_speed <= _COLLISION_STOPPED_THRESHOLD:
        return CollisionType.STOPPED_EGO_OPEN
    track_speed = scene.track_speeds.get(token)
    if track_speed is None:
        track_speed = estimate_track_speed(scene, token, time_idx)
    if track_speed <= _COLLISION_STOPPED_THRESHOLD:
        return CollisionType.STOPPED_TRACK_OPEN
    if is_track_behind_ego(ego_state, track_polygon):
        return CollisionType.ACTIVE_REAR_BUMPER
    front_bumper = LineString([ego_polygon.exterior.coords[0], ego_polygon.exterior.coords[3]])
    if front_bumper.intersects(track_polygon):
        return CollisionType.ACTIVE_FRONT_BUMPER
    return CollisionType.ACTIVE_LATERAL


def collision_penalty(
    collision_type: CollisionType, ego_area: np.ndarray, token: str, scene: SceneContext
) -> float:
    """At-fault collision penalty.

    Uses scene.track_object_types to distinguish agents from static objects.
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


def no_at_fault_collision(
    simulated_states: np.ndarray,
    ego_polygons: np.ndarray,
    ego_areas: np.ndarray,
    scene: SceneContext,
) -> np.ndarray:
    scores = np.ones(len(simulated_states), dtype=np.float64)
    collided_track_ids = [set(scene.collided_track_ids) for _ in range(len(simulated_states))]
    for time_idx in range(simulated_states.shape[1]):
        occupancy_map = scene_geometry.get_occupancy_map(scene, time_idx)
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
                collision_type = classify_collision_type(
                    simulated_states[proposal_idx, time_idx],
                    ego_polygon,
                    track_polygon,
                    token,
                    scene,
                    time_idx,
                )
                at_fault_score = collision_penalty(
                    collision_type, ego_areas[proposal_idx, time_idx], token, scene
                )
                scores[proposal_idx] = min(scores[proposal_idx], at_fault_score)
                if at_fault_score >= 1.0:
                    collided_track_ids[proposal_idx].add(token)
    return scores


def time_to_collision(
    simulated_states: np.ndarray,
    ego_coords: np.ndarray,
    ego_areas: np.ndarray,
    scene: SceneContext,
    future_collision_horizon: float = 1.0,
    stopped_speed_threshold: float = 0.005,
) -> np.ndarray:
    dt = scene_geometry.dt(scene)
    total_forward_steps = max(int(round(future_collision_horizon / dt)), 1)
    sample_stride = max(int(round(0.33 / dt)), 1)
    future_offsets = np.arange(0, total_forward_steps, sample_stride, dtype=int)
    if len(future_offsets) == 0:
        future_offsets = np.asarray([0], dtype=int)
    speeds = np.hypot(
        simulated_states[..., StateIndex.VELOCITY_X], simulated_states[..., StateIndex.VELOCITY_Y]
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
        projected_coords[:, :, offset_idx] += dxy_per_second[:, :, None, :] * (
            float(future_offset) * dt
        )
    projected_polygons = coords_to_polygons(projected_coords)
    scores = np.ones(len(simulated_states), dtype=np.float64)
    collided_track_ids = [set(scene.collided_track_ids) for _ in range(len(simulated_states))]
    num_time_steps = simulated_states.shape[1]
    for time_idx in range(num_time_steps):
        for offset_idx, future_offset in enumerate(future_offsets):
            current_time_idx = time_idx + future_offset
            occupancy_map = scene_geometry.get_occupancy_map(scene, current_time_idx)
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
                    if is_ttc_violation(
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


def penetration_direction(ego_heading: float, ego_centroid, intersection_centroid) -> str:
    """Determine penetration direction: intersection centroid relative to ego center in ego frame."""
    dx = intersection_centroid.x - ego_centroid.x
    dy = intersection_centroid.y - ego_centroid.y
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)
    local_x = dx * cos_h - dy * sin_h
    local_y = dx * sin_h + dy * cos_h
    angle = np.arctan2(local_y, local_x)
    if -np.pi / 4 <= angle <= np.pi / 4:
        return "front"
    elif np.pi / 4 < angle <= 3 * np.pi / 4:
        return "left"
    elif -3 * np.pi / 4 <= angle < -np.pi / 4:
        return "right"
    else:
        return "rear"


def collision_metrics(
    simulated_states: np.ndarray,
    ego_polygons: np.ndarray,
    ego_areas: np.ndarray,
    scene: SceneContext,
    *,
    include_details: bool = False,
) -> dict:
    """NC continuous score + raw collision geometry + per-step collision info.

    Returns dict with:
        "nc": np.ndarray (batch_size,) — existing NC continuous [0,1]
        "max_collision_overlap": np.ndarray (batch_size,) — worst-frame overlap_area/ego_area [0,1]
        "max_collision_penetration_distance": np.ndarray (batch_size,) — sqrt(overlap_area), linear-scale proxy (meters)
        "per_step_collisions": list[list[dict|None]] — per (batch, step) collision info
    """
    batch_size = len(ego_polygons)
    num_steps = ego_polygons.shape[1]
    collision_records: list[dict[str, tuple[float, float]]] = [{} for _ in range(batch_size)]
    forgiven: list[set[str]] = [set(scene.collided_track_ids) for _ in range(batch_size)]
    max_overlaps = np.zeros(batch_size, dtype=np.float64)
    max_penetration_dists = np.zeros(batch_size, dtype=np.float64)
    per_step_collisions: list[list[dict | None]] = [
        [None] * num_steps if include_details else [] for _ in range(batch_size)
    ]
    for time_idx in range(num_steps):
        occupancy_map = scene_geometry.get_occupancy_map(scene, time_idx)
        if occupancy_map is None:
            continue
        for pi in range(batch_size):
            ego_poly = ego_polygons[pi, time_idx]
            if not occupancy_map.intersects(ego_poly)[0]:
                continue
            ego_area = float(ego_poly.area)
            if ego_area < 1e-09:
                continue
            for token in occupancy_map.get_colliding_tokens(ego_poly):
                if RED_LIGHT_TOKEN_PREFIX in token or token in forgiven[pi]:
                    continue
                track_polygon = occupancy_map[token]
                if token not in collision_records[pi]:
                    collision_type = classify_collision_type(
                        simulated_states[pi, time_idx],
                        ego_poly,
                        track_polygon,
                        token,
                        scene,
                        time_idx,
                    )
                    if collision_type in (
                        CollisionType.ACTIVE_REAR_BUMPER,
                        CollisionType.STOPPED_EGO_OPEN,
                    ):
                        forgiven[pi].add(token)
                        continue
                    at_fault_floor = collision_penalty(
                        collision_type, ego_areas[pi, time_idx], token, scene
                    )
                    if at_fault_floor >= 1.0:
                        forgiven[pi].add(token)
                        continue
                    collision_records[pi][token] = (at_fault_floor, 1.0)
                intersection_geom = ego_poly.intersection(track_polygon)
                overlap = intersection_geom.area
                if overlap < 1e-12:
                    continue
                severity = min(overlap / ego_area, 1.0)
                max_overlaps[pi] = max(max_overlaps[pi], severity)
                penetration = float(np.sqrt(overlap))
                max_penetration_dists[pi] = max(max_penetration_dists[pi], penetration)
                floor, cum_prod = collision_records[pi][token]
                collision_records[pi][token] = (floor, cum_prod * (1.0 - severity))
                if include_details and (
                    per_step_collisions[pi][time_idx] is None
                    or penetration > per_step_collisions[pi][time_idx]["penetration"]
                ):
                    ego_heading = float(simulated_states[pi, time_idx, StateIndex.HEADING])
                    direction = penetration_direction(
                        ego_heading, ego_poly.centroid, intersection_geom.centroid
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


def obstacle_distance_series(
    ego_polygons: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig
) -> np.ndarray:
    """Per-step nearest obstacle distance for each proposal. Shape: (B, T)."""
    batch_size, num_steps = ego_polygons.shape[:2]
    margin = rl_config.obstacle_clearance_margin
    dist_series = np.full((batch_size, num_steps), margin, dtype=np.float64)
    for time_idx in range(num_steps):
        occupancy_map = scene_geometry.get_occupancy_map(scene, time_idx)
        if occupancy_map is None or occupancy_map._tree is None or len(occupancy_map) == 0:
            continue
        for pi in range(batch_size):
            ego_poly = ego_polygons[pi, time_idx]
            nearby_idx = occupancy_map._tree.query(ego_poly.buffer(margin), predicate="intersects")
            if len(nearby_idx) == 0:
                continue
            valid = np.array(
                [RED_LIGHT_TOKEN_PREFIX not in occupancy_map._tokens[j] for j in nearby_idx]
            )
            if not valid.any():
                continue
            nearby_polys = occupancy_map._polygons[nearby_idx[valid]]
            dists = shapely.distance(
                np.full(len(nearby_polys), ego_poly, dtype=object), nearby_polys
            )
            if len(dists) > 0:
                dist_series[pi, time_idx] = float(np.clip(np.min(dists), 0.0, margin))
    return dist_series


def ttc_continuous(
    simulated_states: np.ndarray,
    ego_coords: np.ndarray,
    ego_areas: np.ndarray,
    scene: SceneContext,
    rl_config: RLScorerConfig,
) -> np.ndarray:
    dt = scene_geometry.dt(scene)
    total_forward_steps = max(int(round(rl_config.future_collision_horizon / dt)), 1)
    sample_stride = max(int(round(0.33 / dt)), 1)
    future_offsets = np.arange(0, total_forward_steps, sample_stride, dtype=int)
    if len(future_offsets) == 0:
        future_offsets = np.asarray([0], dtype=int)
    speeds = np.hypot(
        simulated_states[..., StateIndex.VELOCITY_X], simulated_states[..., StateIndex.VELOCITY_Y]
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
        projected_coords[:, :, offset_idx] += dxy_per_second[:, :, None, :] * (
            float(future_offset) * dt
        )
    projected_polygons = coords_to_polygons(projected_coords)
    batch_size = len(simulated_states)
    first_violation_time = np.full(batch_size, np.inf, dtype=np.float64)
    collided_track_ids = [set(scene.collided_track_ids) for _ in range(batch_size)]
    num_time_steps = simulated_states.shape[1]
    for time_idx in range(num_time_steps):
        for offset_idx, future_offset in enumerate(future_offsets):
            current_time_idx = time_idx + future_offset
            occupancy_map = scene_geometry.get_occupancy_map(scene, current_time_idx)
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
                    if is_ttc_violation(
                        simulated_states[proposal_idx, time_idx],
                        occupancy_map[token],
                        ego_areas[proposal_idx, time_idx],
                        ego_in_intersection,
                    ):
                        first_violation_time[proposal_idx] = float(future_offset) * dt
                    else:
                        collided_track_ids[proposal_idx].add(token)
    ttc_seconds = np.where(
        np.isinf(first_violation_time), rl_config.ttc_horizon, first_violation_time
    )
    return np.clip(ttc_seconds / rl_config.ttc_horizon, 0.0, 1.0)


def is_ttc_violation(
    ego_state: np.ndarray,
    track_polygon: BaseGeometry,
    ego_area: np.ndarray,
    ego_in_intersection: bool,
) -> bool:
    agent_xy = np.asarray(track_polygon.centroid.coords[0], dtype=np.float64)
    relative_angle = get_agent_relative_angle(ego_state, agent_xy)
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
            and (not is_behind)
        )
    )


def get_agent_relative_angle(ego_state: np.ndarray, agent_xy: np.ndarray) -> float:
    agent_vector = agent_xy - ego_state[[StateIndex.X, StateIndex.Y]]
    norm = np.linalg.norm(agent_vector)
    if norm < 1e-12:
        return 0.0
    ego_heading = float(ego_state[StateIndex.HEADING])
    ego_vector = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    dot_product = np.clip(np.dot(ego_vector, agent_vector / norm), -1.0, 1.0)
    return float(np.arccos(dot_product))


def estimate_track_speed(scene: SceneContext, token: str, time_idx: int) -> float:
    local_idx = scene_geometry.local_time_idx(scene, time_idx)
    base_map = scene.observation.get_occupancy_map(local_idx)
    if base_map is None or token not in base_map.token_to_idx:
        return 0.0
    base_centroid = np.asarray(base_map[token].centroid.coords[0], dtype=np.float64)
    dt = scene_geometry.dt(scene)
    num_steps = len(scene.observation.global_to_local_idcs)
    for offset in (1, -1):
        neighbor_time = time_idx + offset
        if not 0 <= neighbor_time < num_steps:
            continue
        neighbor_idx = scene_geometry.local_time_idx(scene, neighbor_time)
        neighbor_map = scene.observation.get_occupancy_map(neighbor_idx)
        if neighbor_map is None or token not in neighbor_map.token_to_idx:
            continue
        neighbor_centroid = np.asarray(neighbor_map[token].centroid.coords[0], dtype=np.float64)
        return float(np.linalg.norm(neighbor_centroid - base_centroid) / dt)
    return 0.0


def is_track_behind_ego(ego_state: np.ndarray, track_polygon: BaseGeometry) -> bool:
    """Match NavSim's is_agent_behind: relative angle > 150°."""
    agent_xy = np.asarray(track_polygon.centroid.coords[0], dtype=np.float64)
    return bool(get_agent_relative_angle(ego_state, agent_xy) > np.deg2rad(150.0))
