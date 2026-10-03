"""Convert standard NAVSIM v1 MetricCache objects with their original pickle types."""

from __future__ import annotations

import lzma
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from grava_sim_engine.core.geometry import PDMPath
from grava_sim_engine.core.observation import PDMObservation
from grava_sim_engine.core.occupancy import DrivableMap, OccupancyMap
from grava_sim_engine.core.types import SceneContext

if TYPE_CHECKING:
    from navsim.planning.metric_caching.metric_cache import MetricCache

_AGENT_TYPE_NAMES = {"VEHICLE", "PEDESTRIAN", "BICYCLE", "EGO"}


def read_metric_cache(path: Path) -> MetricCache:
    with lzma.open(path, "rb") as stream:
        return pickle.load(stream)


def _build_ego_state_array(ego_state: Any) -> np.ndarray:
    state = np.zeros(11, dtype=np.float64)
    rear_axle = ego_state.rear_axle
    dynamic = ego_state.dynamic_car_state
    state[0] = rear_axle.x
    state[1] = rear_axle.y
    state[2] = rear_axle.heading
    state[3] = dynamic.rear_axle_velocity_2d.x
    state[4] = dynamic.rear_axle_velocity_2d.y
    state[5] = dynamic.rear_axle_acceleration_2d.x
    state[6] = dynamic.rear_axle_acceleration_2d.y
    state[7] = ego_state.tire_steering_angle
    state[8] = dynamic.tire_steering_rate
    state[9] = dynamic.angular_velocity
    state[10] = dynamic.angular_acceleration
    return state


def _extract_pdm_trajectory_xy(
    metric_cache: Any, ego_state_array: np.ndarray, num_future_steps: int = 40
) -> np.ndarray | None:
    """Extract MetricCache.trajectory as ego-relative PDM reference waypoints.

    MetricCache.trajectory is the explicit PDM closed-loop reference used by V1 scoring.
    We keep sampled[0:num_future_steps+1] so t=0 remains the ego anchor.
    Includes heading (x, y, heading) to avoid heading derivation errors.
    """
    trajectory = metric_cache.trajectory
    sampled = list(trajectory.get_sampled_trajectory())
    end_idx = min(num_future_steps + 1, len(sampled))
    global_xyh = np.array(
        [[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading] for s in sampled[0:end_idx]],
        dtype=np.float64,
    )
    ego_x = ego_state_array[0]
    ego_y = ego_state_array[1]
    ego_h = ego_state_array[2]
    cos_h = np.cos(-ego_h)
    sin_h = np.sin(-ego_h)
    dx = global_xyh[:, 0] - ego_x
    dy = global_xyh[:, 1] - ego_y
    local_x = dx * cos_h - dy * sin_h
    local_y = dx * sin_h + dy * cos_h
    local_heading = global_xyh[:, 2] - ego_h
    return np.stack([local_x, local_y, local_heading], axis=1)


def _convert_observation(source: Any) -> PDMObservation:
    observation = PDMObservation(
        num_steps=len(source._occupancy_maps), interval_time=source._sample_interval
    )
    for time_idx, item in enumerate(source._occupancy_maps):
        tokens = [str(token) for token in item._tokens]
        geometries = np.asarray(item._geometries, dtype=object)
        observation._occupancy_maps[time_idx] = OccupancyMap(tokens, geometries) if tokens else None
        red_indices = [index for index, token in enumerate(tokens) if token.startswith("red_light")]
        red_tokens = [tokens[index] for index in red_indices]
        observation._red_light_maps[time_idx] = (
            OccupancyMap(red_tokens, geometries[red_indices]) if red_tokens else None
        )
    observation._global_to_local_idcs = [int(index) for index in source._global_to_local_idcs]
    observation._observation_sample_res = source._observation_sample_res
    return observation


def metric_cache_to_scene_context(metric_cache: MetricCache, scene_token: str) -> SceneContext:
    ego_state = _build_ego_state_array(metric_cache.ego_state)
    map_data = metric_cache.drivable_area_map
    objects = metric_cache.observation.unique_objects
    return SceneContext(
        scene_token=scene_token,
        log_name=Path(metric_cache.file_path).parts[-4],
        ego_state=ego_state,
        ego_past_states=np.zeros((0, 11), dtype=np.float64),
        observation=_convert_observation(metric_cache.observation),
        drivable_area_map=DrivableMap(
            tokens=[str(token) for token in map_data._tokens],
            types=[layer.name for layer in map_data._map_types],
            polygons=list(map_data._geometries),
        ),
        route_lane_ids={str(lane_id) for lane_id in metric_cache.route_lane_ids},
        centerline=PDMPath(
            np.array([[point.x, point.y] for point in metric_cache.centerline._discrete_path])
        ),
        collided_track_ids={str(token) for token in metric_cache.observation.collided_track_ids},
        pdm_trajectory=_extract_pdm_trajectory_xy(metric_cache, ego_state),
        track_object_types={
            str(token): "agent" if obj.tracked_object_type.name in _AGENT_TYPE_NAMES else "static"
            for token, obj in objects.items()
        },
        track_speeds={
            str(token): float(obj.velocity.magnitude())
            if obj.tracked_object_type.name in _AGENT_TYPE_NAMES
            else 0.0
            for token, obj in objects.items()
        },
    )
