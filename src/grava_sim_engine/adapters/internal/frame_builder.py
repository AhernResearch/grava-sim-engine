"""Build SceneContext objects from internal shard frame.json files."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from grava_common.coordinates import CoordinateConverter
from grava_sim_engine.core.enums import SemanticMapLayer, StateIndex
from grava_sim_engine.core.geometry import PDMPath, normalize_angle
from grava_sim_engine.core.observation import PDMObservation
from grava_sim_engine.core.occupancy import DrivableMap, OccupancyMap
from grava_sim_engine.core.types import KeyActionObstacle, SceneContext

from .builder import build_centerline_from_info, build_drivable_area_map_from_info


FUTURE_OBSTACLE_TRACKS_KEY = "future_obstacle_tracks"
FUTURE_OBSTACLE_TRACKS_COORDINATE_FRAME = "nous_ego_current_frame"


@dataclass(frozen=True)
class _FutureObstacleState:
    rel_time_s: float
    polygon_coords: np.ndarray
    velocity_xy: np.ndarray
    object_type: str
    speed_mps: float


class InternalShardFrameSceneContextBuilder:
    """Convert ego-local internal shard frames into sim-engine SceneContext objects."""

    def __init__(
        self,
        *,
        horizon_waypoints: int = 8,
        num_observation_steps: int = 41,
        interval_time: float = 0.1,
    ) -> None:
        self.horizon_waypoints = horizon_waypoints
        self.num_observation_steps = int(num_observation_steps)
        self.interval_time = float(interval_time)

    def build(
        self,
        frame_data: dict[str, Any],
        *,
        log_name: str | None = None,
        scene_token: str | None = None,
        info_data: dict[str, Any] | None = None,
    ) -> SceneContext:
        """Build a SceneContext from one internal shard frame.json payload."""

        resolved_log_name = str(log_name or frame_data.get("case_id") or "internal")
        raw_scene_token = scene_token
        if raw_scene_token is None:
            raw_scene_token = frame_data.get("timestamp") or frame_data.get("image_id")
        if raw_scene_token is None or raw_scene_token == "":
            raise ValueError("frame_data must contain timestamp or image_id")
        resolved_scene_token = str(raw_scene_token)

        centerline = self._build_centerline(frame_data, info_data=info_data)
        drivable_area_map = self._build_drivable_area_map(frame_data, info_data=info_data)
        route_lane_ids = self._build_route_lane_ids(centerline, drivable_area_map)
        gt_trajectory = self._build_future_trajectory(frame_data)
        candidate_trajectory = self._build_candidate_trajectory(frame_data, info_data=info_data)
        pdm_trajectory = gt_trajectory
        if pdm_trajectory is None:
            pdm_trajectory = self._sample_path_waypoints(centerline.discrete_path)

        observation, track_object_types, track_speeds = self._build_observation(frame_data)
        key_action_obstacles = self._build_key_action_obstacles(frame_data)

        return SceneContext(
            scene_token=resolved_scene_token,
            log_name=resolved_log_name,
            ego_state=self._build_ego_state(frame_data),
            ego_past_states=np.zeros((0, StateIndex.size()), dtype=np.float64),
            observation=observation,
            drivable_area_map=drivable_area_map,
            route_lane_ids=route_lane_ids,
            centerline=centerline,
            gt_trajectory=gt_trajectory,
            gt_progress=self._trajectory_progress(gt_trajectory, centerline),
            pdm_trajectory=pdm_trajectory,
            track_object_types=track_object_types,
            track_speeds=track_speeds,
            key_action_obstacles=key_action_obstacles,
            candidate_trajectory=candidate_trajectory,
            candidate_progress=self._trajectory_progress(candidate_trajectory, centerline),
        )

    def _build_ego_state(self, frame_data: dict[str, Any]) -> np.ndarray:
        ego_car = frame_data.get("ego_car") or {}
        state = np.zeros(StateIndex.size(), dtype=np.float64)

        velocity = ego_car.get("velocity")
        if isinstance(velocity, dict):
            vx, vy = CoordinateConverter.nous_to_nuplan(
                _float(velocity.get("x"), 0.0),
                _float(velocity.get("y"), 0.0),
            )
            state[StateIndex.VELOCITY_X] = vx
            state[StateIndex.VELOCITY_Y] = vy
        else:
            state[StateIndex.VELOCITY_X] = _float(ego_car.get("speed_mps"), 0.0)

        acceleration = ego_car.get("acceleration")
        if isinstance(acceleration, dict):
            ax, ay = CoordinateConverter.nous_to_nuplan(
                _float(acceleration.get("x"), 0.0),
                _float(acceleration.get("y"), 0.0),
            )
            state[StateIndex.ACCELERATION_X] = ax
            state[StateIndex.ACCELERATION_Y] = ay
        else:
            state[StateIndex.ACCELERATION_X] = _float(acceleration, 0.0)

        return state

    def _build_observation(
        self,
        frame_data: dict[str, Any],
    ) -> tuple[PDMObservation, dict[str, str], dict[str, float]]:
        if isinstance(frame_data.get(FUTURE_OBSTACLE_TRACKS_KEY), dict):
            return self._build_future_obstacle_tracks_observation(frame_data)
        return self._build_current_frame_observation(frame_data)

    def _build_current_frame_observation(
        self,
        frame_data: dict[str, Any],
    ) -> tuple[PDMObservation, dict[str, str], dict[str, float]]:
        dynamic_tokens: list[str] = []
        dynamic_coords: list[np.ndarray] = []
        dynamic_velocities: list[tuple[float, float]] = []
        track_object_types: dict[str, str] = {}
        track_speeds: dict[str, float] = {}

        for index, obstacle in enumerate(frame_data.get("obstacles") or []):
            bbox = obstacle.get("bbox_3d")
            if not _is_sequence(bbox) or len(bbox) < 7:
                continue

            center_x, center_y = CoordinateConverter.nous_to_nuplan(
                _float(bbox[0], 0.0),
                _float(bbox[1], 0.0),
            )
            length = max(_float(bbox[3], 0.0), 0.1)
            width = max(_float(bbox[4], 0.0), 0.1)
            velocity_x, velocity_y = self._obstacle_velocity(obstacle)
            heading = self._obstacle_heading(bbox, velocity_x, velocity_y)
            token = str(obstacle.get("id") or obstacle.get("track_id") or f"obstacle_{index}")

            dynamic_tokens.append(token)
            dynamic_coords.append(
                np.asarray(
                    _box_corners(center_x, center_y, length, width, heading),
                    dtype=np.float64,
                )
            )
            dynamic_velocities.append((velocity_x, velocity_y))
            track_object_types[token] = self._track_object_type(obstacle)
            track_speeds[token] = _float(
                obstacle.get("speed_mps"),
                math.hypot(velocity_x, velocity_y),
            )

        observation = PDMObservation(
            num_steps=self.num_observation_steps,
            interval_time=self.interval_time,
        )
        observation.update(
            static_coords=np.empty((0, 4, 2), dtype=np.float64),
            static_tokens=[],
            dynamic_coords=np.asarray(dynamic_coords, dtype=np.float64)
            if dynamic_coords
            else np.empty((0, 4, 2), dtype=np.float64),
            dynamic_tokens=dynamic_tokens,
            dynamic_velocities=np.asarray(dynamic_velocities, dtype=np.float64)
            if dynamic_velocities
            else np.empty((0, 2), dtype=np.float64),
            red_light_coords=np.empty((0, 4, 2), dtype=np.float64),
            red_light_tokens=[],
        )
        return observation, track_object_types, track_speeds

    def _build_future_obstacle_tracks_observation(
        self,
        frame_data: dict[str, Any],
    ) -> tuple[PDMObservation, dict[str, str], dict[str, float]]:
        future_tracks = frame_data.get(FUTURE_OBSTACLE_TRACKS_KEY) or {}
        coordinate_frame = future_tracks.get("coordinate_frame")
        if coordinate_frame != FUTURE_OBSTACLE_TRACKS_COORDINATE_FRAME:
            raise ValueError(
                f"Unsupported {FUTURE_OBSTACLE_TRACKS_KEY}.coordinate_frame: "
                f"{coordinate_frame!r}"
            )

        records_by_token: dict[str, list[_FutureObstacleState]] = {}
        for track_index, track in enumerate(future_tracks.get("tracks") or []):
            if not isinstance(track, dict):
                continue
            token = str(track.get("id") or track.get("track_id") or f"future_track_{track_index}")
            object_type = str(track.get("object_type") or track.get("type") or "agent")
            default_speed = _float(track.get("speed_mps"), 0.0)
            states: list[_FutureObstacleState] = []
            for state in track.get("states") or []:
                parsed_state = self._parse_future_obstacle_state(
                    state,
                    object_type=object_type,
                    default_speed=default_speed,
                )
                if parsed_state is not None:
                    states.append(parsed_state)
            if states:
                records_by_token[token] = sorted(states, key=lambda item: item.rel_time_s)

        occupancy_maps: list[OccupancyMap | None] = []
        track_object_types: dict[str, str] = {}
        track_speeds: dict[str, float] = {}

        for step_idx in range(self.num_observation_steps):
            timestamp_s = step_idx * self.interval_time
            tokens: list[str] = []
            polygons: list[Polygon] = []
            for token, records in records_by_token.items():
                coords, record = self._interpolate_future_obstacle_state(records, timestamp_s)
                if coords is None or record is None:
                    continue
                polygon = Polygon(coords)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if polygon.is_empty:
                    continue
                tokens.append(token)
                polygons.append(polygon)
                track_object_types.setdefault(token, record.object_type)
                track_speeds.setdefault(token, record.speed_mps)
            if tokens:
                occupancy_maps.append(
                    OccupancyMap(tokens=tokens, polygons=np.asarray(polygons, dtype=object))
                )
            else:
                occupancy_maps.append(None)

        observation = PDMObservation(
            num_steps=self.num_observation_steps,
            interval_time=self.interval_time,
        )
        observation._occupancy_maps = occupancy_maps
        observation._red_light_maps = [None] * self.num_observation_steps
        observation._global_to_local_idcs = list(range(self.num_observation_steps))
        observation._observation_sample_res = 1
        return observation, track_object_types, track_speeds

    def _parse_future_obstacle_state(
        self,
        state: Any,
        *,
        object_type: str,
        default_speed: float,
    ) -> _FutureObstacleState | None:
        if not isinstance(state, dict):
            return None
        rel_time_s = _float(state.get("rel_time_s"), math.nan)
        if not math.isfinite(rel_time_s):
            return None

        polygon_points = (
            state.get("polygon")
            or state.get("polygon_points")
            or state.get("corners")
        )
        polygon_coords = _points_to_nuplan_xy(polygon_points or [])
        if len(polygon_coords) < 3:
            return None

        velocity_x, velocity_y = self._future_state_velocity(state.get("velocity"))
        return _FutureObstacleState(
            rel_time_s=rel_time_s,
            polygon_coords=np.asarray(polygon_coords, dtype=np.float64),
            velocity_xy=np.asarray([velocity_x, velocity_y], dtype=np.float64),
            object_type=object_type,
            speed_mps=_float(state.get("speed_mps"), default_speed),
        )

    def _future_state_velocity(self, velocity: Any) -> tuple[float, float]:
        if isinstance(velocity, dict):
            return CoordinateConverter.nous_to_nuplan(
                _float(velocity.get("x"), 0.0),
                _float(velocity.get("y"), 0.0),
            )
        if _is_sequence(velocity) and len(velocity) >= 2:
            return CoordinateConverter.nous_to_nuplan(
                _float(velocity[0], 0.0),
                _float(velocity[1], 0.0),
            )
        return 0.0, 0.0

    def _interpolate_future_obstacle_state(
        self,
        records: list[_FutureObstacleState],
        timestamp_s: float,
    ) -> tuple[np.ndarray | None, _FutureObstacleState | None]:
        first = records[0]
        if timestamp_s < first.rel_time_s - 1e-6:
            return None, None
        if len(records) == 1 or timestamp_s >= records[-1].rel_time_s:
            last = records[-1]
            delta_t = max(0.0, timestamp_s - last.rel_time_s)
            return last.polygon_coords + last.velocity_xy[None, :] * delta_t, last

        for prev_record, next_record in zip(records[:-1], records[1:]):
            if prev_record.rel_time_s <= timestamp_s <= next_record.rel_time_s:
                duration = next_record.rel_time_s - prev_record.rel_time_s
                if duration <= 1e-6:
                    return next_record.polygon_coords, next_record
                alpha = (timestamp_s - prev_record.rel_time_s) / duration
                coords = (
                    (1.0 - alpha) * prev_record.polygon_coords
                    + alpha * next_record.polygon_coords
                )
                return coords, prev_record if alpha < 0.5 else next_record

        return None, None

    def _obstacle_velocity(self, obstacle: dict[str, Any]) -> tuple[float, float]:
        velocity = obstacle.get("velocity")
        if isinstance(velocity, dict):
            return CoordinateConverter.nous_to_nuplan(
                _float(velocity.get("x"), 0.0),
                _float(velocity.get("y"), 0.0),
            )

        speed = _float(obstacle.get("speed_mps"), 0.0)
        return speed, 0.0

    def _obstacle_heading(
        self,
        bbox: Iterable[Any],
        velocity_x: float,
        velocity_y: float,
    ) -> float:
        if math.hypot(velocity_x, velocity_y) > 0.2:
            return math.atan2(velocity_y, velocity_x)

        raw_heading = _float(list(bbox)[6], 1.5 * math.pi)
        return _normalize_angle(raw_heading - 1.5 * math.pi)

    def _track_object_type(self, obstacle: dict[str, Any]) -> str:
        subtype = str(obstacle.get("sub_type") or obstacle.get("type") or "").upper()
        dynamic_types = {
            "CAR",
            "TRUCK",
            "BUS",
            "VEHICLE",
            "MOTORCYCLE",
            "CYCLIST",
            "BICYCLE",
            "TRICYCLE",
            "PEDESTRIAN",
        }
        if subtype in dynamic_types:
            return "agent"
        return "static"

    def _build_drivable_area_map(
        self,
        frame_data: dict[str, Any],
        *,
        info_data: dict[str, Any] | None = None,
    ) -> DrivableMap:
        tokens: list[str] = []
        polygons: list[Polygon] = []
        types: list[SemanticMapLayer] = []

        for index, lane in enumerate((frame_data.get("map") or {}).get("lanes") or []):
            polygon = self._lane_polygon(lane)
            if polygon is None:
                continue

            token = str(lane.get("id") or lane.get("lane_id") or f"lane_{index}")
            tokens.append(token)
            polygons.append(polygon)
            types.append(SemanticMapLayer.LANE)

        if not tokens and info_data is not None:
            return build_drivable_area_map_from_info(info_data)

        return DrivableMap(
            tokens=tokens,
            types=types,
            polygons=np.asarray(polygons, dtype=object),
        )

    def _lane_polygon(self, lane: dict[str, Any]) -> Polygon | None:
        left = _points_to_nuplan_xy(lane.get("left_boundary_points") or [])
        right = _points_to_nuplan_xy(lane.get("right_boundary_points") or [])
        if len(left) < 2 or len(right) < 2:
            return None

        polygon = Polygon(left + list(reversed(right)))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area < 1e-3:
            return None
        return polygon

    def _build_route_lane_ids(
        self,
        centerline: PDMPath,
        drivable_area_map: DrivableMap,
    ) -> set[str]:
        if len(drivable_area_map.tokens) == 0:
            return set()

        route_area = centerline.linestring.buffer(2.0)
        route_lane_ids = {
            str(token)
            for token in drivable_area_map.tokens
            if drivable_area_map[token].intersects(route_area)
        }
        if route_lane_ids:
            return route_lane_ids
        return set(str(token) for token in drivable_area_map.tokens)

    def _build_centerline(
        self,
        frame_data: dict[str, Any],
        *,
        info_data: dict[str, Any] | None = None,
    ) -> PDMPath:
        nav_path = self._nav_path_points(frame_data)
        if len(nav_path) >= 2:
            return PDMPath(np.asarray(nav_path, dtype=np.float64))

        lane_path = self._closest_lane_centerline(frame_data)
        if len(lane_path) >= 2:
            return PDMPath(np.asarray(lane_path, dtype=np.float64))

        if info_data is not None:
            try:
                return build_centerline_from_info(info_data)
            except ValueError:
                pass

        future_path = self._future_path_points(frame_data)
        if len(future_path) >= 2:
            return PDMPath(np.asarray(future_path, dtype=np.float64))

        raise ValueError("frame does not contain enough route, lane, or future trajectory points")

    def _nav_path_points(self, frame_data: dict[str, Any]) -> list[tuple[float, float]]:
        nav_paths_world = (frame_data.get("map") or {}).get("nav_paths_world") or {}
        return _dedupe_points(
            _points_to_nuplan_xy(_flatten_points(nav_paths_world.get("routing_points") or []))
        )

    def _closest_lane_centerline(self, frame_data: dict[str, Any]) -> list[tuple[float, float]]:
        best_path: list[tuple[float, float]] = []
        best_distance = math.inf

        for lane in (frame_data.get("map") or {}).get("lanes") or []:
            path = _dedupe_points(
                _points_to_nuplan_xy(
                    _flatten_points(lane.get("central_curve_segments") or [])
                )
            )
            if len(path) < 2:
                continue

            distance = min(math.hypot(x, y) for x, y in path)
            if distance < best_distance:
                best_distance = distance
                best_path = path

        return best_path

    def _future_path_points(self, frame_data: dict[str, Any]) -> list[tuple[float, float]]:
        future = (frame_data.get("ego_car") or {}).get("future_trajectory") or []
        return _dedupe_points(_points_to_nuplan_xy(future))

    def _build_future_trajectory(self, frame_data: dict[str, Any]) -> np.ndarray | None:
        future = (frame_data.get("ego_car") or {}).get("future_trajectory") or []
        points = _points_to_nuplan_xy(future)
        if len(points) < 2:
            return None

        if math.hypot(points[0][0], points[0][1]) < 0.2:
            points = points[1:]
        if not points:
            return None

        sampled = _pad_or_trim(points, self.horizon_waypoints)
        return np.asarray(sampled, dtype=np.float64)

    def _build_candidate_trajectory(
        self,
        frame_data: dict[str, Any],
        *,
        info_data: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        ego_attr_candidates: list[Any] = []
        for container in (
            frame_data.get("ego_car_attribute"),
            (frame_data.get("info") or {}).get("ego_car_attribute")
            if isinstance(frame_data.get("info"), dict)
            else None,
            info_data.get("ego_car_attribute") if isinstance(info_data, dict) else None,
        ):
            if isinstance(container, dict):
                ego_attr_candidates.append(container.get("candidate_trajectory_point"))

        for candidate in ego_attr_candidates:
            points = _points_to_nuplan_xy(_flatten_points(candidate or []))
            points = _dedupe_points(points)
            if len(points) >= 2:
                return np.asarray(points, dtype=np.float64)
        return None

    def _build_key_action_obstacles(self, frame_data: dict[str, Any]) -> list[KeyActionObstacle]:
        annotations: list[KeyActionObstacle] = []
        for index, obstacle in enumerate(frame_data.get("obstacles") or []):
            if not isinstance(obstacle, dict):
                continue
            label_raw = obstacle.get("label")
            try:
                label = int(label_raw)
            except (TypeError, ValueError):
                continue
            if label not in (0, 1):
                continue

            bbox = obstacle.get("bbox_3d")
            if not _is_sequence(bbox) or len(bbox) < 7:
                continue
            center_x, center_y = CoordinateConverter.nous_to_nuplan(
                _float(bbox[0], 0.0),
                _float(bbox[1], 0.0),
            )
            length = max(_float(bbox[3], 0.0), 0.1)
            width = max(_float(bbox[4], 0.0), 0.1)
            velocity_x, velocity_y = self._obstacle_velocity(obstacle)
            heading = self._obstacle_heading(bbox, velocity_x, velocity_y)
            token = str(obstacle.get("id") or obstacle.get("track_id") or f"obstacle_{index}")
            annotations.append(
                KeyActionObstacle(
                    token=token,
                    label=label,
                    polygon_coords=np.asarray(
                        _box_corners(center_x, center_y, length, width, heading),
                        dtype=np.float64,
                    ),
                    object_type=self._track_object_type(obstacle),
                    speed_mps=_float(obstacle.get("speed_mps"), math.hypot(velocity_x, velocity_y)),
                )
            )
        return annotations

    def _trajectory_progress(
        self,
        trajectory: np.ndarray | None,
        centerline: PDMPath,
    ) -> float | None:
        if trajectory is None:
            return None
        points = np.asarray(trajectory, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
            return None
        start_s = centerline.project(Point(0.0, 0.0))
        projected = [centerline.project(Point(float(x), float(y))) for x, y in points[:, :2]]
        return max(0.0, float(max(projected) - start_s))

    def _sample_path_waypoints(self, path: Iterable[tuple[float, float]]) -> np.ndarray | None:
        path = [(float(x), float(y)) for x, y in path]
        if len(path) < 2:
            return None

        ahead = [point for point in path if point[0] > 0.2]
        source = ahead if len(ahead) >= 2 else path
        sampled = _sample_by_arc_length(source, self.horizon_waypoints, spacing=2.5)
        return np.asarray(sampled, dtype=np.float64)


def load_frame_json(path: str | Path) -> dict[str, Any]:
    """Load one internal shard frame.json file."""

    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return data


def build_scene_context_from_frame(
    frame_data: dict[str, Any],
    *,
    log_name: str | None = None,
    scene_token: str | None = None,
    info_data: dict[str, Any] | None = None,
) -> SceneContext:
    """Build a SceneContext from one internal shard frame.json payload."""

    return InternalShardFrameSceneContextBuilder().build(
        frame_data,
        log_name=log_name,
        scene_token=scene_token,
        info_data=info_data,
    )


def build_future_trajectory_from_frame(
    frame_data: dict[str, Any],
    *,
    horizon_waypoints: int = 8,
) -> np.ndarray | None:
    """Build only the internal ego future GT trajectory from a shard frame."""

    return InternalShardFrameSceneContextBuilder(
        horizon_waypoints=horizon_waypoints,
    )._build_future_trajectory(frame_data)


def _float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _point_to_nuplan_xy(point: Any) -> tuple[float, float] | None:
    if isinstance(point, dict):
        if "points" in point:
            return None
        x_value = point.get("x")
        y_value = point.get("y")
    elif _is_sequence(point) and len(point) >= 2:
        x_value = point[0]
        y_value = point[1]
    else:
        return None

    x_nous = _float(x_value, math.nan)
    y_nous = _float(y_value, math.nan)
    if not math.isfinite(x_nous) or not math.isfinite(y_nous):
        return None
    x_nuplan, y_nuplan = CoordinateConverter.nous_to_nuplan(x_nous, y_nous)
    return x_nuplan, y_nuplan


def _points_to_nuplan_xy(points: Iterable[Any]) -> list[tuple[float, float]]:
    converted: list[tuple[float, float]] = []
    for point in points:
        converted_point = _point_to_nuplan_xy(point)
        if converted_point is not None:
            converted.append(converted_point)
    return converted


def _flatten_points(value: Any) -> list[Any]:
    points: list[Any] = []
    if isinstance(value, dict):
        if "points" in value:
            points.extend(_flatten_points(value["points"]))
        elif "x" in value and "y" in value:
            points.append(value)
        else:
            for child in value.values():
                points.extend(_flatten_points(child))
    elif isinstance(value, list):
        if len(value) >= 2 and all(isinstance(value[index], (int, float)) for index in (0, 1)):
            points.append(value)
        else:
            for child in value:
                points.extend(_flatten_points(child))
    return points


def _dedupe_points(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or math.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) > 1e-3:
            deduped.append(point)
    return deduped


def _pad_or_trim(
    points: list[tuple[float, float]],
    num_waypoints: int,
) -> list[tuple[float, float]]:
    if len(points) >= num_waypoints:
        return points[:num_waypoints]

    padded = list(points)
    while len(padded) < num_waypoints:
        padded.append(padded[-1])
    return padded


def _sample_by_arc_length(
    points: list[tuple[float, float]],
    num_waypoints: int,
    *,
    spacing: float,
) -> list[tuple[float, float]]:
    line = LineString(points)
    length = max(line.length, 0.0)
    if length <= 1e-6:
        return _pad_or_trim(points, num_waypoints)

    sampled = []
    for index in range(num_waypoints):
        distance = min((index + 1) * spacing, length)
        point = line.interpolate(distance)
        sampled.append((float(point.x), float(point.y)))
    return sampled


def _box_corners(
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    heading: float,
) -> list[tuple[float, float]]:
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    half_length = length / 2.0
    half_width = width / 2.0
    corners: list[tuple[float, float]] = []
    for local_x, local_y in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ):
        world_x = center_x + local_x * cos_h - local_y * sin_h
        world_y = center_y + local_x * sin_h + local_y * cos_h
        corners.append((world_x, world_y))
    return corners


def _normalize_angle(angle: float) -> float:
    return float(normalize_angle(angle))
