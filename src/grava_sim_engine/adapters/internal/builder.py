from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from grava_common.coordinates import CoordinateConverter
from shapely.geometry import LineString, Polygon

from grava_sim_engine.core.enums import SemanticMapLayer, StateIndex
from grava_sim_engine.core.geometry import PDMPath, normalize_angle
from grava_sim_engine.core.observation import PDMObservation
from grava_sim_engine.core.occupancy import DrivableMap
from grava_sim_engine.core.types import SceneContext


def load_info_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"info.json must contain an object, got {type(data).__name__}")
    return data


def build_scene_context_from_info(
    info_data: dict[str, Any],
    *,
    log_name: str,
    scene_token: str,
) -> SceneContext:
    return InternalSceneContextBuilder().build(info_data, log_name=log_name, scene_token=scene_token)


def build_drivable_area_map_from_info(info_data: dict[str, Any]) -> DrivableMap:
    """Build a lane-based drivable map from raw internal info/info.json data."""

    return InternalSceneContextBuilder().build_drivable_area_map(info_data)


def build_centerline_from_info(info_data: dict[str, Any]) -> PDMPath:
    """Build the routed centerline from raw internal info/info.json data."""

    return InternalSceneContextBuilder().build_centerline(info_data)


@dataclass
class _EgoTransform:
    position: tuple[float, float, float]
    rotation_world_to_nous: np.ndarray


class InternalSceneContextBuilder:
    """Convert internal raw info/info.json payloads into SceneContext."""

    def __init__(self, *, num_observation_steps: int = 41, interval_time: float = 0.1) -> None:
        self.num_observation_steps = int(num_observation_steps)
        self.interval_time = float(interval_time)

    def build(
        self,
        info_data: dict[str, Any],
        *,
        log_name: str,
        scene_token: str,
    ) -> SceneContext:
        transform = self._build_ego_transform(info_data)
        ego_state = self._build_ego_state(info_data)
        ego_past_states = self._build_ego_past_states(info_data, transform)

        lane_tokens, lane_polygons = self._build_lane_polygons(info_data, transform)
        if not lane_polygons:
            raise ValueError(f"No valid lane polygons for {log_name}/{scene_token}")
        drivable_area_map = self._lane_drivable_map(lane_tokens, lane_polygons)

        centerline = self._build_centerline(info_data, transform)
        route_lane_ids = self._match_route_lanes(lane_tokens, lane_polygons, centerline)
        observation, track_object_types, track_speeds = self._build_observation(info_data, transform)

        gt_trajectory = self._build_sparse_trajectory(
            (info_data.get("ego_car_attribute") or {}).get("future_track"),
            transform,
            prefer_time=False,
        )
        pdm_trajectory = self._build_reference_trajectory(info_data, transform)

        return SceneContext(
            scene_token=str(scene_token),
            log_name=str(log_name),
            ego_state=ego_state,
            ego_past_states=ego_past_states,
            observation=observation,
            drivable_area_map=drivable_area_map,
            route_lane_ids=route_lane_ids,
            centerline=centerline,
            gt_trajectory=gt_trajectory,
            pdm_trajectory=pdm_trajectory,
            track_object_types=track_object_types,
            track_speeds=track_speeds,
        )

    def build_drivable_area_map(self, info_data: dict[str, Any]) -> DrivableMap:
        transform = self._build_ego_transform(info_data)
        lane_tokens, lane_polygons = self._build_lane_polygons(info_data, transform)
        return self._lane_drivable_map(lane_tokens, lane_polygons)

    def build_centerline(self, info_data: dict[str, Any]) -> PDMPath:
        transform = self._build_ego_transform(info_data)
        return self._build_centerline(info_data, transform)

    def _lane_drivable_map(
        self,
        lane_tokens: list[str],
        lane_polygons: list[Polygon],
    ) -> DrivableMap:
        return DrivableMap(
            tokens=lane_tokens,
            types=[SemanticMapLayer.LANE] * len(lane_tokens),
            polygons=np.asarray(lane_polygons, dtype=object),
        )

    def _build_ego_transform(self, info_data: dict[str, Any]) -> _EgoTransform:
        ego_attr = info_data.get("ego_car_attribute") or {}
        position = ego_attr.get("position") or []
        if not (isinstance(position, list) and len(position) >= 2):
            raise ValueError("ego_car_attribute.position is required")
        yaw = ego_attr.get("ego_heading")
        if not isinstance(yaw, (int, float)):
            raise ValueError("ego_car_attribute.ego_heading is required")

        hx = float(position[0])
        hy = float(position[1])
        hz = float(position[2]) if len(position) > 2 else 0.0
        theta = float(yaw)
        c = math.cos(theta)
        s = math.sin(theta)
        rotation = np.array(
            [
                [s, -c, 0.0],
                [c, s, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return _EgoTransform(position=(hx, hy, hz), rotation_world_to_nous=rotation)

    def _build_ego_state(self, info_data: dict[str, Any]) -> np.ndarray:
        ego_attr = info_data.get("ego_car_attribute") or {}
        speed = float(ego_attr.get("speed") or 0.0)
        acceleration = float(ego_attr.get("acceleration") or 0.0)
        state = np.zeros(StateIndex.size(), dtype=np.float64)
        state[StateIndex.VELOCITY_X] = speed
        state[StateIndex.ACCELERATION_X] = acceleration
        return state

    def _build_ego_past_states(
        self,
        info_data: dict[str, Any],
        transform: _EgoTransform,
    ) -> np.ndarray:
        his_track = (info_data.get("ego_car_attribute") or {}).get("his_track") or []
        poses = [self._pose_to_nuplan_xyh(item, transform) for item in his_track]
        poses = [pose for pose in poses if pose is not None]
        if not poses:
            return np.zeros((0, StateIndex.size()), dtype=np.float64)

        pose_array = np.asarray(poses[-20:], dtype=np.float64)
        states = np.zeros((len(pose_array), StateIndex.size()), dtype=np.float64)
        states[:, StateIndex.X] = pose_array[:, 0]
        states[:, StateIndex.Y] = pose_array[:, 1]
        states[:, StateIndex.HEADING] = pose_array[:, 2]
        if len(pose_array) > 1:
            times = np.arange(len(pose_array), dtype=np.float64) * self.interval_time
            states[:, StateIndex.VELOCITY_X] = np.gradient(pose_array[:, 0], times, edge_order=1)
            states[:, StateIndex.VELOCITY_Y] = np.gradient(pose_array[:, 1], times, edge_order=1)
        return states

    def _build_lane_polygons(
        self,
        info_data: dict[str, Any],
        transform: _EgoTransform,
    ) -> tuple[list[str], list[Polygon]]:
        tokens: list[str] = []
        polygons: list[Polygon] = []
        for index, lane in enumerate(info_data.get("lanes_info") or []):
            if not isinstance(lane, dict):
                continue
            left = self._boundary_points_to_nuplan(lane.get("leftBoundary"), transform)
            right = self._boundary_points_to_nuplan(lane.get("rightBoundary"), transform)
            if len(left) < 2 or len(right) < 2:
                continue
            polygon = Polygon(left + list(reversed(right)))
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 1e-3:
                continue
            token = self._extract_id(lane.get("laneId")) or self._extract_id(lane.get("id")) or f"lane_{index}"
            tokens.append(str(token))
            polygons.append(polygon)
        return tokens, polygons

    def _build_centerline(self, info_data: dict[str, Any], transform: _EgoTransform) -> PDMPath:
        ego_attr = info_data.get("ego_car_attribute") or {}
        pnc_local_routing = ego_attr.get("pnc_local_routing") or {}
        routing_points = pnc_local_routing.get("routing_points") if isinstance(pnc_local_routing, dict) else None
        points = self._flatten_points(routing_points)
        centerline_points = self._points_to_nuplan_xy(points, transform)

        if len(centerline_points) < 2:
            candidate = ego_attr.get("candidate_trajectory_point") or []
            centerline_points = [
                xyh[:2]
                for xyh in (self._pose_to_nuplan_xyh(item, transform) for item in candidate)
                if xyh is not None
            ]

        deduped = self._dedupe_points(centerline_points)
        if len(deduped) < 2:
            raise ValueError("Unable to build centerline from routing or candidate trajectory")
        return PDMPath(np.asarray(deduped, dtype=np.float64))

    def _match_route_lanes(
        self,
        lane_tokens: list[str],
        lane_polygons: list[Polygon],
        centerline: PDMPath,
    ) -> set[str]:
        route_buffer = centerline.linestring.buffer(2.0)
        matched = {
            token
            for token, polygon in zip(lane_tokens, lane_polygons)
            if polygon.intersects(route_buffer)
        }
        return matched or set(lane_tokens)

    def _build_observation(
        self,
        info_data: dict[str, Any],
        transform: _EgoTransform,
    ) -> tuple[PDMObservation, dict[str, str], dict[str, float]]:
        dynamic_tokens: list[str] = []
        dynamic_coords: list[np.ndarray] = []
        dynamic_velocities: list[tuple[float, float]] = []
        track_object_types: dict[str, str] = {}
        track_speeds: dict[str, float] = {}

        for index, obstacle in enumerate(info_data.get("obs_items") or []):
            if not isinstance(obstacle, dict):
                continue
            center = self._point_to_nuplan_xy(obstacle.get("position"), transform)
            if center is None:
                continue
            length = self._float_or_none(obstacle.get("length"))
            width = self._float_or_none(obstacle.get("width"))
            if length is None or width is None or length <= 0.0 or width <= 0.0:
                continue
            heading = self._heading_to_nuplan(obstacle.get("heading"), transform)
            corners = self._box_corners(center, length, width, heading)
            token = str(obstacle.get("id") if obstacle.get("id") is not None else f"obs_{index}")
            speed = self._float_or_none(obstacle.get("speed"))
            velocity = self._velocity_to_nuplan(obstacle.get("velocity"), transform)
            if velocity is None and speed is not None:
                velocity = (float(speed) * math.cos(heading), float(speed) * math.sin(heading))
            if velocity is None:
                velocity = (0.0, 0.0)

            dynamic_tokens.append(token)
            dynamic_coords.append(corners)
            dynamic_velocities.append(velocity)
            track_object_types[token] = self._track_object_type(obstacle)
            track_speeds[token] = float(speed) if speed is not None else float(math.hypot(*velocity))

        observation = PDMObservation(num_steps=self.num_observation_steps, interval_time=self.interval_time)
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

    def _build_reference_trajectory(
        self,
        info_data: dict[str, Any],
        transform: _EgoTransform,
    ) -> np.ndarray | None:
        ego_attr = info_data.get("ego_car_attribute") or {}
        candidate = self._build_sparse_trajectory(
            ego_attr.get("candidate_trajectory_point"),
            transform,
            prefer_time=True,
        )
        if candidate is not None:
            return candidate
        return self._build_sparse_trajectory(ego_attr.get("future_track"), transform, prefer_time=False)

    def _build_sparse_trajectory(
        self,
        raw_points: Any,
        transform: _EgoTransform,
        *,
        prefer_time: bool,
    ) -> np.ndarray | None:
        if not isinstance(raw_points, list) or not raw_points:
            return None
        poses = [self._pose_to_nuplan_xyh(item, transform) for item in raw_points]
        valid = [(item, pose) for item, pose in zip(raw_points, poses) if pose is not None]
        if not valid:
            return None

        selected: list[np.ndarray] = []
        targets = [0.5 * i for i in range(1, 9)]
        if prefer_time and isinstance(valid[0][0], dict) and "relativeTime" in valid[0][0]:
            times = np.asarray([float(item.get("relativeTime", 0.0)) for item, _ in valid], dtype=np.float64)
            for target in targets:
                selected.append(valid[int(np.argmin(np.abs(times - target)))][1])
        elif len(valid) >= 40:
            for target in targets:
                index = min(max(int(round(target / self.interval_time)) - 1, 0), len(valid) - 1)
                selected.append(valid[index][1])
        else:
            indices = np.linspace(0, len(valid) - 1, min(8, len(valid))).round().astype(int)
            selected.extend(valid[int(index)][1] for index in indices)

        if len(selected) < 2:
            return None
        while len(selected) < 8:
            selected.append(selected[-1].copy())
        return np.asarray(selected[:8], dtype=np.float64)

    def _pose_to_nuplan_xyh(self, item: Any, transform: _EgoTransform) -> np.ndarray | None:
        if isinstance(item, dict):
            point = [item.get("x"), item.get("y"), item.get("z", 0.0)]
            heading_raw = item.get("theta", item.get("heading"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            point = item
            heading_raw = item[2] if len(item) >= 3 else None
        else:
            return None

        xy = self._point_to_nuplan_xy(point, transform)
        if xy is None:
            return None
        heading = self._heading_to_nuplan(heading_raw, transform) if heading_raw is not None else 0.0
        return np.asarray([xy[0], xy[1], heading], dtype=np.float64)

    def _boundary_points_to_nuplan(
        self,
        boundary: Any,
        transform: _EgoTransform,
    ) -> list[tuple[float, float]]:
        if not isinstance(boundary, dict):
            return []
        points: list[Any] = []
        for segment in ((boundary.get("curve") or {}).get("segment") or []):
            line_segment = (segment.get("lineSegment") or {}) if isinstance(segment, dict) else {}
            points.extend(line_segment.get("point") or [])
        return self._points_to_nuplan_xy(points, transform)

    def _points_to_nuplan_xy(
        self,
        points: Iterable[Any] | None,
        transform: _EgoTransform,
    ) -> list[tuple[float, float]]:
        if points is None:
            return []
        output = []
        for point in points:
            xy = self._point_to_nuplan_xy(point, transform)
            if xy is not None:
                output.append(xy)
        return output

    def _point_to_nuplan_xy(self, point: Any, transform: _EgoTransform) -> tuple[float, float] | None:
        xyz = self._coerce_xyz(point)
        if xyz is None:
            return None
        relative_world = np.asarray(xyz, dtype=np.float64) - np.asarray(transform.position, dtype=np.float64)
        nous_xyz = transform.rotation_world_to_nous @ relative_world
        x_nuplan, y_nuplan = CoordinateConverter.nous_to_nuplan(float(nous_xyz[0]), float(nous_xyz[1]))
        return x_nuplan, y_nuplan

    def _velocity_to_nuplan(
        self,
        velocity: Any,
        transform: _EgoTransform,
    ) -> tuple[float, float] | None:
        xyz = self._coerce_xyz(velocity)
        if xyz is None:
            return None
        nous_xyz = transform.rotation_world_to_nous @ np.asarray(xyz, dtype=np.float64)
        return CoordinateConverter.nous_to_nuplan(float(nous_xyz[0]), float(nous_xyz[1]))

    def _heading_to_nuplan(self, heading: Any, transform: _EgoTransform) -> float:
        if not isinstance(heading, (int, float)):
            return 0.0
        world_dir = np.asarray([math.cos(float(heading)), math.sin(float(heading)), 0.0], dtype=np.float64)
        nous_dir = transform.rotation_world_to_nous @ world_dir
        nx, ny = CoordinateConverter.nous_to_nuplan(float(nous_dir[0]), float(nous_dir[1]))
        return float(normalize_angle(math.atan2(ny, nx)))

    @staticmethod
    def _box_corners(
        center: tuple[float, float],
        length: float,
        width: float,
        heading: float,
    ) -> np.ndarray:
        cx, cy = center
        forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
        left = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float64)
        half_l = float(length) / 2.0
        half_w = float(width) / 2.0
        center_vec = np.asarray([cx, cy], dtype=np.float64)
        return np.asarray(
            [
                center_vec + forward * half_l + left * half_w,
                center_vec - forward * half_l + left * half_w,
                center_vec - forward * half_l - left * half_w,
                center_vec + forward * half_l - left * half_w,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _coerce_xyz(value: Any) -> tuple[float, float, float] | None:
        if isinstance(value, dict):
            if "x" not in value or "y" not in value:
                return None
            return (
                float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)),
            )
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return (
                float(value[0]),
                float(value[1]),
                float(value[2]) if len(value) > 2 else 0.0,
            )
        return None

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _extract_id(value: Any) -> str | None:
        if isinstance(value, dict) and value.get("id") is not None:
            return str(value["id"])
        if value is not None and not isinstance(value, dict):
            return str(value)
        return None

    @staticmethod
    def _flatten_points(value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        if value and all(isinstance(item, (list, tuple, dict)) for item in value):
            if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], (list, tuple, dict)):
                return [point for group in value for point in group]
            return value
        return []

    @staticmethod
    def _dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        output: list[tuple[float, float]] = []
        for point in points:
            if output and math.hypot(point[0] - output[-1][0], point[1] - output[-1][1]) < 1e-3:
                continue
            output.append(point)
        if len(output) >= 2:
            return output
        if len(points) >= 2:
            line = LineString(points)
            if line.length > 1e-3:
                return list(line.coords)
        return output

    @staticmethod
    def _track_object_type(obstacle: dict[str, Any]) -> str:
        type_name = str(obstacle.get("type") or obstacle.get("subtype") or "").upper()
        if type_name in {"VEHICLE", "PEDESTRIAN", "BICYCLE", "CYCLIST", "TRICYCLE"}:
            return "agent"
        return "static"
