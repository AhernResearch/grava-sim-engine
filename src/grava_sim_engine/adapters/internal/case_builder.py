"""Case-level SceneContext builder for internal shard frame sequences."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from grava_common.coordinates import CoordinateConverter
from grava_sim_engine.core.types import SceneContext

from .frame_builder import (
    FUTURE_OBSTACLE_TRACKS_COORDINATE_FRAME,
    FUTURE_OBSTACLE_TRACKS_KEY,
    InternalShardFrameSceneContextBuilder,
    _box_corners,
    _float,
    _is_sequence,
)


@dataclass(frozen=True)
class _EgoWorldTransform:
    position_xy: np.ndarray
    rotation_world_to_nous: np.ndarray

    @classmethod
    def from_frame(cls, frame_data: dict[str, Any]) -> "_EgoWorldTransform":
        ego_car = frame_data.get("ego_car") or {}
        position = ego_car.get("position_world") or {}
        if not isinstance(position, dict):
            raise ValueError("ego_car.position_world is required for case-level observation")
        yaw = ego_car.get("yaw")
        if not isinstance(yaw, (int, float)):
            raise ValueError("ego_car.yaw is required for case-level observation")

        theta = float(yaw)
        rotation = np.asarray(
            [
                [math.sin(theta), -math.cos(theta)],
                [math.cos(theta), math.sin(theta)],
            ],
            dtype=np.float64,
        )
        return cls(
            position_xy=np.asarray(
                [_float(position.get("x"), 0.0), _float(position.get("y"), 0.0)],
                dtype=np.float64,
            ),
            rotation_world_to_nous=rotation,
        )


@dataclass(frozen=True)
class _ObstacleRecord:
    timestamp_s: float
    rel_time_s: float
    polygon_coords: np.ndarray
    velocity_xy: np.ndarray
    object_type: str
    speed_mps: float


class InternalCaseRecordSceneContextBuilder:
    """Enrich internal frame.json records with case-level future obstacle tracks.

    Each internal frame stores obstacle boxes in that frame's ego-local NOUS
    coordinates. This builder projects future frame records into each target
    frame and writes them into a single-frame ``future_obstacle_tracks`` field.
    """

    def __init__(
        self,
        *,
        horizon_seconds: float = 4.0,
        interval_time: float = 0.1,
        frame_builder: InternalShardFrameSceneContextBuilder | None = None,
    ) -> None:
        self.horizon_seconds = float(horizon_seconds)
        self.interval_time = float(interval_time)
        self.num_observation_steps = int(round(self.horizon_seconds / self.interval_time)) + 1
        self._frame_builder = frame_builder or InternalShardFrameSceneContextBuilder(
            num_observation_steps=self.num_observation_steps,
            interval_time=self.interval_time,
        )

    def build_case(
        self,
        case_frames: list[dict[str, Any]],
        *,
        log_name: str | None = None,
        limit: int = 0,
    ) -> list[SceneContext]:
        sorted_frames = self._sort_frames(case_frames)
        contexts: list[SceneContext] = []
        for target_index in range(len(sorted_frames)):
            if limit > 0 and len(contexts) >= limit:
                break
            contexts.append(
                self.build_target(
                    sorted_frames,
                    target_index=target_index,
                    log_name=log_name,
                )
            )
        return contexts

    def build_target(
        self,
        case_frames: list[dict[str, Any]],
        *,
        target_index: int,
        log_name: str | None = None,
        target_info_data: dict[str, Any] | None = None,
    ) -> SceneContext:
        sorted_frames = self._sort_frames(case_frames)
        if not 0 <= target_index < len(sorted_frames):
            raise IndexError(f"target_index out of range: {target_index}")

        target_frame = dict(sorted_frames[target_index])
        target_frame[FUTURE_OBSTACLE_TRACKS_KEY] = self.build_future_obstacle_tracks(
            sorted_frames,
            target_index=target_index,
        )
        ctx = self._frame_builder.build(
            target_frame,
            log_name=log_name or str(target_frame.get("case_id") or "internal"),
            scene_token=str(target_frame.get("timestamp") or target_frame.get("image_id")),
            info_data=target_info_data,
        )
        return ctx

    def enrich_case(
        self,
        case_frames: list[dict[str, Any]],
        *,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        sorted_frames = self._sort_frames(case_frames)
        enriched_frames: list[dict[str, Any]] = []
        for target_index in range(len(sorted_frames)):
            if limit > 0 and len(enriched_frames) >= limit:
                break
            enriched_frames.append(self.enrich_frame(sorted_frames, target_index=target_index))
        return enriched_frames

    def enrich_frame(
        self,
        case_frames: list[dict[str, Any]],
        *,
        target_index: int,
    ) -> dict[str, Any]:
        sorted_frames = self._sort_frames(case_frames)
        if not 0 <= target_index < len(sorted_frames):
            raise IndexError(f"target_index out of range: {target_index}")

        enriched_frame = copy.deepcopy(sorted_frames[target_index])
        enriched_frame[FUTURE_OBSTACLE_TRACKS_KEY] = self.build_future_obstacle_tracks(
            sorted_frames,
            target_index=target_index,
        )
        return enriched_frame

    def build_future_obstacle_tracks(
        self,
        case_frames: list[dict[str, Any]],
        *,
        target_index: int,
    ) -> dict[str, Any]:
        sorted_frames = self._sort_frames(case_frames)
        if not 0 <= target_index < len(sorted_frames):
            raise IndexError(f"target_index out of range: {target_index}")

        target_frame = sorted_frames[target_index]
        target_timestamp = _frame_timestamp(target_frame)
        target_transform = _EgoWorldTransform.from_frame(target_frame)
        records_by_token: dict[str, list[_ObstacleRecord]] = {}

        for frame_data in sorted_frames[target_index:]:
            timestamp_s = _frame_timestamp(frame_data)
            rel_time_s = timestamp_s - target_timestamp
            if rel_time_s < -1e-6:
                continue
            if rel_time_s > self.horizon_seconds + 1e-6:
                break

            source_transform = _EgoWorldTransform.from_frame(frame_data)
            for obstacle_index, obstacle in enumerate(frame_data.get("obstacles") or []):
                token = str(
                    obstacle.get("id")
                    or obstacle.get("track_id")
                    or f"obstacle_{obstacle_index}"
                )
                record = self._obstacle_record(
                    obstacle,
                    timestamp_s=timestamp_s,
                    rel_time_s=rel_time_s,
                    source_transform=source_transform,
                    target_transform=target_transform,
                )
                if record is None:
                    continue
                records_by_token.setdefault(token, []).append(record)

        tracks: list[dict[str, Any]] = []
        for token, records in sorted(records_by_token.items()):
            sorted_records = sorted(records, key=lambda item: item.rel_time_s)
            tracks.append(
                {
                    "id": token,
                    "object_type": sorted_records[0].object_type,
                    "speed_mps": sorted_records[0].speed_mps,
                    "states": [_obstacle_record_to_state(record) for record in sorted_records],
                }
            )

        return {
            "schema_version": 1,
            "source": "internal_case_record",
            "horizon_seconds": self.horizon_seconds,
            "coordinate_frame": FUTURE_OBSTACLE_TRACKS_COORDINATE_FRAME,
            "tracks": tracks,
        }

    def _obstacle_record(
        self,
        obstacle: dict[str, Any],
        *,
        timestamp_s: float,
        rel_time_s: float,
        source_transform: _EgoWorldTransform,
        target_transform: _EgoWorldTransform,
    ) -> _ObstacleRecord | None:
        bbox = obstacle.get("bbox_3d")
        if not _is_sequence(bbox) or len(bbox) < 7:
            return None

        center_x, center_y = CoordinateConverter.nous_to_nuplan(
            _float(bbox[0], 0.0),
            _float(bbox[1], 0.0),
        )
        length = max(_float(bbox[3], 0.0), 0.1)
        width = max(_float(bbox[4], 0.0), 0.1)
        source_velocity = np.asarray(
            self._frame_builder._obstacle_velocity(obstacle),
            dtype=np.float64,
        )
        heading = self._frame_builder._obstacle_heading(
            bbox,
            source_velocity[0],
            source_velocity[1],
        )
        source_coords = np.asarray(
            _box_corners(center_x, center_y, length, width, heading),
            dtype=np.float64,
        )
        target_coords = _transform_nuplan_points_between_ego_frames(
            source_coords,
            source_transform=source_transform,
            target_transform=target_transform,
        )
        target_velocity = _transform_nuplan_vector_between_ego_frames(
            source_velocity,
            source_transform=source_transform,
            target_transform=target_transform,
        )
        return _ObstacleRecord(
            timestamp_s=float(timestamp_s),
            rel_time_s=float(rel_time_s),
            polygon_coords=target_coords,
            velocity_xy=target_velocity,
            object_type=self._frame_builder._track_object_type(obstacle),
            speed_mps=_float(obstacle.get("speed_mps"), float(np.linalg.norm(target_velocity))),
        )

    @staticmethod
    def _sort_frames(case_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not case_frames:
            raise ValueError("case_frames must not be empty")
        return sorted(case_frames, key=_frame_timestamp)


def build_scene_contexts_from_case_frames(
    case_frames: list[dict[str, Any]],
    *,
    log_name: str | None = None,
    limit: int = 0,
) -> list[SceneContext]:
    return InternalCaseRecordSceneContextBuilder().build_case(
        case_frames,
        log_name=log_name,
        limit=limit,
    )


def _frame_timestamp(frame_data: dict[str, Any]) -> float:
    raw_timestamp = frame_data.get("timestamp") or frame_data.get("image_id")
    if raw_timestamp is None:
        raise ValueError("frame_data must contain timestamp or image_id")
    return _float(raw_timestamp, math.nan)


def _obstacle_record_to_state(record: _ObstacleRecord) -> dict[str, Any]:
    polygon_nous = CoordinateConverter.nuplan_to_nous_batch(record.polygon_coords)
    center_nous = polygon_nous.mean(axis=0)
    velocity_nous = CoordinateConverter.nuplan_to_nous(
        float(record.velocity_xy[0]),
        float(record.velocity_xy[1]),
    )
    return {
        "timestamp": record.timestamp_s,
        "rel_time_s": record.rel_time_s,
        "center": {
            "x": float(center_nous[0]),
            "y": float(center_nous[1]),
        },
        "polygon": [
            {
                "x": float(point[0]),
                "y": float(point[1]),
            }
            for point in polygon_nous
        ],
        "velocity": {
            "x": float(velocity_nous[0]),
            "y": float(velocity_nous[1]),
            "z": 0.0,
        },
        "speed_mps": record.speed_mps,
    }


def _transform_nuplan_points_between_ego_frames(
    points_nuplan: np.ndarray,
    *,
    source_transform: _EgoWorldTransform,
    target_transform: _EgoWorldTransform,
) -> np.ndarray:
    source_nous = CoordinateConverter.nuplan_to_nous_batch(points_nuplan)
    world_xy = source_nous @ source_transform.rotation_world_to_nous + source_transform.position_xy
    target_nous = (
        world_xy - target_transform.position_xy
    ) @ target_transform.rotation_world_to_nous.T
    return CoordinateConverter.nous_to_nuplan_batch(target_nous)


def _transform_nuplan_vector_between_ego_frames(
    vector_nuplan: np.ndarray,
    *,
    source_transform: _EgoWorldTransform,
    target_transform: _EgoWorldTransform,
) -> np.ndarray:
    vector_nous = np.asarray(CoordinateConverter.nuplan_to_nous(*vector_nuplan), dtype=np.float64)
    world_vector = vector_nous @ source_transform.rotation_world_to_nous
    target_nous = world_vector @ target_transform.rotation_world_to_nous.T
    return np.asarray(CoordinateConverter.nous_to_nuplan(*target_nous), dtype=np.float64)
