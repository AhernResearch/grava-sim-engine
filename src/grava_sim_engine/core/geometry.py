from __future__ import annotations

"""Geometry helpers under NUPLAN ego-frame convention.

The engine uses x-forward, y-left coordinates for ego-relative poses.
"""

from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from shapely.creation import polygons
from shapely.geometry import LineString, Point

from .enums import BBCoordsIndex, StateIndex
from .types import VehicleParams


def normalize_angle(angle: Any, min_: float = -np.pi) -> Any:
    angle_array = np.asarray(angle)
    if not np.all(np.isfinite(angle_array)):
        raise AssertionError("angle is not finite")
    return (angle_array - min_) % (2 * np.pi) + min_


def convert_absolute_to_relative_se2(
    origin_pose: np.ndarray,
    absolute_poses: np.ndarray,
) -> np.ndarray:
    origin_pose = np.asarray(origin_pose, dtype=np.float64)
    absolute_poses = np.asarray(absolute_poses, dtype=np.float64)

    if origin_pose.shape != (3,):
        raise ValueError(f"origin_pose must have shape (3,), got {origin_pose.shape}")
    if absolute_poses.shape[-1] != 3:
        raise ValueError(f"absolute_poses last dim must be 3, got {absolute_poses.shape}")

    theta = -origin_pose[StateIndex.HEADING]
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )

    relative_poses = absolute_poses - origin_pose
    relative_poses[..., :2] = relative_poses[..., :2] @ rotation.T
    relative_poses[..., 2] = normalize_angle(relative_poses[..., 2])
    return relative_poses


def translate_lon_lat(
    centers: np.ndarray,
    headings: np.ndarray,
    lon: float,
    lat: float,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    translated = np.empty_like(centers, dtype=np.float64)
    translated[..., 0] = (
        centers[..., 0] + lon * np.cos(headings) - lat * np.sin(headings)
    )
    translated[..., 1] = (
        centers[..., 1] + lon * np.sin(headings) + lat * np.cos(headings)
    )
    return translated


def state_to_coords(states: np.ndarray, vehicle: VehicleParams) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.shape[-1] != StateIndex.size():
        raise ValueError(f"states last dim must be {StateIndex.size()}, got {states.shape}")

    rear_axles = states[..., [StateIndex.X, StateIndex.Y]]
    headings = states[..., StateIndex.HEADING]
    centers = translate_lon_lat(rear_axles, headings, vehicle.rear_axle_to_center, 0.0)

    coords = np.zeros((*states.shape[:-1], len(BBCoordsIndex), 2), dtype=np.float64)
    coords[..., BBCoordsIndex.CENTER, :] = centers

    front_lon = vehicle.half_length
    rear_lon = -vehicle.half_length

    coords[..., BBCoordsIndex.FRONT_LEFT, :] = translate_lon_lat(
        centers, headings, front_lon, vehicle.half_width
    )
    coords[..., BBCoordsIndex.REAR_LEFT, :] = translate_lon_lat(
        centers, headings, rear_lon, vehicle.half_width
    )
    coords[..., BBCoordsIndex.REAR_RIGHT, :] = translate_lon_lat(
        centers, headings, rear_lon, -vehicle.half_width
    )
    coords[..., BBCoordsIndex.FRONT_RIGHT, :] = translate_lon_lat(
        centers, headings, front_lon, -vehicle.half_width
    )
    return coords


def coords_to_polygons(corners: np.ndarray):
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape[-2] not in (4, 5) or corners.shape[-1] != 2:
        raise ValueError(f"corners must have shape [..., 4|5, 2], got {corners.shape}")
    return polygons(corners[..., :4, :])


def calculate_progress(corners: np.ndarray, centerline: "PDMPath") -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape[-2:] != (len(BBCoordsIndex), 2):
        raise ValueError(
            f"corners must have shape [..., {len(BBCoordsIndex)}, 2], got {corners.shape}"
        )

    centers = corners[..., BBCoordsIndex.CENTER, :]
    flat_centers = centers.reshape(-1, 2)
    projected = np.array(
        [centerline.project(Point(center)) for center in flat_centers],
        dtype=np.float64,
    )
    return projected.reshape(centers.shape[:-1])


class PDMPath:
    def __init__(self, discrete_path: np.ndarray):
        discrete_path = np.asarray(discrete_path, dtype=np.float64)
        if discrete_path.ndim != 2 or discrete_path.shape[1] != 2:
            raise ValueError(
                f"discrete_path must have shape [N, 2], got {discrete_path.shape}"
            )
        if len(discrete_path) < 2:
            raise ValueError("discrete_path must contain at least two points")

        self._discrete_path = discrete_path
        deltas = np.diff(discrete_path, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        self._progress = np.concatenate(([0.0], np.cumsum(segment_lengths, dtype=np.float64)))
        self._interpolator = interp1d(
            self._progress,
            discrete_path,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        self._linestring = LineString(discrete_path)

    @property
    def length(self) -> float:
        return float(self._linestring.length)

    @property
    def linestring(self) -> LineString:
        return self._linestring

    @property
    def discrete_path(self) -> np.ndarray:
        return self._discrete_path

    def project(self, point: Any) -> float:
        geometry = point if hasattr(point, "geom_type") else Point(point)
        return float(self._linestring.project(geometry))

    def interpolate(self, distance: float):
        clipped_distance = float(np.clip(distance, 0.0, self.length))
        return self._linestring.interpolate(clipped_distance)
