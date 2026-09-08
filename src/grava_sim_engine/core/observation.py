from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .geometry import coords_to_polygons
from .occupancy import OccupancyMap


def _coerce_coords_array(coords: Sequence[np.ndarray] | np.ndarray | None) -> np.ndarray:
    if coords is None:
        return np.empty((0, 4, 2), dtype=np.float64)

    if isinstance(coords, np.ndarray):
        coord_array = np.asarray(coords, dtype=np.float64)
        if coord_array.size == 0:
            return np.empty((0, 4, 2), dtype=np.float64)
        if coord_array.ndim == 2:
            if coord_array.shape[-2:] not in ((4, 2), (5, 2)):
                raise ValueError(f"coords must have shape [N,4|5,2], got {coord_array.shape}")
            return coord_array[None, ...]
        if coord_array.ndim != 3 or coord_array.shape[-2:] not in ((4, 2), (5, 2)):
            raise ValueError(f"coords must have shape [N,4|5,2], got {coord_array.shape}")
        return coord_array

    coord_chunks = [np.asarray(chunk, dtype=np.float64) for chunk in coords if np.asarray(chunk).size > 0]
    if not coord_chunks:
        return np.empty((0, 4, 2), dtype=np.float64)

    if all(chunk.ndim == 2 and chunk.shape[-2:] in ((4, 2), (5, 2)) for chunk in coord_chunks):
        return np.stack(coord_chunks, axis=0)

    if all(chunk.ndim == 3 and chunk.shape[-2:] in ((4, 2), (5, 2)) for chunk in coord_chunks):
        return np.concatenate(coord_chunks, axis=0)

    raise ValueError("coords must contain arrays shaped [4|5,2] or [N,4|5,2]")


def _coerce_velocity_array(velocities: np.ndarray | None) -> np.ndarray:
    if velocities is None:
        return np.empty((0, 2), dtype=np.float64)

    velocity_array = np.asarray(velocities, dtype=np.float64)
    if velocity_array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if velocity_array.ndim == 1:
        if velocity_array.shape[0] != 2:
            raise ValueError(f"dynamic_velocities must have shape [N,2], got {velocity_array.shape}")
        return velocity_array[None, :]
    if velocity_array.ndim != 2 or velocity_array.shape[1] != 2:
        raise ValueError(f"dynamic_velocities must have shape [N,2], got {velocity_array.shape}")
    return velocity_array


def _build_occupancy_map(tokens: List[str], coords: np.ndarray) -> Optional[OccupancyMap]:
    if len(tokens) == 0:
        return None
    polygons = coords_to_polygons(coords)
    return OccupancyMap(tokens=tokens, polygons=polygons)


class PDMObservation:
    """Time-indexed occupancy maps for static, dynamic, and red-light obstacles."""

    def __init__(self, num_steps: int, interval_time: float = 0.1):
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if interval_time <= 0.0:
            raise ValueError(f"interval_time must be positive, got {interval_time}")

        self._num_steps = num_steps
        self._interval_time = interval_time
        self._observation_sample_res = 1
        self._global_to_local_idcs = list(range(num_steps))
        self._occupancy_maps: List[Optional[OccupancyMap]] = [None] * num_steps
        self._red_light_maps: List[Optional[OccupancyMap]] = [None] * num_steps

    @staticmethod
    def _infer_observation_sample_res(interval_time: float) -> int:
        target_observation_dt = 0.2
        sample_res = max(int(round(target_observation_dt / interval_time)), 1)
        return sample_res if np.isclose(sample_res * interval_time, target_observation_dt) else 1

    @property
    def global_to_local_idcs(self) -> List[int]:
        return self._global_to_local_idcs

    @property
    def interval_time(self) -> float:
        return self._interval_time

    def update(
        self,
        static_coords: Sequence[np.ndarray] | np.ndarray,
        static_tokens: List[str],
        dynamic_coords: np.ndarray,
        dynamic_tokens: List[str],
        dynamic_velocities: np.ndarray,
        red_light_coords: Sequence[np.ndarray] | np.ndarray,
        red_light_tokens: List[str],
    ) -> None:
        """Build occupancy maps for all simulation steps."""
        static_coords_array = _coerce_coords_array(static_coords)
        dynamic_coords_array = _coerce_coords_array(dynamic_coords)
        red_light_coords_array = _coerce_coords_array(red_light_coords)
        dynamic_velocities_array = _coerce_velocity_array(dynamic_velocities)

        if len(static_tokens) != len(static_coords_array):
            raise ValueError(
                f"static_tokens/static_coords length mismatch: {len(static_tokens)} != {len(static_coords_array)}"
            )
        if len(dynamic_tokens) != len(dynamic_coords_array):
            raise ValueError(
                f"dynamic_tokens/dynamic_coords length mismatch: {len(dynamic_tokens)} != {len(dynamic_coords_array)}"
            )
        if len(dynamic_tokens) != len(dynamic_velocities_array):
            raise ValueError(
                "dynamic_tokens/dynamic_velocities length mismatch: "
                f"{len(dynamic_tokens)} != {len(dynamic_velocities_array)}"
            )
        if len(red_light_tokens) != len(red_light_coords_array):
            raise ValueError(
                "red_light_tokens/red_light_coords length mismatch: "
                f"{len(red_light_tokens)} != {len(red_light_coords_array)}"
            )

        static_polygons = coords_to_polygons(static_coords_array) if len(static_coords_array) > 0 else None
        red_light_map = _build_occupancy_map(red_light_tokens, red_light_coords_array)

        for time_idx in range(self._num_steps):
            current_tokens: List[str] = []
            polygon_groups: List[np.ndarray] = []

            if static_polygons is not None:
                current_tokens.extend(static_tokens)
                polygon_groups.append(static_polygons)

            if len(dynamic_coords_array) > 0:
                delta_t = time_idx * self._interval_time
                displacement = delta_t * dynamic_velocities_array
                dynamic_coords_t = dynamic_coords_array + displacement[:, None, :]
                polygon_groups.append(coords_to_polygons(dynamic_coords_t))
                current_tokens.extend(dynamic_tokens)

            if len(red_light_tokens) > 0:
                polygon_groups.append(coords_to_polygons(red_light_coords_array))
                current_tokens.extend(red_light_tokens)

            if polygon_groups:
                polygons = np.concatenate(polygon_groups, axis=0)
                self._occupancy_maps[time_idx] = OccupancyMap(current_tokens, polygons)
            else:
                self._occupancy_maps[time_idx] = None

            self._red_light_maps[time_idx] = red_light_map

    def get_occupancy_map(self, time_idx: int) -> Optional[OccupancyMap]:
        if not 0 <= time_idx < self._num_steps:
            raise IndexError(f"time_idx out of range: {time_idx}")
        return self._occupancy_maps[time_idx]

    def get_red_light_map(self, time_idx: int) -> Optional[OccupancyMap]:
        if not 0 <= time_idx < self._num_steps:
            raise IndexError(f"time_idx out of range: {time_idx}")
        return self._red_light_maps[time_idx]
