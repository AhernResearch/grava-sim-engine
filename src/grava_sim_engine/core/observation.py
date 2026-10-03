from __future__ import annotations

from typing import List, Optional

from .occupancy import OccupancyMap


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

    @property
    def global_to_local_idcs(self) -> List[int]:
        return self._global_to_local_idcs

    @property
    def interval_time(self) -> float:
        return self._interval_time

    def get_occupancy_map(self, time_idx: int) -> Optional[OccupancyMap]:
        if not 0 <= time_idx < self._num_steps:
            raise IndexError(f"time_idx out of range: {time_idx}")
        return self._occupancy_maps[time_idx]

    def get_red_light_map(self, time_idx: int) -> Optional[OccupancyMap]:
        if not 0 <= time_idx < self._num_steps:
            raise IndexError(f"time_idx out of range: {time_idx}")
        return self._red_light_maps[time_idx]
