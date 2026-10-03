from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RLScorerConfig:
    """Safety gates and weighted EP/TTC/LK/HC reward configuration."""

    ep_weight: float = 5.0
    ttc_weight: float = 5.0
    hc_weight: float = 2.0
    lk_weight: float = 0.0
    safety_gate_alpha: float = 0.5
    safety_mode: str = "continuous"
    ddc_weight: float = 0.0
    tlc_weight: float = 0.0
    tlc_margin: float = 1.0
    obstacle_clearance_margin: float = 5.0
    progress_distance_threshold: float = 5.0
    ttc_horizon: float = 4.0
    lane_keeping_deviation: float = 0.5
    lane_keeping_max_deviation: float = 2.0
    driving_direction_horizon: float = 1.0
    driving_direction_compliance_threshold: float = 2.0
    driving_direction_violation_threshold: float = 6.0
    stopped_speed_threshold: float = 0.005
    future_collision_horizon: float = 1.0

    @property
    def performance_weights(self) -> npt.NDArray[np.float64]:
        return np.array(
            [self.ep_weight, self.ttc_weight, self.lk_weight, self.hc_weight], dtype=np.float64
        )
