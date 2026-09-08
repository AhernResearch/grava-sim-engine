from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import numpy as np

if TYPE_CHECKING:
    from .geometry import PDMPath
    from .observation import PDMObservation
    from .occupancy import DrivableMap


@dataclass
class VehicleParams:
    """Default: Chrysler Pacifica (nuPlan ego vehicle), matching recogdrive/NavSim."""

    half_length: float = 2.588  # (front_length + rear_length) / 2 = (4.049 + 1.127) / 2
    half_width: float = 1.1485  # width / 2 = 2.297 / 2
    rear_axle_to_center: float = 1.461
    wheel_base: float = 3.089


@dataclass
class KeyActionObstacle:
    """Internal key-action obstacle annotation in sim-engine coordinates."""

    token: str
    label: int
    polygon_coords: np.ndarray
    object_type: str = "agent"
    speed_mps: float = 0.0


@dataclass
class SceneContext:
    """Scene data for scoring.

    `gt_*` fields are analysis-side-channel inputs only: open-loop inspection,
    diagnostics, and optional debug workflows. They are intentionally preserved so
    downstream analysis can compare predictions against human GT, but they are not
    the official v1 scoring reference.

    `pdm_*` fields carry the explicit official v1 reference context used by main
    v1 scoring and RL EP normalization when such a reference is available.
    """

    scene_token: str
    log_name: str
    ego_state: np.ndarray
    ego_past_states: np.ndarray
    observation: "PDMObservation"
    drivable_area_map: "DrivableMap"
    route_lane_ids: Set[str]
    centerline: "PDMPath"
    collided_track_ids: Set[str] = field(default_factory=set)
    gt_trajectory: Optional[np.ndarray] = None  # (T, 2/3) ego-relative GT xy or xyh waypoints for diagnostics
    gt_progress: Optional[float] = None  # precomputed GT centerline progress (raw analysis-side-channel only)
    gt_masked_progress: Optional[float] = None  # gt_progress × gt_NC × gt_DAC (analysis-side-channel only)
    pdm_trajectory: Optional[np.ndarray] = None  # (T, 2/3) ego-relative PDM reference xy or xyh waypoints
    pdm_progress: Optional[float] = None  # precomputed PDM centerline progress (raw official v1 reference)
    pdm_masked_progress: Optional[float] = None  # pdm_progress × pdm_NC × pdm_DAC (official v1 reference)
    track_object_types: Dict[str, str] = field(default_factory=dict)  # token → "agent" | "static"
    track_speeds: Dict[str, float] = field(default_factory=dict)  # token → GT speed (m/s) from annotation
    key_action_obstacles: List[KeyActionObstacle] = field(default_factory=list)
    candidate_trajectory: Optional[np.ndarray] = None  # internal candidate trajectory for progress scale
    candidate_progress: Optional[float] = None


@dataclass
class ScoringResult:
    scoring_version: str = "v1"
    pdm_score: float = 0.0
    no_at_fault_collisions: float = 1.0
    drivable_area_compliance: float = 1.0
    driving_direction_compliance: float = 1.0
    traffic_light_compliance: float = 1.0
    ego_progress: float = 0.0
    time_to_collision: float = 1.0
    lane_keeping: float = 1.0
    history_comfort: float = 1.0
    error: Optional[str] = None
    human_penalty_applied: Optional[List[str]] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RLScoringResult:
    """RL reward scoring — 8 independent sub-rewards with additive aggregation.

    All metrics are in [0, 1] range. Safety metrics can be either continuous
    (distance-based) or discrete (NavSim-compatible) depending on config.
    """

    rl_score: float = 0.0
    # Safety layer
    no_at_fault_collisions: float = 1.0
    drivable_area_compliance: float = 1.0
    driving_direction_compliance: float = 1.0
    traffic_light_compliance: float = 1.0
    # Performance layer
    ego_progress: float = 0.0
    time_to_collision: float = 1.0
    lane_keeping: float = 1.0
    history_comfort: float = 1.0
    # Raw physics for decoupled reward construction
    max_collision_overlap: float = 0.0   # worst-frame overlap_area/ego_area [0,1]
    max_collision_penetration_distance: float = 0.0  # PDMS-aligned linear intrusion proxy (meters)
    min_obstacle_distance: float = 0.0   # min distance to nearest obstacle (meters)
    min_boundary_distance: float = 0.0   # min distance to drivable boundary (meters)
    mean_obstacle_distance_5m: float = 5.0  # mean distance to obstacles within 5m (meters)
    half_lane_width: float = 2.0         # centerline to drivable boundary distance (meters)
    lateral_offset_signed: float = 0.0   # mean signed lateral offset (positive = right)
    lateral_offset_change: float = 0.0   # signed offset end minus start (positive = drifted right)
    centerline_lateral_offset_start_signed: float = 0.0
    centerline_lateral_offset_end_signed: float = 0.0
    centerline_distance_mean: float = 0.0
    centerline_distance_max: float = 0.0
    local_centerline_points: List[List[float]] = field(default_factory=list)
    boundary_distance_start: float = 0.0
    boundary_distance_end: float = 0.0
    boundary_distance_min: float = 0.0
    boundary_distance_mean: float = 0.0
    boundary_distances: List[float] = field(default_factory=list)
    boundary_side: Optional[str] = None
    nearest_boundary_side: Optional[str] = None
    nearest_boundary_distance: float = 0.0
    in_intersection_fraction: float = 0.0
    in_intersection_now: bool = False
    oncoming_fraction: float = 0.0
    non_drivable_fraction: float = 0.0
    multiple_lanes_fraction: float = 0.0
    in_intersection_flags: List[bool] = field(default_factory=list)
    oncoming_flags: List[bool] = field(default_factory=list)
    non_drivable_flags: List[bool] = field(default_factory=list)
    multiple_lanes_flags: List[bool] = field(default_factory=list)
    # Per-step directional diagnostics
    boundary_sides: List[Optional[str]] = field(default_factory=list)  # per-step: "left"/"right"/None (only when off-road)
    collision_per_step: List[Optional[dict]] = field(default_factory=list)  # per-step: {"direction": ..., "penetration": ...} or None
    # Per-waypoint progress
    progress_per_waypoint: List[float] = field(default_factory=list)  # 8 values: pred_cumulative[i] / pdm_cumulative[i]
    # Discrete mode: GRPO reward signals
    safety_gate: float = 1.0       # NC × DAC × DDC × TLC binary product
    raw_progress: float = 0.0      # centerline progress in meters, gated by safety
    pdms_score: float = 0.0        # standard v1 PDMS, for monitoring / filtering only
    pdms_no_at_fault_collisions: float = 1.0
    pdms_drivable_area_compliance: float = 1.0
    pdms_driving_direction_compliance: float = 1.0
    pdms_traffic_light_compliance: float = 1.0
    pdms_ego_progress: float = 0.0
    pdms_time_to_collision: float = 1.0
    pdms_lane_keeping: float = 1.0
    pdms_history_comfort: float = 1.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def sub_rewards(self) -> dict:
        """Return 8 independent sub-rewards for per-component GRPO advantage."""
        return {
            "nc": self.no_at_fault_collisions,
            "dac": self.drivable_area_compliance,
            "ddc": self.driving_direction_compliance,
            "tlc": self.traffic_light_compliance,
            "ep": self.ego_progress,
            "ttc": self.time_to_collision,
            "lk": self.lane_keeping,
            "hc": self.history_comfort,
        }


@dataclass
class InternalScoringResult:
    """Hard internal key-action score for internal L4 scenes."""

    internal_score: float = 0.0
    safety_score: float = 0.0
    comfort_score: float = 0.0
    key_action_score: float = 0.0
    progress_score: float = 0.0
    no_at_fault_collisions: float = 1.0
    drivable_area_compliance: float = 1.0
    sample_valid: bool = True
    invalid_reason: Optional[str] = None
    first_no_nudge_upper_bound: Optional[float] = None
    overrun_no_nudge_gate: bool = False
    num_relevant_labeled: int = 0
    num_key_actions: int = 0
    num_key_actions_passed: int = 0
    ego_front_max: float = 0.0
    ego_rear_max: float = 0.0
    raw_progress: float = 0.0
    progress_norm: float = 0.0
    progress_norm_source: str = "none"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)
