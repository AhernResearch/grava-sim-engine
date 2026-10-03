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

    half_length: float = 2.588
    half_width: float = 1.1485
    rear_axle_to_center: float = 1.461
    wheel_base: float = 3.089


@dataclass
class KeyActionObstacle:
    """Key-action obstacle annotation in sim-engine coordinates."""

    token: str
    label: int
    polygon_coords: np.ndarray
    object_type: str = "agent"
    speed_mps: float = 0.0


@dataclass
class SceneContext:
    """Scene geometry, observations and reference trajectories.

    PDM fields supply the PDMS/RL normalization reference. GT fields support
    open-loop analysis and CDS scene preparation."""

    scene_token: str
    log_name: str
    ego_state: np.ndarray
    ego_past_states: np.ndarray
    observation: "PDMObservation"
    drivable_area_map: "DrivableMap"
    route_lane_ids: Set[str]
    centerline: "PDMPath"
    collided_track_ids: Set[str] = field(default_factory=set)
    gt_trajectory: Optional[np.ndarray] = None
    gt_progress: Optional[float] = None
    gt_masked_progress: Optional[float] = None
    pdm_trajectory: Optional[np.ndarray] = None
    pdm_progress: Optional[float] = None
    pdm_masked_progress: Optional[float] = None
    track_object_types: Dict[str, str] = field(default_factory=dict)
    track_speeds: Dict[str, float] = field(default_factory=dict)
    key_action_obstacles: List[KeyActionObstacle] = field(default_factory=list)
    candidate_trajectory: Optional[np.ndarray] = None
    candidate_progress: Optional[float] = None


@dataclass
class ScoringResult:
    pdm_score: float = 0.0
    no_at_fault_collisions: float = 1.0
    drivable_area_compliance: float = 1.0
    driving_direction_compliance: float = 1.0
    traffic_light_compliance: float = 1.0
    ego_progress: float = 0.0
    time_to_collision: float = 1.0
    lane_keeping: float = 1.0
    history_comfort: float = 1.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RLScoringResult:
    """Driving rewards, scene diagnostics and discrete PDMS monitoring."""

    rl_score: float = 0.0
    no_at_fault_collisions: float = 1.0
    drivable_area_compliance: float = 1.0
    driving_direction_compliance: float = 1.0
    traffic_light_compliance: float = 1.0
    ego_progress: float = 0.0
    time_to_collision: float = 1.0
    lane_keeping: float = 1.0
    history_comfort: float = 1.0
    max_collision_overlap: float = 0.0
    max_collision_penetration_distance: float = 0.0
    min_obstacle_distance: float = 0.0
    min_boundary_distance: float = 0.0
    mean_obstacle_distance_5m: float = 5.0
    half_lane_width: float = 2.0
    lateral_offset_signed: float = 0.0
    lateral_offset_change: float = 0.0
    centerline_lateral_offset_start_signed: float = 0.0
    centerline_lateral_offset_end_signed: float = 0.0
    centerline_distance_mean: float = 0.0
    centerline_distance_max: float = 0.0
    local_centerline_points: List[List[float]] = field(default_factory=list)
    boundary_distance_start: float = 0.0
    boundary_distance_end: float = 0.0
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
    boundary_sides: List[Optional[str]] = field(default_factory=list)
    collision_per_step: List[Optional[dict]] = field(default_factory=list)
    progress_per_waypoint: List[float] = field(default_factory=list)
    safety_gate: float = 1.0
    gated_progress: float | None = None

    pdm_score: float = 0.0
    pdms_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CDSScoringResult:
    """Closed-loop Driving Score and its component metrics."""

    cds: float = 0.0
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

    def to_dict(self) -> dict:
        return dict(self.__dict__)
