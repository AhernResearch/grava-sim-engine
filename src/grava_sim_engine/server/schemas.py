from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

from grava_sim_engine.core.types import InternalScoringResult, RLScoringResult, ScoringResult


class ScoreRequest(BaseModel):
    trajectory: List[List[float]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"
    include_ego: bool = False


class BatchScoreRequest(BaseModel):
    trajectories: List[List[List[float]]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"
    include_ego: bool = False


class ScoreResponse(BaseModel):
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
    error: str | None = None

    @classmethod
    def from_result(cls, result: ScoringResult) -> "ScoreResponse":
        return cls(
            scoring_version=result.scoring_version,
            pdm_score=result.pdm_score,
            no_at_fault_collisions=result.no_at_fault_collisions,
            drivable_area_compliance=result.drivable_area_compliance,
            driving_direction_compliance=result.driving_direction_compliance,
            traffic_light_compliance=result.traffic_light_compliance,
            ego_progress=result.ego_progress,
            time_to_collision=result.time_to_collision,
            lane_keeping=result.lane_keeping,
            history_comfort=result.history_comfort,
            error=result.error,
        )


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]


# ── Control Signal Scoring Schemas ─────────────────────────────────────


class ControlScoreRequest(BaseModel):
    """Score from direct control signals, bypassing LQR controller."""
    control_signals: List[List[float]]  # 8 × [accel_m/s², heading_rate_rad/s]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"
    include_ego: bool = False


class BatchControlScoreRequest(BaseModel):
    control_signals_batch: List[List[List[float]]]  # B × 8 × [accel, heading_rate]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"
    include_ego: bool = False


# ── RL Scoring Schemas ──────────────────────────────────────────────────


class RLConfigOverrides(BaseModel):
    """Optional per-request overrides for RLScorerConfig fields."""

    # Active metrics
    nc_weight: Optional[float] = None
    dac_weight: Optional[float] = None
    ep_weight: Optional[float] = None
    ttc_weight: Optional[float] = None
    hc_weight: Optional[float] = None
    # Inactive in v1 by default (still overridable)
    ddc_weight: Optional[float] = None
    tlc_weight: Optional[float] = None
    lk_weight: Optional[float] = None
    collision_distance_scale: Optional[float] = None
    dac_margin: Optional[float] = None
    tlc_margin: Optional[float] = None
    progress_distance_threshold: Optional[float] = None
    ttc_horizon: Optional[float] = None
    lane_keeping_deviation: Optional[float] = None
    lane_keeping_max_deviation: Optional[float] = None


class RLScoreRequest(BaseModel):
    trajectory: List[List[float]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_mode: Literal["continuous", "discrete"] = "continuous"
    config_overrides: Optional[RLConfigOverrides] = None
    include_ego: bool = False


class BatchRLScoreRequest(BaseModel):
    trajectories: List[List[List[float]]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_mode: Literal["continuous", "discrete"] = "continuous"
    config_overrides: Optional[RLConfigOverrides] = None
    include_ego: bool = False


# ── Internal Key-Action Scoring Schemas ────────────────────────────────────


class InternalConfigOverrides(BaseModel):
    follow_margin: Optional[float] = None
    nudge_trigger_margin: Optional[float] = None
    min_no_nudge_upper_bound: Optional[float] = None
    no_nudge_gate_min_front_gap: Optional[float] = None
    pass_margin: Optional[float] = None
    s_min: Optional[float] = None
    d_back: Optional[float] = None
    interaction_corridor_half_width: Optional[float] = None
    s_horizon_min: Optional[float] = None
    key_action_horizon_cap: Optional[float] = None
    key_action_reachable_margin: Optional[float] = None
    w_key: Optional[float] = None
    w_progress: Optional[float] = None
    w_comfort: Optional[float] = None


class InternalScoreRequest(BaseModel):
    trajectory: List[List[float]]
    scene_token: str
    log_name: str
    dataset: str
    config_overrides: Optional[InternalConfigOverrides] = None
    include_ego: bool = False


class BatchInternalScoreRequest(BaseModel):
    trajectories: List[List[List[float]]]
    scene_token: str
    log_name: str
    dataset: str
    config_overrides: Optional[InternalConfigOverrides] = None
    include_ego: bool = False


class InternalScoreResponse(BaseModel):
    internal_score: float = 0.0
    safety_score: float = 0.0
    comfort_score: float = 0.0
    key_action_score: float = 0.0
    progress_score: float = 0.0
    no_at_fault_collisions: float = 1.0
    drivable_area_compliance: float = 1.0
    sample_valid: bool = True
    invalid_reason: str | None = None
    first_no_nudge_upper_bound: float | None = None
    overrun_no_nudge_gate: bool = False
    num_relevant_labeled: int = 0
    num_key_actions: int = 0
    num_key_actions_passed: int = 0
    ego_front_max: float = 0.0
    ego_rear_max: float = 0.0
    raw_progress: float = 0.0
    progress_norm: float = 0.0
    progress_norm_source: str = "none"
    error: str | None = None

    @classmethod
    def from_result(cls, result: InternalScoringResult) -> "InternalScoreResponse":
        return cls(
            internal_score=float(result.internal_score),
            safety_score=float(result.safety_score),
            comfort_score=float(result.comfort_score),
            key_action_score=float(result.key_action_score),
            progress_score=float(result.progress_score),
            no_at_fault_collisions=float(result.no_at_fault_collisions),
            drivable_area_compliance=float(result.drivable_area_compliance),
            sample_valid=bool(result.sample_valid),
            invalid_reason=result.invalid_reason,
            first_no_nudge_upper_bound=result.first_no_nudge_upper_bound,
            overrun_no_nudge_gate=bool(result.overrun_no_nudge_gate),
            num_relevant_labeled=int(result.num_relevant_labeled),
            num_key_actions=int(result.num_key_actions),
            num_key_actions_passed=int(result.num_key_actions_passed),
            ego_front_max=_round2(result.ego_front_max),
            ego_rear_max=_round2(result.ego_rear_max),
            raw_progress=_round2(result.raw_progress),
            progress_norm=_round2(result.progress_norm),
            progress_norm_source=result.progress_norm_source,
            error=result.error,
        )


class BatchInternalScoreResponse(BaseModel):
    results: List[InternalScoreResponse]


def _round2(value: float) -> float:
    return round(float(value), 2)


def _round2_list(values: list[float]) -> list[float]:
    return [_round2(v) for v in values]


def _round2_points(points: list[list[float]]) -> list[list[float]]:
    return [[_round2(x), _round2(y)] for x, y in points]


class RLScoreResponse(BaseModel):
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
    local_centerline_points: List[List[float]] = []
    boundary_distance_start: float = 0.0
    boundary_distance_end: float = 0.0
    boundary_distance_min: float = 0.0
    boundary_distance_mean: float = 0.0
    boundary_distances: List[float] = []
    boundary_side: str | None = None
    nearest_boundary_side: str | None = None
    nearest_boundary_distance: float = 0.0
    in_intersection_fraction: float = 0.0
    in_intersection_now: bool = False
    oncoming_fraction: float = 0.0
    non_drivable_fraction: float = 0.0
    multiple_lanes_fraction: float = 0.0
    in_intersection_flags: List[bool] = []
    oncoming_flags: List[bool] = []
    non_drivable_flags: List[bool] = []
    multiple_lanes_flags: List[bool] = []
    boundary_sides: List[Optional[str]] = []
    collision_per_step: List[Optional[Dict[str, object]]] = []
    progress_per_waypoint: List[float] = []
    safety_gate: float = 1.0
    raw_progress: float = 0.0
    pdms_score: float = 0.0
    pdms_no_at_fault_collisions: float = 1.0
    pdms_drivable_area_compliance: float = 1.0
    pdms_driving_direction_compliance: float = 1.0
    pdms_traffic_light_compliance: float = 1.0
    pdms_ego_progress: float = 0.0
    pdms_time_to_collision: float = 1.0
    pdms_lane_keeping: float = 1.0
    pdms_history_comfort: float = 1.0
    sub_rewards: Dict[str, float] = {}
    error: str | None = None

    @classmethod
    def from_result(cls, result: RLScoringResult) -> "RLScoreResponse":
        return cls(
            # Core metrics: full precision for V1-aligned PDMS compatibility
            rl_score=float(result.rl_score),
            no_at_fault_collisions=float(result.no_at_fault_collisions),
            drivable_area_compliance=float(result.drivable_area_compliance),
            driving_direction_compliance=float(result.driving_direction_compliance),
            traffic_light_compliance=float(result.traffic_light_compliance),
            ego_progress=float(result.ego_progress),
            time_to_collision=float(result.time_to_collision),
            lane_keeping=float(result.lane_keeping),
            history_comfort=float(result.history_comfort),
            # Physics diagnostics: round to 2 decimals for readability
            max_collision_overlap=_round2(result.max_collision_overlap),
            max_collision_penetration_distance=_round2(result.max_collision_penetration_distance),
            min_obstacle_distance=_round2(result.min_obstacle_distance),
            min_boundary_distance=_round2(result.min_boundary_distance),
            mean_obstacle_distance_5m=_round2(result.mean_obstacle_distance_5m),
            half_lane_width=_round2(result.half_lane_width),
            lateral_offset_signed=_round2(result.lateral_offset_signed),
            lateral_offset_change=_round2(result.lateral_offset_change),
            centerline_lateral_offset_start_signed=_round2(result.centerline_lateral_offset_start_signed),
            centerline_lateral_offset_end_signed=_round2(result.centerline_lateral_offset_end_signed),
            centerline_distance_mean=_round2(result.centerline_distance_mean),
            centerline_distance_max=_round2(result.centerline_distance_max),
            local_centerline_points=_round2_points(result.local_centerline_points),
            boundary_distance_start=_round2(result.boundary_distance_start),
            boundary_distance_end=_round2(result.boundary_distance_end),
            boundary_distance_min=_round2(result.boundary_distance_min),
            boundary_distance_mean=_round2(result.boundary_distance_mean),
            boundary_distances=_round2_list(result.boundary_distances),
            boundary_side=result.boundary_side,
            nearest_boundary_side=result.nearest_boundary_side,
            nearest_boundary_distance=_round2(result.nearest_boundary_distance),
            in_intersection_fraction=_round2(result.in_intersection_fraction),
            in_intersection_now=result.in_intersection_now,
            oncoming_fraction=_round2(result.oncoming_fraction),
            non_drivable_fraction=_round2(result.non_drivable_fraction),
            multiple_lanes_fraction=_round2(result.multiple_lanes_fraction),
            in_intersection_flags=result.in_intersection_flags,
            oncoming_flags=result.oncoming_flags,
            non_drivable_flags=result.non_drivable_flags,
            multiple_lanes_flags=result.multiple_lanes_flags,
            boundary_sides=result.boundary_sides,
            collision_per_step=result.collision_per_step,
            progress_per_waypoint=_round2_list(result.progress_per_waypoint),
            # Standard v1 PDMS monitoring fields: full precision
            safety_gate=float(result.safety_gate),
            raw_progress=float(result.raw_progress),
            pdms_score=float(result.pdms_score),
            pdms_no_at_fault_collisions=float(result.pdms_no_at_fault_collisions),
            pdms_drivable_area_compliance=float(result.pdms_drivable_area_compliance),
            pdms_driving_direction_compliance=float(result.pdms_driving_direction_compliance),
            pdms_traffic_light_compliance=float(result.pdms_traffic_light_compliance),
            pdms_ego_progress=float(result.pdms_ego_progress),
            pdms_time_to_collision=float(result.pdms_time_to_collision),
            pdms_lane_keeping=float(result.pdms_lane_keeping),
            pdms_history_comfort=float(result.pdms_history_comfort),
            sub_rewards={k: float(v) for k, v in result.sub_rewards().items()},
            error=result.error,
        )


class BatchRLScoreResponse(BaseModel):
    results: List[RLScoreResponse]


# ── Dataset Management Schemas ──────────────────────────────────────────


class DatasetRegisterRequest(BaseModel):
    name: str
    path: str


class DatasetListResponse(BaseModel):
    datasets: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    version: str
    cache_stats: dict[str, int]
    datasets: Dict[str, str] = {}
    boost_cache: Optional[dict] = None
