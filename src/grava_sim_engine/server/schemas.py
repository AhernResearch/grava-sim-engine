"""The public request contract; numerical failures belong to the service, not scores."""

from __future__ import annotations

from dataclasses import fields
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from grava_sim_engine.core.scoring import CDSScorerConfig, RLScorerConfig

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$", max_length=255)]
XY = tuple[FiniteFloat, FiniteFloat]
XYH = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Trajectory = (
    Annotated[list[XY], Field(min_length=8, max_length=8)]
    | Annotated[list[XYH], Field(min_length=8, max_length=8)]
)
Controls = Annotated[list[XY], Field(min_length=8, max_length=8)]
ScoringMode = Literal["pdms", "cds", "continuous", "discrete"]


class SceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Identifier
    log_name: Identifier
    scene_token: Identifier
    include_details: bool = False


class TrajectoryRequest(SceneRequest):
    scoring_mode: ScoringMode = "pdms"
    config_overrides: dict[str, FiniteFloat] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_config(self) -> TrajectoryRequest:
        config = CDSScorerConfig if self.scoring_mode == "cds" else RLScorerConfig
        allowed = {field.name for field in fields(config)} - {"safety_mode"}
        if self.scoring_mode == "pdms":
            allowed = set()
        unknown = self.config_overrides.keys() - allowed
        if unknown:
            raise ValueError(f"Unsupported {self.scoring_mode} config keys: {sorted(unknown)}")
        for key, value in self.config_overrides.items():
            if value < 0:
                raise ValueError(f"{key} must be non-negative")
        # These lengths/scales are divisors in the reward formulas.
        positive = {
            "tlc_margin",
            "obstacle_clearance_margin",
            "progress_distance_threshold",
            "ttc_horizon",
            "lane_keeping_max_deviation",
            "min_no_nudge_upper_bound",
            "s_min",
            "interaction_corridor_half_width",
        }
        for key in positive & self.config_overrides.keys():
            if self.config_overrides[key] == 0:
                raise ValueError(f"{key} must be positive")
        return self


class ScoreRequest(TrajectoryRequest):
    trajectory: Trajectory


class BatchScoreRequest(TrajectoryRequest):
    trajectories: Annotated[list[Trajectory], Field(min_length=1)]


class ControlScoreRequest(SceneRequest):
    control_signals: Controls


class BatchControlScoreRequest(SceneRequest):
    control_signals_batch: Annotated[list[Controls], Field(min_length=1)]
