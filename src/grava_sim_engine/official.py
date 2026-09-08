"""Official NAVSIM v1 service, run in its separate pinned CPU environment."""
from __future__ import annotations

import inspect
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel


class ScoreRequest(BaseModel):
    trajectory: list[list[float]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"


class BatchScoreRequest(BaseModel):
    trajectories: list[list[list[float]]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"


def verify_runtime():
    from navsim.evaluate.pdm_score import pdm_score
    if "traffic_agents_policy" in inspect.signature(pdm_score).parameters:
        raise RuntimeError("Expected pinned NAVSIM v1; a v2 scoring runtime was imported")
    return pdm_score


def score_metric_cache(metric_cache, waypoints) -> dict:
    from navsim.common.dataclasses import Trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    points = np.asarray(waypoints, dtype=np.float64)
    if points.shape != (8, 2) or not np.isfinite(points).all():
        raise ValueError("Official v1 scoring requires eight finite future xy points")
    deltas = np.diff(np.vstack([np.zeros((1, 2)), points]), axis=0)
    poses = np.column_stack([points, np.arctan2(deltas[:, 1], deltas[:, 0])])
    model = Trajectory(poses=poses,
                       trajectory_sampling=TrajectorySampling(num_poses=8, interval_length=.5))
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    result = verify_runtime()(
        metric_cache=metric_cache, model_trajectory=model, future_sampling=sampling,
        simulator=PDMSimulator(proposal_sampling=sampling),
        scorer=PDMScorer(proposal_sampling=sampling),
    )
    return {"scoring_version": "v1", "pdm_score": float(result.score),
            "no_at_fault_collisions": float(result.no_at_fault_collisions),
            "drivable_area_compliance": float(result.drivable_area_compliance),
            "driving_direction_compliance": float(result.driving_direction_compliance),
            "ego_progress": float(result.ego_progress),
            "time_to_collision": float(result.time_to_collision_within_bound),
            "history_comfort": float(result.comfort),
            # These gates are not part of the v1 protocol; neutral values maintain the API.
            "traffic_light_compliance": 1.0, "lane_keeping": 1.0}


def create_app():
    verify_runtime()
    from fastapi import FastAPI, HTTPException
    from grava_sim_engine.adapters.navsim.cache_loader import load_metric_cache

    datasets = dict(item.split("=", 1) for item in os.environ.get("SIM_ENGINE_DATASETS", "").split(",") if item)
    if not datasets:
        raise ValueError("Register at least one dataset using --dataset NAME=PATH")
    cached_load = lru_cache(maxsize=32)(load_metric_cache)
    app = FastAPI(title="GRAVA official NAVSIM v1 scoring")

    def compute(trajectory, payload):
        if payload.scoring_version != "v1":
            raise HTTPException(422, "This service implements NAVSIM v1 only")
        if payload.dataset not in datasets:
            raise HTTPException(404, f"Unknown dataset: {payload.dataset}")
        try:
            root = Path(datasets[payload.dataset])
            if not any((root / payload.log_name / kind / payload.scene_token / "metric_cache.pkl").is_file()
                       for kind in ("unknown", "original", "synthetic")):
                raise FileNotFoundError("No official cache matches the requested log and scene")
            cache = cached_load(datasets[payload.dataset], payload.log_name, payload.scene_token)
            return score_metric_cache(cache, trajectory)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/score")
    def score(payload: ScoreRequest):
        return compute(payload.trajectory, payload)

    @app.post("/v1/score/batch")
    def score_batch(payload: BatchScoreRequest):
        return {"results": [compute(t, payload) for t in payload.trajectories]}

    @app.get("/v1/health")
    def health():
        return {"status": "ok", "backend": "official_navsim_v1", "datasets": list(datasets)}

    return app
