"""HTTP validation, scoring dispatch and response formatting."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from grava_sim_engine import __version__
from grava_sim_engine.adapters.dataset_loader import Dataset, SceneNotFound, SceneStore
from grava_sim_engine.core.scoring import (
    CDSScorer,
    CDSScorerConfig,
    PDMSScorer,
    RLScorer,
    RLScorerConfig,
)
from grava_sim_engine.core.scoring.trajectory import derive_relative_headings
from grava_sim_engine.server.schemas import (
    BatchControlScoreRequest,
    BatchScoreRequest,
    ControlScoreRequest,
    SceneRequest,
    ScoreRequest,
    TrajectoryRequest,
)
from grava_sim_engine.utils.warmup import read_warmup

logger = logging.getLogger(__name__)
DETAIL_FIELDS = {
    "local_centerline_points",
    "boundary_distances",
    "in_intersection_flags",
    "oncoming_flags",
    "non_drivable_flags",
    "multiple_lanes_flags",
    "boundary_sides",
    "collision_per_step",
    "progress_per_waypoint",
}


def with_heading(trajectory: np.ndarray) -> np.ndarray:
    if trajectory.shape[1] == 3:
        return trajectory
    heading = derive_relative_headings(trajectory[None, ...])[0]
    return np.column_stack([trajectory, heading])


def serialize_score(result, include_details: bool) -> dict:
    response = asdict(result)
    details = {key: response.pop(key) for key in DETAIL_FIELDS if key in response}
    if response.get("gated_progress", 0) is None:
        del response["gated_progress"]
    if include_details and details:
        response["details"] = details
    return response


def create_app(datasets: list[Dataset] | None = None, cache_dir: Path | None = None) -> FastAPI:
    if datasets is None:
        config = json.loads(os.environ["GRAVA_SIM_CONFIG"])
        datasets = [Dataset.open(name, path) for name, path in config["datasets"].items()]
        cache_dir = Path(config["cache_dir"])
    stores = {dataset.name: SceneStore(dataset, cache_dir) for dataset in datasets}
    app = FastAPI(title="GRAVA Sim Engine", version=__version__)

    @app.middleware("http")
    async def log_failures(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception:
            logger.exception("Scoring request failed: %s %s", request.method, request.url.path)
            raise

    def scene_for(request: SceneRequest, mode: str):
        store = stores.get(request.dataset)
        if store is None:
            raise HTTPException(404, f"Unknown dataset: {request.dataset}")
        if mode not in store.dataset.scoring_modes:
            raise HTTPException(422, f"Dataset {request.dataset} does not support {mode}")
        try:
            return store.load(request.log_name, request.scene_token)
        except SceneNotFound as exc:
            raise HTTPException(404, str(exc)) from exc

    def score_trajectories(request: TrajectoryRequest, trajectories: list) -> list[dict]:
        scene = scene_for(request, request.scoring_mode)
        # A batch may contain both XY and XYH inputs. Derive headings for XY before stacking.
        arrays = [np.asarray(trajectory, dtype=np.float64) for trajectory in trajectories]
        if len({array.shape[1] for array in arrays}) > 1:
            arrays = [with_heading(array) for array in arrays]
        batch = np.stack(arrays)
        if request.scoring_mode == "pdms":
            results = PDMSScorer().score_batch(batch, scene)
        elif request.scoring_mode == "cds":
            results = CDSScorer().score_batch(
                batch, scene, CDSScorerConfig(**request.config_overrides)
            )
        else:
            config = RLScorerConfig(safety_mode=request.scoring_mode, **request.config_overrides)
            results = RLScorer().score_batch(
                batch, scene, config, include_details=request.include_details
            )
        return [serialize_score(result, request.include_details) for result in results]

    def score_controls(request: SceneRequest, controls: list) -> list[dict]:
        scene = scene_for(request, "pdms")
        results = PDMSScorer().score_batch_from_controls(np.asarray(controls), scene)
        return [serialize_score(result, request.include_details) for result in results]

    @app.post("/v1/score")
    def score(request: ScoreRequest):
        return score_trajectories(request, [request.trajectory])[0]

    @app.post("/v1/score/batch")
    def score_batch(request: BatchScoreRequest):
        return {"results": score_trajectories(request, request.trajectories)}

    @app.post("/v1/score/control")
    def score_control(request: ControlScoreRequest):
        return score_controls(request, [request.control_signals])[0]

    @app.post("/v1/score/control/batch")
    def score_control_batch(request: BatchControlScoreRequest):
        return {"results": score_controls(request, request.control_signals_batch)}

    @app.get("/v1/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/v1/datasets")
    def list_datasets():
        return {
            "datasets": [
                {
                    "name": dataset.name,
                    "format": dataset.format,
                    "scoring_modes": dataset.scoring_modes,
                    "control_scoring": "pdms" in dataset.scoring_modes,
                }
                for dataset in datasets
            ]
        }

    @app.get("/v1/warmup")
    def warmup():
        return {"datasets": read_warmup(datasets, Path(cache_dir))}

    return app
