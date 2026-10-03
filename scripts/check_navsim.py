"""Compare the engine with the pinned NAVSIM v1 scorer on real MetricCaches."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import navsim
import numpy as np
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator as OfficialSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from grava_sim_engine.adapters.dataset_loader import precompute_reference
from grava_sim_engine.adapters.navsim.cache_loader import (
    metric_cache_to_scene_context,
    read_metric_cache,
)
from grava_sim_engine.core.scoring import PDMSScorer
from grava_sim_engine.core.scoring.trajectory import derive_relative_headings

REFERENCE_COMMIT = "0811876c274e8b058ab2be9b3dcd4d37bd23f177"


def compare(cache_path: Path, trajectory: np.ndarray) -> dict:
    input_columns = trajectory.shape[1]
    cache = read_metric_cache(cache_path)
    scene = metric_cache_to_scene_context(cache, cache_path.parent.name)
    precompute_reference(scene)
    actual = PDMSScorer().score(trajectory, scene).to_dict()
    # XY input uses the engine's existing tangent-heading convention on both paths.
    if trajectory.shape[1] == 2:
        heading = derive_relative_headings(trajectory[None, ...])[0]
        trajectory = np.column_stack([trajectory, heading])
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    result = pdm_score(
        cache,
        Trajectory(trajectory, TrajectorySampling(num_poses=8, interval_length=0.5)),
        sampling,
        OfficialSimulator(proposal_sampling=sampling),
        PDMScorer(proposal_sampling=sampling),
    )
    expected = {
        "pdm_score": float(result.score),
        "no_at_fault_collisions": float(result.no_at_fault_collisions),
        "drivable_area_compliance": float(result.drivable_area_compliance),
        "ego_progress": float(result.ego_progress),
        "time_to_collision": float(result.time_to_collision_within_bound),
        "history_comfort": float(result.comfort),
    }
    return {
        "scene_token": scene.scene_token,
        "columns": input_columns,
        "actual": {key: actual[key] for key in expected},
        "official": expected,
        "max_delta": max(abs(actual[key] - value) for key, value in expected.items()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="JSON array: log_name, scene_token, trajectory (8x2 or 8x3)",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(navsim.__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REFERENCE_COMMIT:
        raise SystemExit(f"Expected NAVSIM {REFERENCE_COMMIT}, found {revision} in {source}")
    records = json.loads(args.cases.read_text())
    results = []
    for record in records:
        paths = list(
            (args.metric_cache / record["log_name"]).glob(
                f"*/{record['scene_token']}/metric_cache.pkl"
            )
        )
        (path,) = paths
        result = compare(path, np.asarray(record["trajectory"], dtype=np.float64))
        results.append(result)
        print(record["scene_token"], result["max_delta"], flush=True)
    args.output.write_text(
        json.dumps({"reference_commit": REFERENCE_COMMIT, "results": results}, indent=2) + "\n"
    )
    if not results or any(result["max_delta"] > args.tolerance for result in results):
        raise SystemExit("NAVSIM v1 comparison failed; see the output report")


if __name__ == "__main__":
    main()
