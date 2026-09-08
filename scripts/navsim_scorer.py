#!/usr/bin/env python3
"""
End-to-end validation: compare grava-sim-engine PDM scores against NavSim reference.

Requires NavSim environment (.venv-navsim) for MetricCache loading AND reference scoring.

Usage:
    python scripts/navsim_scorer.py \
        --metric-cache-dir /path/to/metric_cache \
        --log-name 2021.05.12.22.00.38_veh-35_01008_01518 \
        --scene-token abc123def456

    # With custom trajectory (JSON array of [x,y] pairs):
    python scripts/navsim_scorer.py \
        --metric-cache-dir /path/to/metric_cache \
        --log-name ... --scene-token ... \
        --trajectory '[[0.1, 2.0], [0.2, 4.0], ...]'
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import interp1d
except ImportError:  # pragma: no cover - depends on validation environment
    interp1d = None

try:
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer as NavSimPDMScorer,
        PDMScorerConfig as NavSimPDMScorerConfig,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
        PDMSimulator,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
except ImportError:  # pragma: no cover - depends on NavSim environment
    Trajectory = None
    pdm_score = None
    PDMSimulator = None
    NavSimPDMScorer = None
    NavSimPDMScorerConfig = None
    TrajectorySampling = None

# ── grava-sim-engine imports ──────────────────────────────────────────────
from grava_sim_engine.adapters.navsim.cache_loader import (
    load_metric_cache,
    metric_cache_to_scene_context,
)
from grava_sim_engine.core.scorer import PDMScorer, PDMScorerConfig


def _extract_trajectory_from_cache(metric_cache, scene_context=None) -> np.ndarray:
    """Extract trajectory waypoints from MetricCache's PDM-Closed trajectory.

    Returns ego-relative (x, y) waypoints, trimmed to 50 points (5s at 10Hz).
    """
    traj = metric_cache.trajectory
    states = list(traj.get_sampled_trajectory())
    global_xy = np.array(
        [[s.rear_axle.x, s.rear_axle.y] for s in states],
        dtype=np.float64,
    )

    if scene_context is not None:
        # Convert global → ego-relative coordinates
        ego_x, ego_y = scene_context.ego_state[0], scene_context.ego_state[1]
        ego_heading = scene_context.ego_state[2]
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        dx = global_xy[:, 0] - ego_x
        dy = global_xy[:, 1] - ego_y
        local_xy = np.column_stack([
            dx * cos_h - dy * sin_h,
            dx * sin_h + dy * cos_h,
        ])
        # Skip first point (ego position) and take up to 50 points
        return local_xy[1:51]

    return global_xy


def _print_result(label: str, scores: dict[str, float]) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    keys = [
        ("pdm_score", "PDM Score"),
        ("no_at_fault_collisions", "No At-Fault Collisions"),
        ("drivable_area_compliance", "Drivable Area Compliance"),
        ("driving_direction_compliance", "Driving Direction Compliance"),
        ("traffic_light_compliance", "Traffic Light Compliance"),
        ("ego_progress", "Ego Progress"),
        ("time_to_collision", "Time to Collision"),
        ("lane_keeping", "Lane Keeping"),
        ("history_comfort", "History Comfort"),
    ]
    for key, name in keys:
        val = scores.get(key, "N/A")
        if isinstance(val, float):
            print(f"  {name:<35s} {val:.6f}")
        else:
            print(f"  {name:<35s} {val}")
    if scores.get("error"):
        print(f"  {'ERROR':<35s} {scores['error']}")


def _compare_results(ours: dict, ref: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"  Comparison (ours - navsim)")
    print(f"{'=' * 50}")
    keys = [
        "pdm_score", "no_at_fault_collisions", "drivable_area_compliance",
        "driving_direction_compliance", "traffic_light_compliance",
        "ego_progress", "time_to_collision", "lane_keeping", "history_comfort",
    ]
    max_diff = 0.0
    for key in (key for key in keys if key in ref):
        ours_val = ours.get(key, 0.0)
        ref_val = ref.get(key, 0.0)
        diff = ours_val - ref_val
        max_diff = max(max_diff, abs(diff))
        flag = " " if abs(diff) < 1e-4 else "*"
        print(f"  {flag} {key:<35s} {diff:+.6f}  ({ours_val:.6f} vs {ref_val:.6f})")
    print(f"\n  Max absolute difference: {max_diff:.6f}")
    if max_diff <= 1e-4:
        print("  ✓ PASS (<= 1e-4)")
    else:
        print("  FAIL (> 1e-4)")


def _run_navsim_reference(metric_cache, waypoints: np.ndarray) -> dict:
    """Compute pinned NAVSIM v1 PDMS; fail if the official runtime is unavailable."""
    if Trajectory is None or pdm_score is None:
        raise RuntimeError("Install the pinned NAVSIM reference environment")
    points = np.asarray(waypoints, dtype=np.float64)
    if points.shape != (8, 2):
        raise ValueError("Expected 8 future waypoints at 0.5 seconds")
    delta = np.diff(np.vstack([np.zeros((1, 2)), points]), axis=0)
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    model_traj = Trajectory(
        poses=np.column_stack([points, headings]),
        trajectory_sampling=TrajectorySampling(num_poses=8, interval_length=.5),
    )
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    result = pdm_score(
        metric_cache=metric_cache, model_trajectory=model_traj, future_sampling=sampling,
        simulator=PDMSimulator(proposal_sampling=sampling),
        scorer=NavSimPDMScorer(proposal_sampling=sampling, config=NavSimPDMScorerConfig()),
    )
    return {
        "pdm_score": float(result.score),
        "no_at_fault_collisions": float(result.no_at_fault_collisions),
        "drivable_area_compliance": float(result.drivable_area_compliance),
        "driving_direction_compliance": float(result.driving_direction_compliance),
        "ego_progress": float(result.ego_progress),
        "time_to_collision": float(result.time_to_collision_within_bound),
        "history_comfort": float(result.comfort),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate grava-sim-engine against NavSim reference")
    parser.add_argument("--metric-cache-dir", required=True, help="MetricCache root directory")
    parser.add_argument("--log-name", required=True, help="NavSim log name")
    parser.add_argument("--scene-token", required=True, help="Scene token")
    parser.add_argument("--trajectory", default=None, help="Trajectory as JSON [[x,y], ...]")
    parser.add_argument("--no-reference", action="store_true", help="Skip NavSim reference comparison")
    args = parser.parse_args()

    # 1. Load MetricCache
    print(f"Loading MetricCache: {args.log_name}/{args.scene_token}")
    t0 = time.time()
    mc = load_metric_cache(args.metric_cache_dir, args.log_name, args.scene_token)
    print(f"  Loaded in {time.time() - t0:.2f}s")

    # 2. Convert to SceneContext
    t0 = time.time()
    scene = metric_cache_to_scene_context(mc, args.scene_token)
    print(f"  Converted to SceneContext in {time.time() - t0:.2f}s")
    print(f"  ego_state: x={scene.ego_state[0]:.2f} y={scene.ego_state[1]:.2f} "
          f"h={scene.ego_state[2]:.3f} vx={scene.ego_state[3]:.2f}")
    print(f"  past_states: {len(scene.ego_past_states)} frames")
    print(f"  route_lane_ids: {len(scene.route_lane_ids)} lanes")

    # 3. Get trajectory
    if args.trajectory:
        waypoints = np.array(json.loads(args.trajectory), dtype=np.float64)
        print(f"  Using provided trajectory: {len(waypoints)} waypoints")
    else:
        from nuplan.common.actor_state.state_representation import TimePoint
        from nuplan.common.geometry.convert import absolute_to_relative_poses
        times = [mc.ego_state.time_point.time_us + int(i * 500000) for i in range(1, 9)]
        poses = [mc.trajectory.get_state_at_time(TimePoint(t)).rear_axle for t in times]
        relative = absolute_to_relative_poses([mc.ego_state.rear_axle, *poses])[1:]
        waypoints = np.array([[p.x, p.y] for p in relative])
        print(f"  Using PDM-Closed trajectory (ego-relative): {len(waypoints)} waypoints")

    # 4. Score with grava-sim-engine
    print("\nScoring with grava-sim-engine...")
    t0 = time.time()
    scorer = PDMScorer(config=PDMScorerConfig())
    result = scorer.score(waypoints, scene)
    elapsed = time.time() - t0
    ours = result.to_dict()
    _print_result(f"grava-sim-engine (took {elapsed:.3f}s)", ours)

    if result.error:
        print(f"\nERROR: {result.error}")
        sys.exit(1)

    # 5. (Optional) NavSim reference
    if not args.no_reference:
        print("\nScoring with NavSim reference...")
        t0 = time.time()
        ref = _run_navsim_reference(mc, waypoints)
        if ref is not None:
            elapsed = time.time() - t0
            _print_result(f"NavSim reference (took {elapsed:.3f}s)", ref)
            _compare_results(ours, ref)
            if any(abs(ours.get(k, 0) - v) > 1e-4 for k, v in ref.items()):
                raise RuntimeError("Standalone scorer differs from official NAVSIM v1")
        else:
            print("  NavSim not available for comparison (missing imports)")


if __name__ == "__main__":
    main()
