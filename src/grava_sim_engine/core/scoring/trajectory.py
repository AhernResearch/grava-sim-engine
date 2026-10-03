"""Trajectory shared by the driving scorers."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..enums import StateIndex
from ..geometry import normalize_angle
from ..types import SceneContext


def coerce_trajectories(trajectories_xy: np.ndarray) -> np.ndarray:
    trajectory_array = np.asarray(trajectories_xy, dtype=np.float64)
    if trajectory_array.ndim == 2:
        if trajectory_array.shape[1] not in (2, 3):
            raise ValueError(f"waypoints must have shape [T,2|3], got {trajectory_array.shape}")
        trajectory_array = trajectory_array[None, ...]
    elif trajectory_array.ndim != 3 or trajectory_array.shape[-1] not in (2, 3):
        raise ValueError(
            f"trajectories must have shape [B,T,2|3] or [T,2|3], got {trajectory_array.shape}"
        )
    T = trajectory_array.shape[1]
    if T != 8:
        raise ValueError(
            f"Input must be exactly {8} waypoints at 0.5s intervals (4s horizon), got {T} waypoints. Expected shape: [B, {8}, 2|3]"
        )
    return trajectory_array


def derive_relative_headings(waypoints_xy: np.ndarray) -> np.ndarray:
    """Derive headings from xy waypoints using cubic spline tangent direction.

    Simple arctan2(dy, dx) on sparse 0.5s waypoints gives the chord direction
    between consecutive points, not the tangent at each point.  On curves this
    introduces significant heading error that propagates through LQR simulation
    and causes false drivable-area violations.

    Cubic spline fit + first derivative gives the tangent heading, which is
    consistent with NavSim/RecogDrive where the agent directly outputs heading.
    """
    batch_size, num_points, _ = waypoints_xy.shape
    t = np.arange(num_points, dtype=np.float64)
    headings = np.zeros((batch_size, num_points), dtype=np.float64)
    for b in range(batch_size):
        x, y = (waypoints_xy[b, :, 0], waypoints_xy[b, :, 1])
        deltas = np.diff(waypoints_xy[b, :, :2], axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        total_dist = np.sum(segment_lengths)
        if total_dist < 1e-06:
            headings[b] = 0.0
            continue
        moving = segment_lengths > 0.001
        if not np.all(moving):
            segment_headings = np.zeros(num_points - 1, dtype=np.float64)
            segment_headings[moving] = np.arctan2(deltas[moving, 1], deltas[moving, 0])
            first_heading = float(segment_headings[np.flatnonzero(moving)[0]])
            current_heading = first_heading
            headings[b, 0] = current_heading
            for point_idx in range(1, num_points):
                if moving[point_idx - 1]:
                    current_heading = float(segment_headings[point_idx - 1])
                headings[b, point_idx] = current_heading
            continue
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        headings[b] = np.arctan2(cs_y(t, 1), cs_x(t, 1))
    return headings


def ego_to_global(waypoints_xy: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
    """Convert ego-relative waypoints to global SE2.

    Expects ego-relative waypoints in NUPLAN frame:
    - x: forward
    - y: left
    """
    batch_size, horizon, _ = waypoints_xy.shape
    ego_x = float(ego_state[StateIndex.X])
    ego_y = float(ego_state[StateIndex.Y])
    ego_heading = float(ego_state[StateIndex.HEADING])
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)
    relative_xy = waypoints_xy[..., :2]
    global_xy = np.zeros((batch_size, horizon, 2), dtype=np.float64)
    global_xy[..., 0] = relative_xy[..., 0] * cos_h - relative_xy[..., 1] * sin_h + ego_x
    global_xy[..., 1] = relative_xy[..., 0] * sin_h + relative_xy[..., 1] * cos_h + ego_y
    relative_heading = (
        waypoints_xy[..., 2]
        if waypoints_xy.shape[-1] == 3
        else derive_relative_headings(relative_xy)
    )
    headings = normalize_angle(relative_heading + ego_heading)
    return np.concatenate([global_xy, headings[..., None]], axis=-1)


def build_proposals(
    waypoints: np.ndarray,
    scene: SceneContext,
    input_interval: float = 0.5,
) -> np.ndarray:
    """Prepend the current pose and interpolate to the simulator's time step."""
    ego_state = scene.ego_state
    global_coarse = ego_to_global(waypoints, ego_state)
    sim_dt = float(scene.observation.interval_time)
    ratio = round(input_interval / sim_dt)
    if ratio <= 1:
        ego_pose = ego_state[:3]
        return np.concatenate(
            [np.broadcast_to(ego_pose, (len(global_coarse), 1, 3)), global_coarse], axis=1
        )
    batch_size, num_coarse, _ = global_coarse.shape
    ego_pose = np.array(
        [ego_state[StateIndex.X], ego_state[StateIndex.Y], ego_state[StateIndex.HEADING]],
        dtype=np.float64,
    )
    extended = np.concatenate(
        [np.broadcast_to(ego_pose, (batch_size, 1, 3)), global_coarse], axis=1
    )
    num_extended = num_coarse + 1
    num_fine = (num_extended - 1) * ratio + 1
    t_coarse = np.arange(num_extended, dtype=np.float64)
    t_fine = np.linspace(0, num_extended - 1, num_fine)
    extended_heading = np.unwrap(extended[..., 2], axis=1)
    proposals = np.zeros((batch_size, num_fine, 3), dtype=np.float64)
    for b in range(batch_size):
        proposals[b, :, 0] = np.interp(t_fine, t_coarse, extended[b, :, 0])
        proposals[b, :, 1] = np.interp(t_fine, t_coarse, extended[b, :, 1])
        proposals[b, :, 2] = normalize_angle(np.interp(t_fine, t_coarse, extended_heading[b]))
    return proposals
