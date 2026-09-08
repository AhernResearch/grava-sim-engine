from __future__ import annotations

"""Comfort metrics aligned with NavSim official pdm_comfort_metrics.py.

NavSim approach:
- Acceleration: read directly from state array (ACCELERATION_X/Y), savgol-smooth
- Jerk: savgol-smooth acceleration first (window=8), then savgol derivative
- Yaw rate/accel: savgol derivative of phase-unwrapped heading (window=5 always)
- Bounds: strict inequality (> / <), matching NavSim's _within_bound
- Round to 8 decimals after each signal extraction
"""

import numpy as np
from scipy.signal import savgol_filter

from .enums import StateIndex

COMFORT_THRESHOLDS = {
    "max_abs_mag_jerk": 8.37,
    "max_abs_lat_accel": 4.89,
    "max_lon_accel": 2.40,
    "min_lon_accel": -4.05,
    "max_abs_yaw_accel": 1.93,
    "max_abs_lon_jerk": 4.13,
    "max_abs_yaw_rate": 0.95,
}


# ── NavSim-aligned primitive operations ──────────────────────────────


def _phase_unwrap(headings: np.ndarray) -> np.ndarray:
    """Phase-unwrap headings so successive differences are <= pi.

    Matches NavSim's _phase_unwrap exactly.
    """
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(
        np.round(np.diff(headings, axis=-1) / two_pi), axis=-1,
    )
    return headings - two_pi * adjustments


def _extract_ego_acceleration(
    states: np.ndarray,
    acceleration_coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> np.ndarray:
    """Extract smoothed acceleration from state array.

    Matches NavSim's _extract_ego_acceleration:
    - Reads acceleration directly from state (not derived from position)
    - Applies savgol_filter for smoothing
    """
    if acceleration_coordinate == "x":
        acceleration = states[..., StateIndex.ACCELERATION_X].copy()
    elif acceleration_coordinate == "y":
        acceleration = states[..., StateIndex.ACCELERATION_Y].copy()
    elif acceleration_coordinate == "magnitude":
        acceleration = np.hypot(
            states[..., StateIndex.ACCELERATION_X],
            states[..., StateIndex.ACCELERATION_Y],
        )
    else:
        raise ValueError(f"Unknown acceleration_coordinate: {acceleration_coordinate}")

    n_time = states.shape[-2]
    w = min(window_length, n_time)
    # savgol requires odd window >= polyorder + 1
    if w % 2 == 0:
        w = max(w - 1, 1)
    if w >= poly_order + 1:
        acceleration = savgol_filter(
            acceleration, polyorder=poly_order, window_length=w, axis=-1,
        )
    return np.round(acceleration, decimals=decimals)


def _approximate_derivatives(
    y: np.ndarray,
    time_steps_s: np.ndarray,
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> np.ndarray:
    """Savgol derivative matching NavSim's _approximate_derivatives."""
    n = y.shape[axis]
    w = min(window_length, n)
    if w % 2 == 0:
        w = max(w - 1, 1)
    if not (poly_order < w):
        raise ValueError(f"{poly_order} < {w} does not hold!")

    dx = np.diff(time_steps_s, axis=-1)
    dx = dx.mean()

    return savgol_filter(
        y, polyorder=poly_order, window_length=w, deriv=deriv_order, delta=dx, axis=axis,
    )


def _extract_ego_jerk(
    states: np.ndarray,
    acceleration_coordinate: str,
    time_steps_s: np.ndarray,
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> np.ndarray:
    """Extract jerk: smooth accel first (default window=8), then derivative.

    Matches NavSim's _extract_ego_jerk.
    """
    n_time = states.shape[-2]
    ego_acceleration = _extract_ego_acceleration(
        states, acceleration_coordinate=acceleration_coordinate,
    )
    jerk = _approximate_derivatives(
        ego_acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    return np.round(jerk, decimals=decimals)


def _extract_ego_yaw_rate(
    states: np.ndarray,
    time_steps_s: np.ndarray,
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,  # noqa: ARG001 — accepted but NOT forwarded (NavSim bug)
) -> np.ndarray:
    """Extract yaw rate/accel from heading.

    IMPORTANT: NavSim's _extract_ego_yaw_rate has window_length param but does NOT
    forward it to _approximate_derivatives, so yaw rate/accel always use default
    window=5. We replicate this behavior exactly.
    """
    ego_headings = states[..., StateIndex.HEADING]
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(ego_headings),
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        # window_length NOT forwarded — uses default=5 (NavSim behavior)
    )
    return np.round(ego_yaw_rate, decimals=decimals)


# ── Bound checks (strict inequality, matching NavSim) ───────────────


def _within_bound(
    metric: np.ndarray,
    min_bound: float | None = None,
    max_bound: float | None = None,
) -> np.ndarray:
    """Check if all values along last axis are strictly within bounds.

    NavSim uses strict inequality: (metric > min_bound) & (metric < max_bound).
    """
    lo = min_bound if min_bound is not None else float(-np.inf)
    hi = max_bound if max_bound is not None else float(np.inf)
    return np.all((metric > lo) & (metric < hi), axis=-1)


# ── Per-metric compute functions (NavSim signature) ─────────────────


def _compute_lon_acceleration(states: np.ndarray, time_steps_s: np.ndarray) -> np.ndarray:
    n_time = states.shape[-2]
    lon_acceleration = _extract_ego_acceleration(states, "x", window_length=n_time)
    return _within_bound(lon_acceleration, min_bound=-4.05, max_bound=2.40)


def _compute_lat_acceleration(states: np.ndarray, time_steps_s: np.ndarray) -> np.ndarray:
    n_time = states.shape[-2]
    lat_acceleration = _extract_ego_acceleration(states, "y", window_length=n_time)
    return _within_bound(lat_acceleration, min_bound=-4.89, max_bound=4.89)


def _compute_jerk_metric(states: np.ndarray, time_steps_s: np.ndarray) -> np.ndarray:
    n_time = states.shape[-2]
    jerk = _extract_ego_jerk(states, "magnitude", time_steps_s, window_length=n_time)
    return _within_bound(jerk, min_bound=-8.37, max_bound=8.37)


def _compute_lon_jerk_metric(states: np.ndarray, time_steps_s: np.ndarray) -> np.ndarray:
    n_time = states.shape[-2]
    lon_jerk = _extract_ego_jerk(states, "x", time_steps_s, window_length=n_time)
    return _within_bound(lon_jerk, min_bound=-4.13, max_bound=4.13)


def _compute_yaw_accel(states: np.ndarray, time_steps_s: np.ndarray) -> np.ndarray:
    n_time = states.shape[-2]
    yaw_accel = _extract_ego_yaw_rate(
        states, time_steps_s, deriv_order=2, poly_order=3, window_length=n_time,
    )
    return _within_bound(yaw_accel, min_bound=-1.93, max_bound=1.93)


def _compute_yaw_rate(states: np.ndarray, time_steps_s: np.ndarray) -> np.ndarray:
    n_time = states.shape[-2]
    yaw_rate = _extract_ego_yaw_rate(states, time_steps_s, window_length=n_time)
    return _within_bound(yaw_rate, min_bound=-0.95, max_bound=0.95)


# ── Public API ───────────────────────────────────────────────────────


def ego_is_comfortable(
    states: np.ndarray,
    time_points_s: np.ndarray,
) -> bool:
    """Evaluate comfort thresholds (NavSim-aligned, batch or single).

    Accepts states with shape [T, 11] (single) or [B, T, 11] (batch).
    For batch input, returns True only if ALL proposals are comfortable.
    """
    states = np.asarray(states, dtype=np.float64)
    time_points_s = np.asarray(time_points_s, dtype=np.float64)

    if states.ndim == 2:
        states = states[None, ...]  # [1, T, 11]

    n_batch, n_time, n_states = states.shape
    assert n_time == len(time_points_s)
    assert n_states == StateIndex.size()

    if n_time < 3:
        return True

    comfort_fns = [
        _compute_lon_acceleration,
        _compute_lat_acceleration,
        _compute_jerk_metric,
        _compute_lon_jerk_metric,
        _compute_yaw_accel,
        _compute_yaw_rate,
    ]
    results = np.zeros((n_batch, len(comfort_fns)), dtype=np.bool_)
    for idx, fn in enumerate(comfort_fns):
        results[:, idx] = fn(states, time_points_s)

    return bool(np.all(results))


def ego_comfort_violation(
    states: np.ndarray,
    time_points_s: np.ndarray,
) -> float:
    """Continuous comfort metric: 1.0 = fully comfortable, 0.0 = severely violated.

    Uses NavSim-aligned signal extraction, then maps max violation ratio to [0, 1].
    """
    states = np.asarray(states, dtype=np.float64)
    time_points_s = np.asarray(time_points_s, dtype=np.float64)

    if states.ndim == 2:
        states = states[None, ...]

    n_batch, n_time, n_states = states.shape
    if n_time < 3:
        return 1.0

    # Extract signals using NavSim-aligned functions
    lon_accel = _extract_ego_acceleration(states, "x", window_length=n_time)
    lat_accel = _extract_ego_acceleration(states, "y", window_length=n_time)
    mag_jerk = _extract_ego_jerk(states, "magnitude", time_points_s, window_length=n_time)
    lon_jerk = _extract_ego_jerk(states, "x", time_points_s, window_length=n_time)
    yaw_accel = _extract_ego_yaw_rate(
        states, time_points_s, deriv_order=2, poly_order=3, window_length=n_time,
    )
    yaw_rate = _extract_ego_yaw_rate(states, time_points_s, window_length=n_time)

    t = COMFORT_THRESHOLDS
    violation_ratios = [
        np.max(np.abs(mag_jerk) / t["max_abs_mag_jerk"] - 1.0),
        np.max(np.abs(lat_accel) / t["max_abs_lat_accel"] - 1.0),
        np.max(lon_accel / t["max_lon_accel"] - 1.0),
        np.max(-lon_accel / (-t["min_lon_accel"]) - 1.0),
        np.max(np.abs(yaw_accel) / t["max_abs_yaw_accel"] - 1.0),
        np.max(np.abs(lon_jerk) / t["max_abs_lon_jerk"] - 1.0),
        np.max(np.abs(yaw_rate) / t["max_abs_yaw_rate"] - 1.0),
    ]

    max_violation = max(0.0, float(max(violation_ratios)))
    return 1.0 - min(max_violation, 1.0)
