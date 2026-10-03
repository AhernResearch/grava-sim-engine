"""Comfort metrics shared by the driving scorers."""

from __future__ import annotations

import numpy as np

from ..comfort import ego_comfort_violation, ego_is_comfortable
from ..types import SceneContext
from . import scene_geometry


def history_comfort(
    simulated_states: np.ndarray, scene: SceneContext, *, use_past_states: bool = True
) -> np.ndarray:
    """Discrete comfort: 1.0 if comfortable, 0.0 if not.

    Args:
        use_past_states: Prepend ego history. PDMS uses simulated states only.
    """
    scores = np.ones(len(simulated_states), dtype=np.float64)
    if use_past_states:
        past_states = np.asarray(scene.ego_past_states, dtype=np.float64)
        if len(past_states) == 0:
            return scores
    dt = scene_geometry.dt(scene)
    for proposal_idx in range(len(simulated_states)):
        if use_past_states:
            padded = np.concatenate([past_states, simulated_states[proposal_idx]], axis=0)
        else:
            padded = simulated_states[proposal_idx]
        time_points_s = np.arange(len(padded), dtype=np.float64) * dt
        scores[proposal_idx] = 1.0 if ego_is_comfortable(padded, time_points_s) else 0.0
    return scores


def hc_continuous(simulated_states: np.ndarray, scene: SceneContext) -> np.ndarray:
    """Continuous comfort metric using max violation ratio.

    Prepends ego_past_states (if available) to improve acceleration/jerk
    estimation at the start of the simulated trajectory.
    """
    scores = np.ones(len(simulated_states), dtype=np.float64)
    dt = scene_geometry.dt(scene)
    past = scene.ego_past_states
    for proposal_idx in range(len(simulated_states)):
        states = simulated_states[proposal_idx]
        if past is not None and len(past) > 0:
            states = np.concatenate([past, states], axis=0)
        time_points_s = np.arange(len(states), dtype=np.float64) * dt
        scores[proposal_idx] = ego_comfort_violation(states, time_points_s)
    return scores
