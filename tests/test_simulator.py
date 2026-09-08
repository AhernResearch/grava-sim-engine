from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from grava_sim_engine.core.enums import StateIndex
from grava_sim_engine.core.simulator import PDMSimulator


def _ego_state() -> np.ndarray:
    state = np.zeros(StateIndex.size(), dtype=np.float64)
    state[StateIndex.VELOCITY_X] = 4.0
    return state


def _proposal(kind: int) -> np.ndarray:
    steps = np.arange(41, dtype=np.float64)
    x = 0.35 * steps + 0.02 * kind * steps
    y = 0.04 * kind * np.sin(steps / 5.0)
    heading = np.arctan2(np.gradient(y), np.gradient(x))
    return np.stack([x, y, heading], axis=-1)[None, :, :]


def test_simulate_proposals_is_safe_for_shared_simulator_concurrency():
    ego_state = _ego_state()
    proposals = [_proposal(kind) for kind in range(-4, 5)]
    baselines = [
        PDMSimulator().simulate_proposals(ego_state=ego_state, proposals=proposal)
        for proposal in proposals
    ]

    shared = PDMSimulator()

    def run(index: int) -> tuple[int, np.ndarray]:
        proposal_index = index % len(proposals)
        result = shared.simulate_proposals(
            ego_state=ego_state,
            proposals=proposals[proposal_index],
        )
        return proposal_index, result

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(run, index) for index in range(200)]
        for future in as_completed(futures):
            proposal_index, result = future.result()
            np.testing.assert_allclose(result, baselines[proposal_index], rtol=0.0, atol=0.0)
