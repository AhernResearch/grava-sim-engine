from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .bicycle_model import BatchKinematicBicycleModel
from .enums import StateIndex
from .lqr_tracker import BatchLQRTracker
from .types import VehicleParams

if TYPE_CHECKING:
    from .observation import PDMObservation


class PDMSimulator:
    """Batch closed-loop proposal simulator matching NavSim's PDM rollout logic."""

    def __init__(
        self,
        discretization_time: float = 0.1,
        vehicle: VehicleParams | None = None,
    ) -> None:
        if discretization_time <= 0.0:
            raise ValueError(f"discretization_time must be positive, got {discretization_time}")

        self._discretization_time = float(discretization_time)
        self._vehicle = vehicle or VehicleParams()
        self._motion_model = BatchKinematicBicycleModel(vehicle=self._vehicle)

    @staticmethod
    def _extract_pose(state: np.ndarray) -> np.ndarray:
        return np.asarray(
            state[[StateIndex.X, StateIndex.Y, StateIndex.HEADING]],
            dtype=np.float64,
        )

    def _resolve_dt(self, observation: "PDMObservation" | None) -> float:
        if observation is None:
            return self._discretization_time

        interval_time = float(observation.interval_time)
        if interval_time <= 0.0:
            raise ValueError(f"observation interval_time must be positive, got {interval_time}")
        return interval_time

    def simulate_proposals(
        self,
        ego_state: np.ndarray,
        proposals: np.ndarray,
        observation: "PDMObservation" | None = None,
    ) -> np.ndarray:
        """Simulate proposals. Expects proposals to include t=0 ego pose.

        Input: proposals (B, T, 3) where T should be num_poses+1 (e.g. 41 for 4s @ 0.1s).
        Output: simulated_states (B, T, 11).
        """
        ego_state = np.asarray(ego_state, dtype=np.float64)
        proposals = np.asarray(proposals, dtype=np.float64)

        if ego_state.shape != (StateIndex.size(),):
            raise ValueError(f"ego_state must have shape [{StateIndex.size()}], got {ego_state.shape}")
        if proposals.ndim != 3 or proposals.shape[-1] != 3:
            raise ValueError(f"proposals must have shape [B, T, 3], got {proposals.shape}")
        if proposals.shape[1] == 0:
            raise ValueError("proposals must contain at least one pose")

        dt = self._resolve_dt(observation)

        # Proposals already include t=0 ego pose — use directly as reference.
        batch_size, num_steps, _ = proposals.shape

        simulated_states = np.zeros((batch_size, num_steps, StateIndex.size()), dtype=np.float64)
        simulated_states[:, 0, :] = ego_state[None, :]
        tracker = BatchLQRTracker(discretization_time=dt, vehicle=self._vehicle)
        tracker.update(proposals)

        for time_idx in range(num_steps - 1):
            accelerations, steering_rates = tracker.track_trajectory(
                current_state=simulated_states[:, time_idx, :],
                time_idx=time_idx,
            )
            simulated_states[:, time_idx + 1, :] = self._motion_model.propagate_state(
                states=simulated_states[:, time_idx, :],
                accel_cmds=accelerations,
                steering_rate_cmds=steering_rates,
                dt=dt,
            )

        return simulated_states

    def simulate_from_controls(
        self,
        ego_state: np.ndarray,
        control_signals: np.ndarray,
        observation: "PDMObservation" | None = None,
    ) -> np.ndarray:
        """Simulate from direct control signals, bypassing LQR tracker.

        Uses zero-order hold (each 0.5s control held for 5 × 0.1s steps)
        and simple kinematic integration (no bicycle model low-pass filter).

        Args:
            ego_state: (11,) initial state vector.
            control_signals: (B, T_coarse, 2) — [accel_m/s², heading_rate_rad/s]
                at 0.5s intervals. T_coarse is typically 8 (4s horizon).
            observation: optional, for dt resolution.

        Returns:
            simulated_states: (B, T_fine+1, 11) where T_fine = T_coarse * ratio.
        """
        ego_state = np.asarray(ego_state, dtype=np.float64)
        control_signals = np.asarray(control_signals, dtype=np.float64)

        if ego_state.shape != (StateIndex.size(),):
            raise ValueError(f"ego_state must have shape [{StateIndex.size()}], got {ego_state.shape}")
        if control_signals.ndim != 3 or control_signals.shape[-1] != 2:
            raise ValueError(f"control_signals must have shape [B, T, 2], got {control_signals.shape}")

        dt = self._resolve_dt(observation)
        input_interval = 0.5
        ratio = round(input_interval / dt)

        batch_size, t_coarse, _ = control_signals.shape
        t_fine = t_coarse * ratio  # 40 steps for 8 controls

        states = np.zeros((batch_size, t_fine + 1, StateIndex.size()), dtype=np.float64)
        states[:, 0, :] = ego_state[None, :]

        for t in range(t_fine):
            coarse_idx = min(t // ratio, t_coarse - 1)
            accel = control_signals[:, coarse_idx, 0]
            heading_rate = control_signals[:, coarse_idx, 1]

            x = states[:, t, StateIndex.X]
            y = states[:, t, StateIndex.Y]
            heading = states[:, t, StateIndex.HEADING]
            velocity = states[:, t, StateIndex.VELOCITY_X]

            new_velocity = np.clip(velocity + accel * dt, 0.0, None)
            new_heading = heading + heading_rate * dt
            new_x = x + new_velocity * np.cos(new_heading) * dt
            new_y = y + new_velocity * np.sin(new_heading) * dt

            states[:, t + 1, StateIndex.X] = new_x
            states[:, t + 1, StateIndex.Y] = new_y
            states[:, t + 1, StateIndex.HEADING] = new_heading
            states[:, t + 1, StateIndex.VELOCITY_X] = new_velocity
            states[:, t + 1, StateIndex.ACCELERATION_X] = accel
            states[:, t + 1, StateIndex.ANGULAR_VELOCITY] = heading_rate

        return states
