from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .enums import StateIndex
from .geometry import normalize_angle as _normalize_angle
from .types import VehicleParams


def forward_integrate(
    init: np.ndarray,
    delta: np.ndarray,
    sampling_time: float,
) -> np.ndarray:
    """Perform Euler integration."""
    init = np.asarray(init, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    return init + delta * float(sampling_time)


@dataclass(frozen=True)
class BicycleModelConfig:
    max_steering_angle: float = np.pi / 3
    accel_time_constant: float = 0.2
    steering_angle_time_constant: float = 0.05


class BatchKinematicBicycleModel:
    """Rear-axle-based kinematic bicycle model with first-order actuator filtering."""

    def __init__(
        self,
        vehicle: VehicleParams | None = None,
        config: BicycleModelConfig | None = None,
    ) -> None:
        self._vehicle = vehicle or VehicleParams()
        self._config = config or BicycleModelConfig()

    def get_state_dot(self, states: np.ndarray) -> np.ndarray:
        """Compute state derivatives for a batch of states."""
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != StateIndex.size():
            raise ValueError(f"states must have shape [B, {StateIndex.size()}], got {states.shape}")

        state_dots = np.zeros_like(states, dtype=np.float64)
        longitudinal_speeds = states[:, StateIndex.VELOCITY_X]
        steering_angles = states[:, StateIndex.STEERING_ANGLE]

        state_dots[:, StateIndex.X] = longitudinal_speeds * np.cos(states[:, StateIndex.HEADING])
        state_dots[:, StateIndex.Y] = longitudinal_speeds * np.sin(states[:, StateIndex.HEADING])
        state_dots[:, StateIndex.HEADING] = (
            longitudinal_speeds * np.tan(steering_angles) / self._vehicle.wheel_base
        )
        state_dots[:, StateIndex.VELOCITY_X] = states[:, StateIndex.ACCELERATION_X]
        state_dots[:, StateIndex.VELOCITY_Y] = 0.0
        state_dots[:, StateIndex.ACCELERATION_X] = 0.0
        state_dots[:, StateIndex.ACCELERATION_Y] = 0.0
        state_dots[:, StateIndex.STEERING_ANGLE] = states[:, StateIndex.STEERING_RATE]

        return state_dots

    def _update_commands(
        self,
        states: np.ndarray,
        accel_cmds: np.ndarray,
        steering_rate_cmds: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Apply first-order low-pass filtering to acceleration and steering angle commands."""
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        accel_cmds = np.asarray(accel_cmds, dtype=np.float64)
        steering_rate_cmds = np.asarray(steering_rate_cmds, dtype=np.float64)
        if accel_cmds.shape != (states.shape[0],):
            raise ValueError(f"accel_cmds must have shape [{states.shape[0]}], got {accel_cmds.shape}")
        if steering_rate_cmds.shape != (states.shape[0],):
            raise ValueError(
                f"steering_rate_cmds must have shape [{states.shape[0]}], got {steering_rate_cmds.shape}"
            )

        propagating_state = np.array(states, dtype=np.float64, copy=True)

        prev_accel = states[:, StateIndex.ACCELERATION_X]
        prev_steering_angle = states[:, StateIndex.STEERING_ANGLE]

        accel_alpha = dt / (self._config.accel_time_constant + dt)
        steering_alpha = dt / (self._config.steering_angle_time_constant + dt)

        ideal_steering_angle = prev_steering_angle + dt * steering_rate_cmds
        updated_accel = prev_accel + accel_alpha * (accel_cmds - prev_accel)
        updated_steering_angle = prev_steering_angle + steering_alpha * (
            ideal_steering_angle - prev_steering_angle
        )
        updated_steering_rate = (updated_steering_angle - prev_steering_angle) / dt

        propagating_state[:, StateIndex.ACCELERATION_X] = updated_accel
        propagating_state[:, StateIndex.ACCELERATION_Y] = 0.0
        propagating_state[:, StateIndex.STEERING_RATE] = updated_steering_rate

        return propagating_state

    def propagate_state(
        self,
        states: np.ndarray,
        accel_cmds: np.ndarray,
        steering_rate_cmds: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Propagate a batch of states forward by one Euler step."""
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != StateIndex.size():
            raise ValueError(f"states must have shape [B, {StateIndex.size()}], got {states.shape}")

        propagating_state = self._update_commands(states, accel_cmds, steering_rate_cmds, dt)
        state_dot = self.get_state_dot(propagating_state)

        output_state = np.array(states, dtype=np.float64, copy=True)
        output_state[:, StateIndex.X] = forward_integrate(
            states[:, StateIndex.X], state_dot[:, StateIndex.X], dt
        )
        output_state[:, StateIndex.Y] = forward_integrate(
            states[:, StateIndex.Y], state_dot[:, StateIndex.Y], dt
        )
        output_state[:, StateIndex.HEADING] = _normalize_angle(
            forward_integrate(states[:, StateIndex.HEADING], state_dot[:, StateIndex.HEADING], dt)
        )
        output_state[:, StateIndex.VELOCITY_X] = forward_integrate(
            states[:, StateIndex.VELOCITY_X], state_dot[:, StateIndex.VELOCITY_X], dt
        )
        output_state[:, StateIndex.VELOCITY_Y] = 0.0
        output_state[:, StateIndex.STEERING_ANGLE] = np.clip(
            forward_integrate(
                propagating_state[:, StateIndex.STEERING_ANGLE],
                state_dot[:, StateIndex.STEERING_ANGLE],
                dt,
            ),
            -self._config.max_steering_angle,
            self._config.max_steering_angle,
        )
        output_state[:, StateIndex.ACCELERATION_X] = state_dot[:, StateIndex.VELOCITY_X]
        output_state[:, StateIndex.ACCELERATION_Y] = 0.0
        output_state[:, StateIndex.ANGULAR_VELOCITY] = (
            output_state[:, StateIndex.VELOCITY_X]
            * np.tan(output_state[:, StateIndex.STEERING_ANGLE])
            / self._vehicle.wheel_base
        )
        output_state[:, StateIndex.ANGULAR_ACCELERATION] = (
            output_state[:, StateIndex.ANGULAR_VELOCITY] - states[:, StateIndex.ANGULAR_VELOCITY]
        ) / float(dt)
        output_state[:, StateIndex.STEERING_RATE] = state_dot[:, StateIndex.STEERING_ANGLE]

        return output_state
