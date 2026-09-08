from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np
from scipy.linalg import solve_discrete_are

from .enums import StateIndex
from .geometry import normalize_angle as _normalize_angle
from .types import VehicleParams

INITIAL_CURVATURE_PENALTY = 1e-10


def batch_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("bij,bjk->bik", a, b)


def solve_dare(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Solve the discrete algebraic Riccati equation for single or batched systems."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    if a.ndim == 2:
        return solve_discrete_are(a, b, q, r).astype(np.float64)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a and b must be 2D or 3D arrays")
    if q.ndim == 2:
        q = np.repeat(q[None, ...], a.shape[0], axis=0)
    if r.ndim == 2:
        r = np.repeat(r[None, ...], a.shape[0], axis=0)

    return np.stack(
        [solve_discrete_are(a_i, b_i, q_i, r_i) for a_i, b_i, q_i, r_i in zip(a, b, q, r)],
        axis=0,
    ).astype(np.float64)


def compute_lqr_gain(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Compute discrete-time LQR gain K for u = -Kx."""
    p = solve_dare(a, b, q, r)
    if a.ndim == 2:
        return np.linalg.solve(b.T @ p @ b + r, b.T @ p @ a).astype(np.float64)

    bt = np.transpose(b, (0, 2, 1))
    lhs = batch_matmul(batch_matmul(bt, p), b) + r
    rhs = batch_matmul(batch_matmul(bt, p), a)
    return np.stack([np.linalg.solve(lhs_i, rhs_i) for lhs_i, rhs_i in zip(lhs, rhs)], axis=0)


def _generate_profile_from_initial_condition_and_derivatives(
    initial_condition: np.ndarray,
    derivatives: np.ndarray,
    discretization_time: float,
) -> np.ndarray:
    if discretization_time <= 0.0:
        raise ValueError(f"discretization_time must be positive, got {discretization_time}")

    initial_condition = np.asarray(initial_condition, dtype=np.float64)
    derivatives = np.asarray(derivatives, dtype=np.float64)

    cumsum = np.cumsum(derivatives * discretization_time, axis=-1, dtype=np.float64)
    return initial_condition[..., None] + np.pad(cumsum, [(0, 0), (1, 0)], mode="constant")


def _get_xy_heading_displacements_from_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-1] != 3:
        raise ValueError(f"poses must have shape [B, T, 3], got {poses.shape}")
    if poses.shape[1] < 2:
        raise ValueError("poses must contain at least two time steps")

    pose_differences = np.diff(poses, axis=1)
    xy_displacements = pose_differences[..., :2]
    heading_displacements = _normalize_angle(pose_differences[..., 2])
    return xy_displacements, heading_displacements


def _make_banded_difference_matrix(number_rows: int) -> np.ndarray:
    if number_rows <= 0:
        return np.zeros((0, max(number_rows + 1, 1)), dtype=np.float64)

    banded_matrix = np.zeros((number_rows, number_rows + 1), dtype=np.float64)
    eye = np.eye(number_rows, dtype=np.float64)
    banded_matrix[:, 1:] = eye
    banded_matrix[:, :-1] = -eye
    return banded_matrix


def _fit_initial_velocity_and_acceleration_profile(
    xy_displacements: np.ndarray,
    heading_profile: np.ndarray,
    discretization_time: float,
    jerk_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if discretization_time <= 0.0:
        raise ValueError(f"discretization_time must be positive, got {discretization_time}")
    if jerk_penalty <= 0.0:
        raise ValueError(f"jerk_penalty must be positive, got {jerk_penalty}")

    xy_displacements = np.asarray(xy_displacements, dtype=np.float64)
    heading_profile = np.asarray(heading_profile, dtype=np.float64)
    if xy_displacements.ndim != 3 or xy_displacements.shape[-1] != 2:
        raise ValueError(
            f"xy_displacements must have shape [B, T-1, 2], got {xy_displacements.shape}"
        )
    if heading_profile.shape != xy_displacements.shape[:2]:
        raise ValueError(
            f"heading_profile must have shape {xy_displacements.shape[:2]}, got {heading_profile.shape}"
        )

    batch_size, num_displacements, _ = xy_displacements.shape
    y = xy_displacements.reshape(batch_size, -1)

    headings = np.array(heading_profile, dtype=np.float64, copy=False)
    a_column = np.zeros_like(y, dtype=np.float64)
    a_column[:, 0::2] = np.cos(headings)
    a_column[:, 1::2] = np.sin(headings)

    if num_displacements == 1:
        a = a_column[:, :, None] * discretization_time
        a_t = np.transpose(a, (0, 2, 1))
        x = np.einsum(
            "bij,bj->bi",
            batch_matmul(np.linalg.pinv(batch_matmul(a_t, a)), a_t),
            y,
        )
        return x[:, 0], np.empty((batch_size, 0), dtype=np.float64)

    a = np.repeat(a_column[..., None] * discretization_time**2, num_displacements, axis=2)
    a[..., 0] = a_column * discretization_time

    upper_triangle_mask = np.triu(np.ones((num_displacements, num_displacements), dtype=bool), k=1)
    upper_triangle_mask = np.repeat(upper_triangle_mask, 2, axis=0)
    a[:, upper_triangle_mask] = 0.0

    banded_matrix = _make_banded_difference_matrix(num_displacements - 2)
    r = np.block([np.zeros((len(banded_matrix), 1), dtype=np.float64), banded_matrix])
    r = np.repeat(r[None, ...], batch_size, axis=0)

    a_t = np.transpose(a, (0, 2, 1))
    r_t = np.transpose(r, (0, 2, 1))
    regularized_inverse = np.linalg.pinv(batch_matmul(a_t, a) + jerk_penalty * batch_matmul(r_t, r))
    x = np.einsum("bij,bj->bi", batch_matmul(regularized_inverse, a_t), y)

    initial_velocity = x[:, 0]
    acceleration_profile = x[:, 1:]
    return initial_velocity, acceleration_profile


def _fit_initial_curvature_and_curvature_rate_profile(
    heading_displacements: np.ndarray,
    velocity_profile: np.ndarray,
    discretization_time: float,
    curvature_rate_penalty: float,
    initial_curvature_penalty: float = INITIAL_CURVATURE_PENALTY,
) -> Tuple[np.ndarray, np.ndarray]:
    if discretization_time <= 0.0:
        raise ValueError(f"discretization_time must be positive, got {discretization_time}")
    if curvature_rate_penalty <= 0.0:
        raise ValueError(
            f"curvature_rate_penalty must be positive, got {curvature_rate_penalty}"
        )
    if initial_curvature_penalty <= 0.0:
        raise ValueError(
            f"initial_curvature_penalty must be positive, got {initial_curvature_penalty}"
        )

    heading_displacements = np.asarray(heading_displacements, dtype=np.float64)
    velocity_profile = np.asarray(velocity_profile, dtype=np.float64)
    if heading_displacements.shape != velocity_profile.shape:
        raise ValueError(
            "heading_displacements and velocity_profile must have the same shape, "
            f"got {heading_displacements.shape} and {velocity_profile.shape}"
        )

    batch_dim, dim = heading_displacements.shape
    a = np.repeat(np.tri(dim, dtype=np.float64)[None, ...], batch_dim, axis=0)
    a[:, :, 0] = velocity_profile * discretization_time
    velocity = velocity_profile * discretization_time**2
    if dim > 1:
        a[:, 1:, 1:] *= velocity[:, None, 1:].transpose(0, 2, 1)

    q = curvature_rate_penalty * np.eye(dim, dtype=np.float64)
    q[0, 0] = initial_curvature_penalty

    a_t = np.transpose(a, (0, 2, 1))
    x = np.einsum(
        "bij,bj->bi",
        batch_matmul(np.linalg.pinv(batch_matmul(a_t, a) + q[None, ...]), a_t),
        heading_displacements,
    )

    initial_curvature = x[:, 0]
    curvature_rate_profile = x[:, 1:]
    return initial_curvature, curvature_rate_profile


def get_velocity_curvature_profiles_with_derivatives_from_poses(
    poses: np.ndarray,
    discretization_time: float,
    jerk_penalty: float,
    curvature_rate_penalty: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xy_displacements, heading_displacements = _get_xy_heading_displacements_from_poses(poses)
    initial_velocity, acceleration_profile = _fit_initial_velocity_and_acceleration_profile(
        xy_displacements=xy_displacements,
        heading_profile=poses[:, :-1, 2],
        discretization_time=discretization_time,
        jerk_penalty=jerk_penalty,
    )
    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=acceleration_profile,
        discretization_time=discretization_time,
    )
    initial_curvature, curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(
        heading_displacements=heading_displacements,
        velocity_profile=velocity_profile,
        discretization_time=discretization_time,
        curvature_rate_penalty=curvature_rate_penalty,
    )
    curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_curvature,
        derivatives=curvature_rate_profile,
        discretization_time=discretization_time,
    )

    return velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile


@dataclass(frozen=True)
class LQRConfig:
    q_longitudinal: Tuple[float, ...] = (10.0,)
    r_longitudinal: Tuple[float, ...] = (1.0,)
    q_lateral: Tuple[float, ...] = (1.0, 10.0, 0.0)
    r_lateral: Tuple[float, ...] = (1.0,)
    tracking_horizon: int = 10
    stopping_velocity: float = 0.2
    jerk_penalty: float = 1e-4
    curvature_rate_penalty: float = 1e-2
    stopping_proportional_gain: float = 0.5


class LateralStateIndex(IntEnum):
    LATERAL_ERROR = 0
    HEADING_ERROR = 1
    STEERING_ANGLE = 2


class BatchLQRTracker:
    """Batch LQR tracker matching NavSim's batch_lqr behavior."""

    def __init__(
        self,
        discretization_time: float = 0.1,
        vehicle: VehicleParams | None = None,
        config: LQRConfig | None = None,
    ) -> None:
        if discretization_time <= 0.0:
            raise ValueError(f"discretization_time must be positive, got {discretization_time}")

        self._config = config or LQRConfig()
        if len(self._config.q_longitudinal) != 1 or len(self._config.r_longitudinal) != 1:
            raise ValueError("longitudinal LQR expects 1 state weight and 1 input weight")
        if len(self._config.q_lateral) != 3 or len(self._config.r_lateral) != 1:
            raise ValueError("lateral LQR expects 3 state weights and 1 input weight")
        if self._config.tracking_horizon <= 1:
            raise ValueError("tracking_horizon must be greater than 1")

        self._vehicle = vehicle or VehicleParams()
        self._discretization_time = float(discretization_time)
        self._wheel_base = self._vehicle.wheel_base
        self._q_longitudinal = float(self._config.q_longitudinal[0])
        self._r_longitudinal = float(self._config.r_longitudinal[0])
        self._q_lateral = np.diag(np.asarray(self._config.q_lateral, dtype=np.float64))
        self._r_lateral = np.diag(np.asarray(self._config.r_lateral, dtype=np.float64))
        self._proposal_states: np.ndarray | None = None
        self._velocity_profile: np.ndarray | None = None
        self._acceleration_profile: np.ndarray | None = None
        self._curvature_profile: np.ndarray | None = None
        self._curvature_rate_profile: np.ndarray | None = None
        self._profile_discretization_time: float | None = None

    @property
    def discretization_time(self) -> float:
        return self._discretization_time

    @discretization_time.setter
    def discretization_time(self, value: float) -> None:
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"discretization_time must be positive, got {value}")
        self._discretization_time = value

    def update(self, proposal_states: np.ndarray) -> None:
        proposal_states = np.asarray(proposal_states, dtype=np.float64)
        if proposal_states.ndim != 3 or proposal_states.shape[2] != 3:
            raise ValueError(f"proposal_states must have shape [B, T, 3], got {proposal_states.shape}")
        if proposal_states.shape[1] < 2:
            raise ValueError("proposal_states must contain at least two poses")

        (
            self._velocity_profile,
            self._acceleration_profile,
            self._curvature_profile,
            self._curvature_rate_profile,
        ) = get_velocity_curvature_profiles_with_derivatives_from_poses(
            poses=proposal_states,
            discretization_time=self._discretization_time,
            jerk_penalty=self._config.jerk_penalty,
            curvature_rate_penalty=self._config.curvature_rate_penalty,
        )
        self._proposal_states = proposal_states
        self._profile_discretization_time = self._discretization_time

    def track_trajectory(
        self,
        current_state: np.ndarray,
        time_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        current_state = np.asarray(current_state, dtype=np.float64)

        if current_state.ndim != 2 or current_state.shape[1] != StateIndex.size():
            raise ValueError(
                f"current_state must have shape [B, {StateIndex.size()}], got {current_state.shape}"
            )
        if self._proposal_states is None or self._velocity_profile is None or self._curvature_profile is None:
            raise RuntimeError("update() must be called before track_trajectory()")
        if self._profile_discretization_time != self._discretization_time:
            raise RuntimeError("discretization_time changed after update(); call update() again")
        if current_state.shape[0] != self._proposal_states.shape[0]:
            raise ValueError(
                "current_state batch size must match cached proposal_states batch size, "
                f"got {current_state.shape[0]} and {self._proposal_states.shape[0]}"
            )
        if not isinstance(time_idx, (int, np.integer)):
            raise TypeError(f"time_idx must be an integer, got {type(time_idx).__name__}")
        if time_idx < 0 or time_idx >= self._proposal_states.shape[1] - 1:
            raise ValueError(
                "time_idx must satisfy 0 <= time_idx < T - 1 for cached proposal_states, "
                f"got {time_idx} with T={self._proposal_states.shape[1]}"
            )

        initial_velocity, initial_lateral_state = self._compute_initial_velocity_and_lateral_state(
            current_state=current_state,
            reference_trajectory=self._proposal_states[:, time_idx : time_idx + 1, :],
        )
        reference_velocities, curvature_profiles = self._compute_reference_velocity_and_curvature_profile(
            time_idx
        )

        batch_size = current_state.shape[0]
        accelerations = np.zeros(batch_size, dtype=np.float64)
        steering_rates = np.zeros(batch_size, dtype=np.float64)

        should_stop_mask = np.logical_and(
            reference_velocities <= self._config.stopping_velocity,
            initial_velocity <= self._config.stopping_velocity,
        )
        if np.any(should_stop_mask):
            stop_accels, stop_steering = self._stopping_controller(
                initial_velocity[should_stop_mask],
                reference_velocities[should_stop_mask],
            )
            accelerations[should_stop_mask] = stop_accels
            steering_rates[should_stop_mask] = stop_steering

        active_mask = ~should_stop_mask
        if np.any(active_mask):
            accelerations[active_mask] = self._longitudinal_lqr_controller(
                initial_velocity[active_mask],
                reference_velocities[active_mask],
            )
            velocity_profiles = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=initial_velocity[active_mask],
                derivatives=np.repeat(
                    accelerations[active_mask, None],
                    self._config.tracking_horizon,
                    axis=-1,
                ),
                discretization_time=self._discretization_time,
            )[:, : self._config.tracking_horizon]
            steering_rates[active_mask] = self._lateral_lqr_controller(
                initial_lateral_state[active_mask],
                velocity_profiles,
                curvature_profiles[active_mask],
            )

        return accelerations, steering_rates

    def _compute_initial_velocity_and_lateral_state(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        initial_reference = reference_trajectory[:, 0]
        x_errors = current_state[:, StateIndex.X] - initial_reference[:, 0]
        y_errors = current_state[:, StateIndex.Y] - initial_reference[:, 1]
        heading_references = initial_reference[:, 2]

        lateral_errors = -x_errors * np.sin(heading_references) + y_errors * np.cos(heading_references)
        heading_errors = _normalize_angle(current_state[:, StateIndex.HEADING] - heading_references)
        initial_velocities = current_state[:, StateIndex.VELOCITY_X]
        initial_lateral_state = np.stack(
            [
                lateral_errors,
                heading_errors,
                current_state[:, StateIndex.STEERING_ANGLE],
            ],
            axis=-1,
        )
        return initial_velocities, initial_lateral_state

    def _compute_reference_velocity_and_curvature_profile(
        self,
        time_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._velocity_profile is None or self._curvature_profile is None:
            raise RuntimeError("update() must be called before computing reference profiles")

        batch_size, num_profile_steps = self._velocity_profile.shape
        reference_idx = min(time_idx + self._config.tracking_horizon, num_profile_steps - 1)
        reference_velocities = self._velocity_profile[:, reference_idx]

        reference_curvature_profiles = np.zeros(
            (batch_size, self._config.tracking_horizon),
            dtype=np.float64,
        )
        reference_length = reference_idx - time_idx
        if reference_length > 0:
            reference_curvature_profiles[:, :reference_length] = self._curvature_profile[
                :, time_idx : time_idx + reference_length
            ]
        reference_curvature_profiles[:, reference_length:] = self._curvature_profile[
            :, reference_idx, None
        ]

        return reference_velocities, reference_curvature_profiles

    def _stopping_controller(
        self,
        initial_velocities: np.ndarray,
        reference_velocities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        accel = -self._config.stopping_proportional_gain * (initial_velocities - reference_velocities)
        return accel.astype(np.float64), np.zeros_like(accel, dtype=np.float64)

    def _longitudinal_lqr_controller(
        self,
        initial_velocities: np.ndarray,
        reference_velocities: np.ndarray,
    ) -> np.ndarray:
        batch_size = initial_velocities.shape[0]
        if batch_size == 0:
            return np.empty((0,), dtype=np.float64)

        a = np.ones(batch_size, dtype=np.float64)
        b = np.full(
            batch_size,
            self._config.tracking_horizon * self._discretization_time,
            dtype=np.float64,
        )
        g = np.zeros(batch_size, dtype=np.float64)
        return self._solve_one_step_longitudinal_lqr(
            initial_state=initial_velocities,
            reference_state=reference_velocities,
            a=a,
            b=b,
            g=g,
        )

    def _lateral_lqr_controller(
        self,
        initial_lateral_state_vector: np.ndarray,
        velocity_profile: np.ndarray,
        curvature_profile: np.ndarray,
    ) -> np.ndarray:
        if velocity_profile.shape[-1] != self._config.tracking_horizon:
            raise ValueError(
                f"velocity_profile must have length {self._config.tracking_horizon}, "
                f"got {velocity_profile.shape[-1]}"
            )
        if curvature_profile.shape[-1] != self._config.tracking_horizon:
            raise ValueError(
                f"curvature_profile must have length {self._config.tracking_horizon}, "
                f"got {curvature_profile.shape[-1]}"
            )

        batch_dim = velocity_profile.shape[0]
        if batch_dim == 0:
            return np.empty((0,), dtype=np.float64)

        n_lateral_states = len(LateralStateIndex)
        identity = np.eye(n_lateral_states, dtype=np.float64)
        input_matrix = np.zeros((n_lateral_states, 1), dtype=np.float64)
        input_matrix[LateralStateIndex.STEERING_ANGLE] = self._discretization_time

        states_matrix_at_step = np.tile(
            identity[None, None, ...],
            (self._config.tracking_horizon, batch_dim, 1, 1),
        )
        states_matrix_at_step[:, :, LateralStateIndex.LATERAL_ERROR, LateralStateIndex.HEADING_ERROR] = (
            velocity_profile.T * self._discretization_time
        )
        states_matrix_at_step[:, :, LateralStateIndex.HEADING_ERROR, LateralStateIndex.STEERING_ANGLE] = (
            velocity_profile.T * self._discretization_time / self._wheel_base
        )

        affine_terms = np.zeros(
            (self._config.tracking_horizon, batch_dim, n_lateral_states),
            dtype=np.float64,
        )
        affine_terms[:, :, LateralStateIndex.HEADING_ERROR] = (
            -velocity_profile.T * curvature_profile.T * self._discretization_time
        )

        a = np.tile(identity[None, ...], (batch_dim, 1, 1))
        b = np.zeros((batch_dim, n_lateral_states, 1), dtype=np.float64)
        g = np.zeros((batch_dim, n_lateral_states), dtype=np.float64)

        for state_matrix_at_step, affine_term in zip(states_matrix_at_step, affine_terms):
            a = np.einsum("bij,bjk->bik", state_matrix_at_step, a)
            b = np.einsum("bij,bjk->bik", state_matrix_at_step, b) + input_matrix
            g = np.einsum("bij,bj->bi", state_matrix_at_step, g) + affine_term

        steering_rate_cmd = self._solve_one_step_lateral_lqr(
            initial_state=initial_lateral_state_vector,
            a=a,
            b=b,
            g=g,
        )
        return np.squeeze(steering_rate_cmd, axis=-1)

    def _solve_one_step_longitudinal_lqr(
        self,
        initial_state: np.ndarray,
        reference_state: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        g: np.ndarray,
    ) -> np.ndarray:
        state_error_zero_input = a * initial_state + g - reference_state
        inverse = -1.0 / (b * self._q_longitudinal * b + self._r_longitudinal)
        return inverse * b * self._q_longitudinal * state_error_zero_input

    def _solve_one_step_lateral_lqr(
        self,
        initial_state: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        g: np.ndarray,
    ) -> np.ndarray:
        bt = np.transpose(b, (0, 2, 1))
        state_error_zero_input = np.einsum("bij,bj->bi", a, initial_state) + g

        angle_diff_indices = [
            LateralStateIndex.HEADING_ERROR.value,
            LateralStateIndex.STEERING_ANGLE.value,
        ]
        angles = state_error_zero_input[..., angle_diff_indices]
        state_error_zero_input[..., angle_diff_indices] = np.arctan2(np.sin(angles), np.cos(angles))

        bt_x_q = np.einsum("bij,jk->bik", bt, self._q_lateral)
        inv = -1.0 / (np.einsum("bij,bji->bi", bt_x_q, b) + self._r_lateral)
        tail = np.einsum("bij,bj->bi", bt_x_q, state_error_zero_input)
        return inv * tail
