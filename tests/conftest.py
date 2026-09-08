"""Fixtures for grava-sim-engine tests.

Builds a minimal but real SceneContext (no mocking of internal data structures).
The scene is a simple straight road with one dynamic obstacle ahead.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from grava_sim_engine.core.enums import SemanticMapLayer, StateIndex
from grava_sim_engine.core.geometry import PDMPath
from grava_sim_engine.core.observation import PDMObservation
from grava_sim_engine.core.occupancy import DrivableMap
from grava_sim_engine.core.types import SceneContext


def _build_straight_road_scene(
    *,
    ego_x: float = 0.0,
    ego_y: float = 0.0,
    ego_heading: float = 0.0,
    ego_speed: float = 5.0,
    obstacle_x: float = 30.0,
    obstacle_y: float = 0.0,
    obstacle_vx: float = 0.0,
    road_half_width: float = 4.0,
    road_length: float = 100.0,
    num_steps: int = 50,
    has_red_light: bool = False,
    red_light_x: float = 20.0,
) -> SceneContext:
    """Build a minimal straight-road scene with one obstacle.

    Road: [-10, road_length] x [-road_half_width, road_half_width]
    Ego: at (ego_x, ego_y) heading east at ego_speed m/s
    Obstacle: at (obstacle_x, obstacle_y), box 4m x 2m
    """
    # Ego state: [x, y, heading, vx, vy, ax, ay, steer_angle, steer_rate, yaw_rate, yaw_accel]
    ego_state = np.zeros(StateIndex.size(), dtype=np.float64)
    ego_state[StateIndex.X] = ego_x
    ego_state[StateIndex.Y] = ego_y
    ego_state[StateIndex.HEADING] = ego_heading
    ego_state[StateIndex.VELOCITY_X] = ego_speed * np.cos(ego_heading)
    ego_state[StateIndex.VELOCITY_Y] = ego_speed * np.sin(ego_heading)

    # Past states: 20 steps of constant velocity driving backward in time
    num_past = 20
    ego_past_states = np.zeros((num_past, StateIndex.size()), dtype=np.float64)
    for i in range(num_past):
        t = -(num_past - i) * 0.1
        ego_past_states[i, StateIndex.X] = ego_x + ego_speed * np.cos(ego_heading) * t
        ego_past_states[i, StateIndex.Y] = ego_y + ego_speed * np.sin(ego_heading) * t
        ego_past_states[i, StateIndex.HEADING] = ego_heading
        ego_past_states[i, StateIndex.VELOCITY_X] = ego_speed * np.cos(ego_heading)
        ego_past_states[i, StateIndex.VELOCITY_Y] = ego_speed * np.sin(ego_heading)

    # Drivable area: one big lane polygon
    road_polygon = box(-10.0, -road_half_width, road_length, road_half_width)
    drivable_map = DrivableMap(
        tokens=["lane_0"],
        types=[SemanticMapLayer.LANE],
        polygons=np.array([road_polygon], dtype=object),
    )

    # Centerline: straight line along x-axis
    centerline = PDMPath(np.array([[-10.0, 0.0], [road_length, 0.0]], dtype=np.float64))

    # Observation: one dynamic obstacle box
    observation = PDMObservation(num_steps=num_steps, interval_time=0.1)

    # Obstacle: 4m x 2m box centered at (obstacle_x, obstacle_y)
    obstacle_corners = np.array([
        [obstacle_x - 2.0, obstacle_y + 1.0],  # front-left
        [obstacle_x + 2.0, obstacle_y + 1.0],  # rear-left
        [obstacle_x + 2.0, obstacle_y - 1.0],  # rear-right
        [obstacle_x - 2.0, obstacle_y - 1.0],  # front-right
    ], dtype=np.float64)[None, :, :]

    red_light_tokens = []
    red_light_coords = np.empty((0, 4, 2), dtype=np.float64)
    if has_red_light:
        # Red light zone: 2m x road_width at red_light_x
        rl_corners = np.array([
            [red_light_x, road_half_width],
            [red_light_x + 2.0, road_half_width],
            [red_light_x + 2.0, -road_half_width],
            [red_light_x, -road_half_width],
        ], dtype=np.float64)[None, :, :]
        red_light_tokens = ["red_light_0"]
        red_light_coords = rl_corners

    observation.update(
        static_coords=np.empty((0, 4, 2), dtype=np.float64),
        static_tokens=[],
        dynamic_coords=obstacle_corners,
        dynamic_tokens=["vehicle_1"],
        dynamic_velocities=np.array([[obstacle_vx, 0.0]], dtype=np.float64),
        red_light_coords=red_light_coords,
        red_light_tokens=red_light_tokens,
    )

    return SceneContext(
        scene_token="test_scene_001",
        log_name="test_log",
        ego_state=ego_state,
        ego_past_states=ego_past_states,
        observation=observation,
        drivable_area_map=drivable_map,
        route_lane_ids={"lane_0"},
        centerline=centerline,
    )


@pytest.fixture
def straight_road_scene() -> SceneContext:
    """Basic straight road, obstacle 30m ahead, no red light."""
    return _build_straight_road_scene()


@pytest.fixture
def red_light_scene() -> SceneContext:
    """Straight road with a red light at x=20m."""
    return _build_straight_road_scene(has_red_light=True, red_light_x=20.0)


@pytest.fixture
def safe_trajectory() -> list[list[float]]:
    """8 waypoints going straight ahead ~20m in 4s — stays on road, avoids obstacle."""
    return [[2.5 * t, 0.0] for t in range(1, 9)]


@pytest.fixture
def collision_trajectory() -> list[list[float]]:
    """8 waypoints that drive straight into the obstacle at x=30."""
    return [[3.75 * t, 0.0] for t in range(1, 9)]


@pytest.fixture
def offroad_trajectory() -> list[list[float]]:
    """8 waypoints that veer off road (y >> road_half_width=4)."""
    return [[2.5 * t, 0.8 * t] for t in range(1, 9)]
