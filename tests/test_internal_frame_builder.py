from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point

from grava_sim_engine.adapters.internal import (
    FUTURE_OBSTACLE_TRACKS_COORDINATE_FRAME,
    FUTURE_OBSTACLE_TRACKS_KEY,
    InternalCaseRecordSceneContextBuilder,
    build_future_trajectory_from_frame,
    build_scene_context_from_frame,
    load_frame_json,
)
from grava_sim_engine.core.geometry import calculate_progress, state_to_coords
from grava_sim_engine.core.scoring import PDMScorerV1


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = (
    REPO_ROOT
    / "data_assets/tmp/internal_schema_probe_15/internal_schema_probe_sample_frames_15"
)
SAMPLE_WITH_FUTURE = (
    SAMPLE_ROOT
    / "ARCF004_20240815080000_1723681804_1723681852"
    / "1723681836.015313/frame.json"
)
SAMPLE_WITHOUT_FUTURE = (
    SAMPLE_ROOT
    / "ARCF004_20240815080000_1723681804_1723681852"
    / "1723681851.815298/frame.json"
)
SAMPLE_WITH_CLEAN_PDM_SCORE = (
    SAMPLE_ROOT
    / "ARCF004_20240815065317_1723678230_1723678239"
    / "1723678230.115316/frame.json"
)
SAMPLE_WITH_SHORT_GT = (
    SAMPLE_ROOT
    / "ARCF004_20240815065317_1723678302_1723678312"
    / "1723678309.915294/frame.json"
)
SAMPLE_CASE_WITH_UNEVEN_TIMESTAMPS = (
    SAMPLE_ROOT
    / "ARCF004_20240815065317_1723678230_1723678239"
)


def _load_sample(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"real internal frame sample is not available: {path}")
    return load_frame_json(path)


def _load_case(case_path: Path) -> list[dict]:
    if not case_path.exists():
        pytest.skip(f"real internal case sample is not available: {case_path}")
    frame_paths = sorted(case_path.glob("*/frame.json"), key=lambda path: float(path.parent.name))
    return [load_frame_json(path) for path in frame_paths]


def _raw_info_boundary(points: list[tuple[float, float]]) -> dict:
    return {
        "curve": {
            "segment": [
                {
                    "lineSegment": {
                        "point": [
                            {"x": float(x), "y": float(y), "z": 0.0}
                            for x, y in points
                        ]
                    }
                }
            ]
        }
    }


def test_internal_frame_builder_uses_real_frame_schema_and_nous_coordinates():
    frame = _load_sample(SAMPLE_WITH_FUTURE)

    scene = build_scene_context_from_frame(frame)

    assert scene.log_name == frame["case_id"]
    assert scene.scene_token == str(frame["timestamp"])
    assert len(scene.drivable_area_map.tokens) > 0
    assert len(scene.route_lane_ids) > 0
    assert scene.centerline.discrete_path.shape[1] == 2
    assert scene.gt_trajectory is not None
    assert scene.gt_trajectory.shape == (8, 2)
    assert scene.pdm_trajectory is not None
    assert scene.pdm_trajectory.shape == (8, 2)

    first_future = frame["ego_car"]["future_trajectory"][1]
    expected_first_waypoint = np.asarray(
        [first_future[1], -first_future[0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(scene.gt_trajectory[0], expected_first_waypoint, atol=1e-6)

    obstacle = next(item for item in frame["obstacles"] if item.get("bbox_3d"))
    bbox = obstacle["bbox_3d"]
    obstacle_token = str(obstacle.get("id") or obstacle.get("track_id") or "obstacle_0")
    expected_center = Point(float(bbox[1]), -float(bbox[0]))
    occupancy_map = scene.observation.get_occupancy_map(0)
    assert occupancy_map is not None
    assert occupancy_map[obstacle_token].contains(expected_center)


def test_internal_frame_builder_falls_back_to_raw_info_lanes_when_frame_map_is_empty():
    frame = {
        "case_id": "case_with_raw_info_map",
        "timestamp": "1.0",
        "map": {},
        "ego_car": {
            "future_trajectory": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.0, "y": 10.0},
                {"x": 0.0, "y": 20.0},
            ],
        },
        "obstacles": [],
    }
    info_data = {
        "ego_car_attribute": {
            "position": [0.0, 0.0, 0.0],
            "ego_heading": 0.0,
        },
        "lanes_info": [
            {
                "id": {"id": "lane_raw"},
                "leftBoundary": _raw_info_boundary([(0.0, 2.0), (20.0, 2.0)]),
                "rightBoundary": _raw_info_boundary([(0.0, -2.0), (20.0, -2.0)]),
            }
        ],
    }

    scene = build_scene_context_from_frame(frame, info_data=info_data)

    assert scene.drivable_area_map.tokens == ["lane_raw"]
    assert scene.route_lane_ids == {"lane_raw"}
    assert scene.drivable_area_map["lane_raw"].contains(Point(5.0, 0.0))


def test_internal_frame_builder_falls_back_to_raw_info_centerline():
    frame = {
        "case_id": "case_with_raw_info_route",
        "timestamp": "1.0",
        "map": {},
        "ego_car": {},
        "obstacles": [],
    }
    info_data = {
        "ego_car_attribute": {
            "position": [0.0, 0.0, 0.0],
            "ego_heading": 0.0,
            "pnc_local_routing": {
                "routing_points": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 10.0, "y": 0.0, "z": 0.0},
                    {"x": 20.0, "y": 0.0, "z": 0.0},
                ]
            },
        },
        "lanes_info": [
            {
                "id": {"id": "lane_route"},
                "leftBoundary": _raw_info_boundary([(0.0, 2.0), (20.0, 2.0)]),
                "rightBoundary": _raw_info_boundary([(0.0, -2.0), (20.0, -2.0)]),
            }
        ],
    }

    scene = build_scene_context_from_frame(frame, info_data=info_data)

    assert scene.gt_trajectory is None
    assert scene.pdm_trajectory is not None
    assert scene.pdm_trajectory.shape == (8, 2)
    assert scene.route_lane_ids == {"lane_route"}
    np.testing.assert_allclose(scene.centerline.discrete_path[-1], [20.0, 0.0], atol=1e-6)


def test_internal_frame_builder_preserves_key_action_labels_from_raw_obstacles():
    frame = {
        "case_id": "case_with_key_action_labels",
        "timestamp": "1.0",
        "map": {},
        "ego_car": {
            "future_trajectory": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.0, "y": 5.0},
                {"x": 0.0, "y": 10.0},
            ],
        },
        "obstacles": [
            {
                "id": 123,
                "bbox_3d": [0.0, 10.0, 0.0, 4.0, 2.0, 1.5, 1.5 * np.pi],
                "label": 1,
                "sub_type": "CAR",
            },
            {
                "id": 456,
                "bbox_3d": [0.0, 20.0, 0.0, 4.0, 2.0, 1.5, 1.5 * np.pi],
                "decision_label": "no_nudge",
            },
        ],
    }
    info_data = {
        "ego_car_attribute": {
            "position": [0.0, 0.0, 0.0],
            "ego_heading": 0.0,
            "pnc_local_routing": {
                "routing_points": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 20.0, "y": 0.0, "z": 0.0},
                ]
            },
            "candidate_trajectory_point": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.0, "y": 15.0},
            ],
        },
        "lanes_info": [
            {
                "id": {"id": "lane_key_action"},
                "leftBoundary": _raw_info_boundary([(0.0, 2.0), (20.0, 2.0)]),
                "rightBoundary": _raw_info_boundary([(0.0, -2.0), (20.0, -2.0)]),
            }
        ],
    }

    scene = build_scene_context_from_frame(frame, info_data=info_data)

    assert len(scene.key_action_obstacles) == 1
    annotation = scene.key_action_obstacles[0]
    assert annotation.token == "123"
    assert annotation.label == 1
    assert annotation.object_type == "agent"
    np.testing.assert_allclose(annotation.polygon_coords.mean(axis=0), [10.0, 0.0], atol=1e-6)
    assert scene.candidate_trajectory is not None
    assert scene.candidate_progress == pytest.approx(15.0)
    assert scene.gt_progress == pytest.approx(10.0)


def test_internal_frame_builder_keeps_stationary_gt_repeated_points():
    frame = {
        "case_id": "case_with_stationary_future",
        "timestamp": "1.0",
        "map": {},
        "ego_car": {
            "future_trajectory": [[0.0, 0.0, 0.0] for _ in range(16)],
        },
        "obstacles": [],
    }
    info_data = {
        "ego_car_attribute": {
            "position": [0.0, 0.0, 0.0],
            "ego_heading": 0.0,
            "pnc_local_routing": {
                "routing_points": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 10.0, "y": 0.0, "z": 0.0},
                    {"x": 20.0, "y": 0.0, "z": 0.0},
                ]
            },
        },
        "lanes_info": [
            {
                "id": {"id": "lane_stationary"},
                "leftBoundary": _raw_info_boundary([(0.0, 2.0), (20.0, 2.0)]),
                "rightBoundary": _raw_info_boundary([(0.0, -2.0), (20.0, -2.0)]),
            }
        ],
    }

    scene = build_scene_context_from_frame(frame, info_data=info_data)
    gt_only = build_future_trajectory_from_frame(frame)

    assert scene.gt_trajectory is not None
    assert scene.gt_trajectory.shape == (8, 2)
    np.testing.assert_allclose(scene.gt_trajectory, np.zeros((8, 2)), atol=1e-6)
    np.testing.assert_allclose(scene.pdm_trajectory, scene.gt_trajectory, atol=1e-6)
    np.testing.assert_allclose(gt_only, scene.gt_trajectory, atol=1e-6)


def test_internal_frame_builder_falls_back_to_centerline_reference_without_future():
    frame = _load_sample(SAMPLE_WITHOUT_FUTURE)

    scene = build_scene_context_from_frame(frame)

    assert scene.gt_trajectory is None
    assert scene.pdm_trajectory is not None
    assert scene.pdm_trajectory.shape == (8, 2)
    assert np.all(scene.pdm_trajectory[:, 0] > 0.0)


def test_internal_frame_scene_scores_and_simulates_with_pdm_reference():
    frame = _load_sample(SAMPLE_WITH_CLEAN_PDM_SCORE)
    scene = build_scene_context_from_frame(frame)
    assert scene.pdm_trajectory is not None

    scorer = PDMScorerV1()
    result = scorer.score(scene.pdm_trajectory, scene)

    assert result.pdm_score == pytest.approx(1.0)
    assert result.no_at_fault_collisions == pytest.approx(1.0)
    assert result.drivable_area_compliance == pytest.approx(1.0)
    assert result.ego_progress == pytest.approx(1.0)

    wrong_coordinate_trajectory = scene.pdm_trajectory[:, [1, 0]]
    wrong_result = scorer.score(wrong_coordinate_trajectory, scene)
    assert wrong_result.pdm_score < result.pdm_score
    assert wrong_result.drivable_area_compliance == pytest.approx(0.0)

    proposals = scorer._build_proposals(scene.pdm_trajectory[None, ...], scene)
    simulated = scorer._simulator.simulate_proposals(
        ego_state=scene.ego_state,
        proposals=proposals,
        observation=scene.observation,
    )
    progress = calculate_progress(state_to_coords(simulated, scorer._vehicle), scene.centerline)[0]

    assert simulated.shape == (1, 41, 11)
    assert np.all(np.diff(progress) >= -1e-4)
    assert progress[-1] - progress[0] > 20.0


def test_internal_frame_short_gt_padding_does_not_turn_ego_off_road():
    frame = _load_sample(SAMPLE_WITH_SHORT_GT)
    scene = build_scene_context_from_frame(frame)
    assert scene.gt_trajectory is not None
    assert scene.gt_trajectory.shape == (8, 2)
    np.testing.assert_allclose(
        scene.gt_trajectory[2:],
        np.repeat(scene.gt_trajectory[1:2], 6, axis=0),
        atol=1e-6,
    )

    scorer = PDMScorerV1()
    result = scorer.score(scene.gt_trajectory, scene)

    assert result.pdm_score == pytest.approx(1.0)
    assert result.drivable_area_compliance == pytest.approx(1.0)
    assert result.time_to_collision == pytest.approx(1.0)


def test_internal_dynamic_obstacle_motion_uses_full_simulation_horizon():
    frame = _load_sample(SAMPLE_WITH_FUTURE)
    scene = build_scene_context_from_frame(frame)
    scorer = PDMScorerV1()

    token = "363554"
    obstacle = next(item for item in frame["obstacles"] if str(item.get("id")) == token)
    bbox = obstacle["bbox_3d"]
    velocity = obstacle["velocity"]
    expected_centroid_at_4s = np.asarray(
        [bbox[1], -bbox[0]],
        dtype=np.float64,
    ) + 4.0 * np.asarray(
        [velocity["y"], -velocity["x"]],
        dtype=np.float64,
    )

    occupancy_map = scorer._get_occupancy_map(scene, 40)
    assert occupancy_map is not None
    observed_centroid = np.asarray(occupancy_map[token].centroid.coords[0], dtype=np.float64)
    np.testing.assert_allclose(observed_centroid, expected_centroid_at_4s, atol=1e-6)

    assert scene.gt_trajectory is not None
    result = scorer.score(scene.gt_trajectory, scene)
    assert result.time_to_collision == pytest.approx(1.0)


def test_internal_case_record_observation_interpolates_uneven_timestamps():
    frames = _load_case(SAMPLE_CASE_WITH_UNEVEN_TIMESTAMPS)
    builder = InternalCaseRecordSceneContextBuilder()
    enriched_frame = builder.enrich_frame(frames, target_index=0)

    future_tracks = enriched_frame[FUTURE_OBSTACLE_TRACKS_KEY]
    assert future_tracks["coordinate_frame"] == FUTURE_OBSTACLE_TRACKS_COORDINATE_FRAME
    assert future_tracks["schema_version"] == 1
    assert any(track["id"] == "88630" for track in future_tracks["tracks"])

    scene = build_scene_context_from_frame(enriched_frame)
    case_scene = builder.build_target(frames, target_index=0)

    assert scene.observation.global_to_local_idcs == list(range(41))

    token = "88630"
    occupancy_0 = scene.observation.get_occupancy_map(0)
    occupancy_09 = scene.observation.get_occupancy_map(9)
    occupancy_18 = scene.observation.get_occupancy_map(18)
    assert occupancy_0 is not None
    assert occupancy_09 is not None
    assert occupancy_18 is not None

    centroid_0 = np.asarray(occupancy_0[token].centroid.coords[0], dtype=np.float64)
    centroid_09 = np.asarray(occupancy_09[token].centroid.coords[0], dtype=np.float64)
    centroid_18 = np.asarray(occupancy_18[token].centroid.coords[0], dtype=np.float64)
    np.testing.assert_allclose(
        centroid_09,
        0.5 * (centroid_0 + centroid_18),
        atol=1e-4,
    )

    case_centroid_09 = np.asarray(
        case_scene.observation.get_occupancy_map(9)[token].centroid.coords[0],
        dtype=np.float64,
    )
    np.testing.assert_allclose(centroid_09, case_centroid_09, atol=1e-6)
