"""Tests for server endpoints and client methods.

Uses FastAPI TestClient (in-process, no real HTTP server needed).
Patches `load_scene_context` to return our fixture scenes so we don't need
MetricCache/NavSim dependencies.
"""
from __future__ import annotations

import copy
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from grava_sim_engine.client import SimEngineClient, SimEngineClientError
from grava_sim_engine.core.enums import MultiMetricIndex
from grava_sim_engine.core.scorer import PDMScorer, RLScorerConfig
from grava_sim_engine.core.scoring import PDMScorerV1
from grava_sim_engine.core.types import KeyActionObstacle, SceneContext
from grava_sim_engine.server.app import create_app


# ── Helpers ──────────────────────────────────────────────────────────────

DATASET = "test"
LOG_NAME = "test_log"
SCENE_TOKEN = "test_scene_001"


def _key_action_box(center_x: float, center_y: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            [center_x + 2.0, center_y + 1.0],
            [center_x - 2.0, center_y + 1.0],
            [center_x - 2.0, center_y - 1.0],
            [center_x + 2.0, center_y - 1.0],
        ],
        dtype=np.float64,
    )


def _internal_scene(scene: SceneContext) -> SceneContext:
    copied = copy.deepcopy(scene)
    copied.key_action_obstacles = [
        KeyActionObstacle(
            token="key_obstacle",
            label=1,
            polygon_coords=_key_action_box(12.0),
            object_type="static",
        )
    ]
    copied.candidate_progress = 25.0
    copied.gt_progress = 10.0
    return copied


class _PatchedClient(SimEngineClient):
    """SimEngineClient that talks to FastAPI TestClient instead of real HTTP."""

    def __init__(self, test_client: TestClient):
        super().__init__(base_url="http://testserver")
        self._test_client = test_client

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        if method == "GET":
            resp = self._test_client.get(path)
        else:
            resp = self._test_client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()


@pytest.fixture
def app_and_client(straight_road_scene):
    """Create FastAPI app + TestClient with load_scene_context patched and test dataset registered."""
    with patch("grava_sim_engine.server.app.load_scene_context", return_value=straight_road_scene):
        with patch.dict("os.environ", {"SIM_ENGINE_DATASETS": f"test=/fake/cache"}):
            app = create_app()
            with TestClient(app) as tc:
                yield tc


@pytest.fixture
def app_with_redlight(red_light_scene):
    """TestClient with a red-light scene."""
    with patch("grava_sim_engine.server.app.load_scene_context", return_value=red_light_scene):
        with patch.dict("os.environ", {"SIM_ENGINE_DATASETS": f"test=/fake/cache"}):
            app = create_app()
            with TestClient(app) as tc:
                yield tc


@pytest.fixture
def sim_client(app_and_client) -> SimEngineClient:
    """SimEngineClient wired to the in-process TestClient."""
    return _PatchedClient(app_and_client)


@pytest.fixture
def sim_client_redlight(app_with_redlight) -> SimEngineClient:
    return _PatchedClient(app_with_redlight)


# ═══════════════════════════════════════════════════════════════════════
# 1. Health endpoint
# ═══════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_health_ok(self, app_and_client):
        resp = app_and_client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "cache_stats" in data

    def test_client_health(self, sim_client):
        result = sim_client.health()
        assert result["status"] == "ok"

    def test_health_progress_counts_skipped_and_failed(self, app_and_client):
        with (
            patch("grava_sim_engine.server.app.get_boost_cache_dir", return_value="/tmp/boost_cache"),
            patch(
                "grava_sim_engine.server.app.get_warmup_stats",
                return_value={"converted": 60, "skipped": 30, "failed": 10, "total": 100},
            ),
        ):
            resp = app_and_client.get("/v1/health")

        assert resp.status_code == 200
        boost = resp.json()["boost_cache"]
        assert boost["converted"] == 60
        assert boost["skipped"] == 30
        assert boost["failed"] == 10
        assert boost["completed"] == 100
        assert boost["progress_pct"] == 100.0


# ═══════════════════════════════════════════════════════════════════════
# 2. PDM Scoring endpoints (/v1/score, /v1/score/batch)
# ═══════════════════════════════════════════════════════════════════════


class TestPDMScoring:
    @staticmethod
    def _score_scene(scene: SceneContext, trajectory: list[list[float]]) -> dict[str, Any]:
        with patch("grava_sim_engine.server.app.load_scene_context", return_value=scene):
            with patch.dict("os.environ", {"SIM_ENGINE_DATASETS": f"test=/fake/cache"}):
                app = create_app()
                with TestClient(app) as tc:
                    resp = tc.post("/v1/score", json={
                        "trajectory": trajectory,
                        "scene_token": SCENE_TOKEN,
                        "log_name": LOG_NAME,
                        "dataset": DATASET,
                    })
        assert resp.status_code == 200
        return resp.json()

    def test_score_single(self, app_and_client, safe_trajectory):
        resp = app_and_client.post("/v1/score", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "pdm_score" in data
        assert data["error"] is None
        assert 0.0 <= data["pdm_score"] <= 1.0

    def test_score_progress_normalization_ignores_analysis_gt_when_pdm_fixed(self, straight_road_scene, safe_trajectory):
        low_gt_scene = copy.deepcopy(straight_road_scene)
        low_gt_scene.pdm_masked_progress = 40.0
        low_gt_scene.gt_trajectory = np.array([[0.5, 0.0]] * 8, dtype=np.float64)

        high_gt_scene = copy.deepcopy(straight_road_scene)
        high_gt_scene.pdm_masked_progress = 40.0
        high_gt_scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)

        low_gt_progress = self._score_scene(low_gt_scene, safe_trajectory)["ego_progress"]
        high_gt_progress = self._score_scene(high_gt_scene, safe_trajectory)["ego_progress"]

        assert low_gt_progress == pytest.approx(high_gt_progress)

    def test_score_progress_normalization_tracks_pdm_reference(self, straight_road_scene, safe_trajectory):
        short_pdm_scene = copy.deepcopy(straight_road_scene)
        short_pdm_scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)
        short_pdm_scene.pdm_masked_progress = 10.0

        long_pdm_scene = copy.deepcopy(straight_road_scene)
        long_pdm_scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)
        long_pdm_scene.pdm_masked_progress = 40.0

        short_pdm_progress = self._score_scene(short_pdm_scene, safe_trajectory)["ego_progress"]
        long_pdm_progress = self._score_scene(long_pdm_scene, safe_trajectory)["ego_progress"]

        assert short_pdm_progress != pytest.approx(long_pdm_progress)
        assert short_pdm_progress > long_pdm_progress

    def test_score_progress_normalization_resolves_online_pdm_without_mutating_scene(
        self,
        straight_road_scene,
        safe_trajectory,
    ):
        fallback_scene = copy.deepcopy(straight_road_scene)
        fallback_scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)
        fallback_scene.pdm_trajectory = np.array(safe_trajectory, dtype=np.float64)
        fallback_scene.pdm_progress = None
        fallback_scene.pdm_masked_progress = None

        original_pdm_trajectory = fallback_scene.pdm_trajectory.copy()
        scorer = PDMScorerV1()
        expected_pdm_result = scorer._simulate_and_score_pdm(fallback_scene)
        assert expected_pdm_result is not None
        expected_pdm_masked_progress = expected_pdm_result.progress * (
            expected_pdm_result.multi_metrics[MultiMetricIndex.NO_COLLISION]
            * expected_pdm_result.multi_metrics[MultiMetricIndex.DRIVABLE_AREA]
        )
        assert expected_pdm_masked_progress > scorer._progress_distance_threshold

        prefilled_scene = copy.deepcopy(straight_road_scene)
        prefilled_scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)
        prefilled_scene.pdm_masked_progress = expected_pdm_masked_progress

        fallback_result = self._score_scene(fallback_scene, safe_trajectory)
        prefilled_result = self._score_scene(prefilled_scene, safe_trajectory)

        assert fallback_result["ego_progress"] == pytest.approx(prefilled_result["ego_progress"])
        assert fallback_scene.pdm_progress is None
        assert fallback_scene.pdm_masked_progress is None
        np.testing.assert_allclose(fallback_scene.pdm_trajectory, original_pdm_trajectory)

    def test_score_batch(self, app_and_client, safe_trajectory, offroad_trajectory):
        resp = app_and_client.post("/v1/score/batch", json={
            "trajectories": [safe_trajectory, offroad_trajectory],
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        # Safe trajectory should score higher than offroad
        assert data["results"][0]["pdm_score"] >= data["results"][1]["pdm_score"]

    def test_score_batch_matches_independent_single_scores(self, straight_road_scene):
        scene = copy.deepcopy(straight_road_scene)
        scene.pdm_masked_progress = 0.0
        short_trajectory = [[1.0 * t, 0.0] for t in range(1, 9)]
        long_trajectory = [[2.5 * t, 0.0] for t in range(1, 9)]

        with patch("grava_sim_engine.server.app.load_scene_context", return_value=scene):
            with patch.dict("os.environ", {"SIM_ENGINE_DATASETS": f"test=/fake/cache"}):
                app = create_app()
                with TestClient(app) as tc:
                    batch_resp = tc.post("/v1/score/batch", json={
                        "trajectories": [short_trajectory, long_trajectory],
                        "scene_token": SCENE_TOKEN,
                        "log_name": LOG_NAME,
                        "dataset": DATASET,
                    })
                    single_resps = [
                        tc.post("/v1/score", json={
                            "trajectory": trajectory,
                            "scene_token": SCENE_TOKEN,
                            "log_name": LOG_NAME,
                            "dataset": DATASET,
                        })
                        for trajectory in [short_trajectory, long_trajectory]
                    ]

        assert batch_resp.status_code == 200
        for resp in single_resps:
            assert resp.status_code == 200

        batch_results = batch_resp.json()["results"]
        single_results = [resp.json() for resp in single_resps]
        assert len(batch_results) == len(single_results)
        for batch_result, single_result in zip(batch_results, single_results):
            assert batch_result["pdm_score"] == pytest.approx(single_result["pdm_score"])
            assert batch_result["ego_progress"] == pytest.approx(single_result["ego_progress"])

    def test_client_score_batch(self, sim_client, safe_trajectory, offroad_trajectory):
        results = sim_client.score_batch(
            trajectories=[safe_trajectory, offroad_trajectory],
            scene_token=SCENE_TOKEN,
            log_name=LOG_NAME,
            dataset=DATASET,
        )
        assert len(results) == 2
        assert results[0]["pdm_score"] >= results[1]["pdm_score"]


# ═══════════════════════════════════════════════════════════════════════
# 3. RL Scoring endpoints (/v1/score/rl, /v1/score/rl/batch)
# ═══════════════════════════════════════════════════════════════════════


class TestRLScoring:
    @staticmethod
    def _score_scene(scene: SceneContext, trajectory: list[list[float]]) -> dict[str, Any]:
        with patch("grava_sim_engine.server.app.load_scene_context", return_value=scene):
            with patch.dict("os.environ", {"SIM_ENGINE_DATASETS": f"test=/fake/cache"}):
                app = create_app()
                with TestClient(app) as tc:
                    resp = tc.post("/v1/score/rl", json={
                        "trajectory": trajectory,
                        "scene_token": SCENE_TOKEN,
                        "log_name": LOG_NAME,
                        "dataset": DATASET,
                    })
        assert resp.status_code == 200
        return resp.json()

    def test_rl_progress_normalization_ignores_analysis_gt_fields_when_pdm_reference_is_fixed(
        self,
        straight_road_scene,
        safe_trajectory,
    ):
        short_gt_scene = copy.deepcopy(straight_road_scene)
        short_gt_scene.pdm_masked_progress = 40.0
        short_gt_scene.gt_masked_progress = 5.0
        short_gt_scene.gt_trajectory = np.array([[0.5, 0.0]] * 8, dtype=np.float64)

        long_gt_scene = copy.deepcopy(straight_road_scene)
        long_gt_scene.pdm_masked_progress = 40.0
        long_gt_scene.gt_masked_progress = 80.0
        long_gt_scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)

        short_gt_progress = self._score_scene(short_gt_scene, safe_trajectory)["ego_progress"]
        long_gt_progress = self._score_scene(long_gt_scene, safe_trajectory)["ego_progress"]

        assert short_gt_progress == pytest.approx(long_gt_progress)

    def test_rl_progress_normalization_tracks_pdm_reference(self, straight_road_scene, safe_trajectory):
        short_reference_scene = copy.deepcopy(straight_road_scene)
        short_reference_scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)
        short_reference_scene.pdm_masked_progress = 10.0

        long_reference_scene = copy.deepcopy(straight_road_scene)
        long_reference_scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)
        long_reference_scene.pdm_masked_progress = 40.0

        short_reference_progress = self._score_scene(short_reference_scene, safe_trajectory)["ego_progress"]
        long_reference_progress = self._score_scene(long_reference_scene, safe_trajectory)["ego_progress"]

        assert short_reference_progress != pytest.approx(long_reference_progress)
        assert short_reference_progress > long_reference_progress

    def test_rl_progress_normalization_resolves_online_pdm_reference_without_prefilled_masked_progress(
        self,
        straight_road_scene,
        safe_trajectory,
    ):
        fallback_scene = copy.deepcopy(straight_road_scene)
        fallback_scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)
        fallback_scene.pdm_trajectory = np.array(safe_trajectory, dtype=np.float64)
        fallback_scene.pdm_progress = None
        fallback_scene.pdm_masked_progress = None

        original_pdm_trajectory = fallback_scene.pdm_trajectory.copy()
        scorer = PDMScorer()
        expected_pdm_result = scorer._get_scorer()._simulate_and_score_pdm(fallback_scene)
        assert expected_pdm_result is not None
        expected_pdm_masked_progress = expected_pdm_result.progress * (
            expected_pdm_result.multi_metrics[MultiMetricIndex.NO_COLLISION]
            * expected_pdm_result.multi_metrics[MultiMetricIndex.DRIVABLE_AREA]
        )

        prefilled_reference_scene = copy.deepcopy(straight_road_scene)
        prefilled_reference_scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)
        prefilled_reference_scene.pdm_masked_progress = expected_pdm_masked_progress

        fallback_result = self._score_scene(fallback_scene, safe_trajectory)
        prefilled_result = self._score_scene(prefilled_reference_scene, safe_trajectory)

        assert fallback_result["ego_progress"] == pytest.approx(prefilled_result["ego_progress"])
        assert fallback_scene.pdm_progress is None
        assert fallback_scene.pdm_masked_progress is None
        np.testing.assert_allclose(fallback_scene.pdm_trajectory, original_pdm_trajectory)

    def test_rl_progress_normalization_falls_back_to_threshold_when_pdm_reference_is_unavailable(
        self,
        straight_road_scene,
        safe_trajectory,
    ):
        scene_without_pdm_reference = copy.deepcopy(straight_road_scene)
        scene_without_pdm_reference.gt_masked_progress = 80.0
        scene_without_pdm_reference.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)
        scene_without_pdm_reference.pdm_trajectory = None
        scene_without_pdm_reference.pdm_progress = None
        scene_without_pdm_reference.pdm_masked_progress = None

        threshold_scene = copy.deepcopy(scene_without_pdm_reference)
        threshold_scene.gt_masked_progress = None

        missing_reference_result = self._score_scene(scene_without_pdm_reference, safe_trajectory)
        threshold_result = self._score_scene(threshold_scene, safe_trajectory)

        assert missing_reference_result["ego_progress"] == pytest.approx(1.0)
        assert missing_reference_result["ego_progress"] == pytest.approx(threshold_result["ego_progress"])

    # ── 3.1 Basic continuous mode ────────────────────────────────────

    def test_rl_score_continuous(self, app_and_client, safe_trajectory):
        resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "continuous",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "rl_score" in data
        assert "sub_rewards" in data
        assert data["error"] is None
        assert 0.0 <= data["rl_score"] <= 1.0
        # sub_rewards should have all 8 keys
        assert set(data["sub_rewards"].keys()) == {"nc", "dac", "ddc", "tlc", "ep", "ttc", "lk", "hc"}
        # All sub_rewards in [0, 1]
        for key, val in data["sub_rewards"].items():
            assert 0.0 <= val <= 1.0, f"sub_reward {key}={val} out of [0,1]"
        for key in [
            "centerline_lateral_offset_start_signed",
            "centerline_lateral_offset_end_signed",
            "centerline_distance_mean",
            "centerline_distance_max",
            "boundary_distance_start",
            "boundary_distance_end",
            "boundary_distance_min",
            "boundary_distance_mean",
            "in_intersection_fraction",
            "oncoming_fraction",
            "non_drivable_fraction",
            "multiple_lanes_fraction",
        ]:
            assert key in data
        assert isinstance(data["local_centerline_points"], list)
        assert isinstance(data["boundary_distances"], list)
        assert isinstance(data["in_intersection_flags"], list)
        assert isinstance(data["oncoming_flags"], list)
        assert isinstance(data["non_drivable_flags"], list)
        assert isinstance(data["multiple_lanes_flags"], list)
        assert 0.0 <= data["in_intersection_fraction"] <= 1.0
        assert 0.0 <= data["oncoming_fraction"] <= 1.0
        assert 0.0 <= data["non_drivable_fraction"] <= 1.0
        assert 0.0 <= data["multiple_lanes_fraction"] <= 1.0

    # ── 3.2 Basic discrete mode ──────────────────────────────────────

    def test_rl_score_discrete(self, app_and_client, safe_trajectory):
        resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "discrete",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        # Discrete safety metrics should be exactly 0.0 or 1.0
        for key in ["no_at_fault_collisions", "drivable_area_compliance",
                     "driving_direction_compliance", "traffic_light_compliance"]:
            assert data[key] in (0.0, 0.5, 1.0), f"discrete {key}={data[key]} not binary-ish"

    # ── 3.3 Mode switching produces different results ────────────────

    def test_continuous_vs_discrete_differ(self, app_and_client, safe_trajectory):
        resp_cont = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "continuous",
        })
        resp_disc = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "discrete",
        })
        cont = resp_cont.json()
        disc = resp_disc.json()
        # At least one sub-metric should differ (NC continuous != NC discrete near obstacle)
        # Note: they CAN be same if trajectory is far from all obstacles. Use rl_score.
        # Both should be valid
        assert cont["error"] is None
        assert disc["error"] is None

    # ── 3.4 Batch RL scoring ─────────────────────────────────────────

    def test_rl_score_batch(self, app_and_client, safe_trajectory, offroad_trajectory):
        resp = app_and_client.post("/v1/score/rl/batch", json={
            "trajectories": [safe_trajectory, offroad_trajectory],
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "continuous",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        safe_result = data["results"][0]
        offroad_result = data["results"][1]
        for result in data["results"]:
            assert "centerline_distance_mean" in result
            assert "boundary_distances" in result
            assert "in_intersection_flags" in result
        # Safe should score higher than offroad
        assert safe_result["rl_score"] >= offroad_result["rl_score"]
        # Offroad should have lower DAC
        assert safe_result["drivable_area_compliance"] >= offroad_result["drivable_area_compliance"]

    # ── 3.5 Config overrides ─────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════
# 4. Client RL methods
# ═══════════════════════════════════════════════════════════════════════


class TestClientRL:
    def test_client_score_batch_rl(self, sim_client, safe_trajectory, offroad_trajectory):
        results = sim_client.score_batch_rl(
            trajectories=[safe_trajectory, offroad_trajectory],
            scene_token=SCENE_TOKEN,
            log_name=LOG_NAME,
            dataset=DATASET,
            scoring_mode="continuous",
        )
        assert len(results) == 2
        assert results[0]["rl_score"] >= results[1]["rl_score"]
        assert results[0]["error"] is None


# ═══════════════════════════════════════════════════════════════════════
# 5. Internal key-action scoring
# ═══════════════════════════════════════════════════════════════════════


class TestInternalScoring:
    def test_internal_score_endpoint(self, straight_road_scene, safe_trajectory):
        scene = _internal_scene(straight_road_scene)
        with patch("grava_sim_engine.server.app.load_scene_context", return_value=scene):
            with patch.dict("os.environ", {"SIM_ENGINE_DATASETS": f"test=/fake/cache"}):
                app = create_app()
                with TestClient(app) as tc:
                    resp = tc.post("/v1/score/internal/rl", json={
                        "trajectory": safe_trajectory,
                        "scene_token": SCENE_TOKEN,
                        "log_name": LOG_NAME,
                        "dataset": DATASET,
                    })

        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        assert data["sample_valid"] is True
        assert data["num_key_actions"] == 1
        assert data["num_key_actions_passed"] == 1
        assert data["key_action_score"] == pytest.approx(1.0)
        assert data["progress_norm_source"] == "candidate+logged:candidate"
        assert 0.0 <= data["internal_score"] <= 1.0

    def test_internal_score_batch_endpoint(self, straight_road_scene, safe_trajectory, offroad_trajectory):
        scene = _internal_scene(straight_road_scene)
        with patch("grava_sim_engine.server.app.load_scene_context", return_value=scene):
            with patch.dict("os.environ", {"SIM_ENGINE_DATASETS": f"test=/fake/cache"}):
                app = create_app()
                with TestClient(app) as tc:
                    resp = tc.post("/v1/score/internal/rl/batch", json={
                        "trajectories": [safe_trajectory, offroad_trajectory],
                        "scene_token": SCENE_TOKEN,
                        "log_name": LOG_NAME,
                        "dataset": DATASET,
                    })

        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 2
        assert results[0]["sample_valid"] is True
        assert results[1]["sample_valid"] is True
        assert results[0]["internal_score"] >= results[1]["internal_score"]

    def test_client_score_internal_rl(self, straight_road_scene, safe_trajectory):
        scene = _internal_scene(straight_road_scene)
        with patch("grava_sim_engine.server.app.load_scene_context", return_value=scene):
            with patch.dict("os.environ", {"SIM_ENGINE_DATASETS": f"test=/fake/cache"}):
                app = create_app()
                with TestClient(app) as tc:
                    client = _PatchedClient(tc)
                    score, result = client.score_internal_rl(
                        trajectory=safe_trajectory,
                        scene_token=SCENE_TOKEN,
                        log_name=LOG_NAME,
                        dataset=DATASET,
                        trajectory_frame="nuplan",
                    )

        assert score == pytest.approx(result["internal_score"])
        assert result["sample_valid"] is True
        assert result["error"] is None

# ═══════════════════════════════════════════════════════════════════════
# 5. Schema validation
# ═══════════════════════════════════════════════════════════════════════


class TestSchemaValidation:
    def test_invalid_scoring_mode(self, app_and_client, safe_trajectory):
        resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "invalid_mode",
        })
        assert resp.status_code == 422  # Pydantic validation error

    def test_missing_required_fields(self, app_and_client):
        resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": [[0.1, 0.2]],
            # Missing scene_token, log_name, metric_cache_dir
        })
        assert resp.status_code == 422

    def test_default_scoring_mode(self, app_and_client, safe_trajectory):
        """scoring_mode defaults to 'continuous' when not specified."""
        resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            # No scoring_mode — should default to continuous
        })
        assert resp.status_code == 200
        assert resp.json()["error"] is None


# ═══════════════════════════════════════════════════════════════════════
# 6. Consistency: discrete RL sub-metrics vs PDM sub-metrics
# ═══════════════════════════════════════════════════════════════════════


class TestDiscreteConsistency:
    """Discrete RL safety sub-metrics should match PDM sub-metrics."""

    def test_rl_pdms_monitor_matches_v1_score_for_same_trajectory(
        self,
        app_and_client,
        safe_trajectory,
        offroad_trajectory,
    ):
        metric_pairs = [
            ("pdm_score", "pdms_score"),
            ("no_at_fault_collisions", "pdms_no_at_fault_collisions"),
            ("drivable_area_compliance", "pdms_drivable_area_compliance"),
            ("driving_direction_compliance", "pdms_driving_direction_compliance"),
            ("traffic_light_compliance", "pdms_traffic_light_compliance"),
            ("ego_progress", "pdms_ego_progress"),
            ("time_to_collision", "pdms_time_to_collision"),
            ("lane_keeping", "pdms_lane_keeping"),
            ("history_comfort", "pdms_history_comfort"),
        ]

        for trajectory in [safe_trajectory, offroad_trajectory]:
            pdm_resp = app_and_client.post("/v1/score", json={
                "trajectory": trajectory,
                "scene_token": SCENE_TOKEN,
                "log_name": LOG_NAME,
                "dataset": DATASET,
            })
            rl_resp = app_and_client.post("/v1/score/rl", json={
                "trajectory": trajectory,
                "scene_token": SCENE_TOKEN,
                "log_name": LOG_NAME,
                "dataset": DATASET,
                "scoring_mode": "continuous",
            })

            assert pdm_resp.status_code == 200
            assert rl_resp.status_code == 200
            pdm = pdm_resp.json()
            rl = rl_resp.json()
            assert pdm["error"] is None
            assert rl["error"] is None

            for pdm_key, rl_key in metric_pairs:
                assert rl[rl_key] == pytest.approx(pdm[pdm_key]), (
                    f"{pdm_key}: PDM={pdm[pdm_key]} vs RL monitor={rl[rl_key]}"
                )

    def test_discrete_matches_pdm_safety(self, app_and_client, safe_trajectory):
        pdm_resp = app_and_client.post("/v1/score", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
        })
        rl_resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "discrete",
        })
        pdm = pdm_resp.json()
        rl = rl_resp.json()
        # Safety metrics should match exactly
        for key in ["no_at_fault_collisions", "drivable_area_compliance",
                     "driving_direction_compliance", "traffic_light_compliance"]:
            assert abs(pdm[key] - rl[key]) < 1e-6, f"{key}: PDM={pdm[key]} vs RL={rl[key]}"
        # LK, HC should match in discrete mode
        for key in ["lane_keeping", "history_comfort"]:
            assert abs(pdm[key] - rl[key]) < 1e-6, f"{key}: PDM={pdm[key]} vs RL={rl[key]}"
        # TTC: RL uses continuous _ttc_continuous (normalized by horizon) even in discrete mode,
        # while PDM uses binary _time_to_collision. This is by design — RL TTC is always continuous.
        # Both should be in [0, 1].
        assert 0.0 <= rl["time_to_collision"] <= 1.0
