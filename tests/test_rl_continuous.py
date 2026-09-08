"""Tests for continuous RL scoring: NC (collision), DAC (sweep area), EP normalization.

Uses synthetic scenes from conftest — no MetricCache dependency.
All trajectories are 8 waypoints at 0.5s spacing (ego-relative xy).
"""
from __future__ import annotations

import lzma
import pickle
import os
from pathlib import Path

import numpy as np
import pytest

from grava_sim_engine.adapters.navsim.cache_loader import (
    _lock_path,
    _load_pickle,
    _write_progress,
    get_boost_cache_dir,
    get_warmup_stats,
    load_scene_context,
    set_boost_cache_dir,
    warmup_boost_cache,
)
from grava_sim_engine.core.enums import SemanticMapLayer, StateIndex
from grava_sim_engine.core.geometry import PDMPath
from grava_sim_engine.core.observation import PDMObservation
from grava_sim_engine.core.occupancy import DrivableMap
from grava_sim_engine.core.scorer import PDMScorer, RLScorerConfig
from grava_sim_engine.core.scoring.base import ScorerBase
from grava_sim_engine.core.types import SceneContext


METRIC_CACHE_SAMPLE_PATH = Path(
    os.environ.get("GRAVA_REFERENCE_CACHE_SAMPLE", "/data/navsim/metric_cache_sample.pkl")
)


class TestSceneContextReferenceSemantics:
    def test_simulate_and_score_pdm_returns_none_without_pdm_reference(self):
        scene = _build_scene(no_obstacle=True)

        assert ScorerBase()._simulate_and_score_pdm(scene) is None

    def test_simulate_and_score_pdm_uses_pdm_reference_not_analysis_gt(self):
        scene = _build_scene(no_obstacle=True)
        scene.gt_trajectory = np.array([[20.0, 0.0]] * 8, dtype=np.float64)
        scene.pdm_trajectory = np.array([[0.5, 0.0]] * 8, dtype=np.float64)

        scorer = ScorerBase()
        pdm_result_a = scorer._simulate_and_score_pdm(scene)
        gt_result_a = scorer._simulate_and_score_gt(scene)

        scene.pdm_trajectory = np.array([[4.0, 0.0]] * 8, dtype=np.float64)
        pdm_result_b = scorer._simulate_and_score_pdm(scene)
        gt_result_b = scorer._simulate_and_score_gt(scene)

        assert pdm_result_a is not None
        assert pdm_result_b is not None
        assert gt_result_a is not None
        assert gt_result_b is not None
        assert pdm_result_a.progress != pdm_result_b.progress
        assert gt_result_a.progress == gt_result_b.progress

    def test_scene_context_keeps_gt_and_pdm_reference_fields_separate(self):
        ego_state = np.zeros(StateIndex.size(), dtype=np.float64)
        ego_past_states = np.zeros((1, StateIndex.size()), dtype=np.float64)
        observation = PDMObservation(num_steps=1, interval_time=0.1)
        drivable_map = DrivableMap(tokens=[], types=[], polygons=np.array([], dtype=object))
        centerline = PDMPath(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64))

        gt_trajectory = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        pdm_trajectory = np.array([[0.5, 0.0], [1.5, 0.0]], dtype=np.float64)

        scene = SceneContext(
            scene_token="test",
            log_name="test",
            ego_state=ego_state,
            ego_past_states=ego_past_states,
            observation=observation,
            drivable_area_map=drivable_map,
            route_lane_ids=set(),
            centerline=centerline,
            gt_trajectory=gt_trajectory,
            gt_progress=12.0,
            gt_masked_progress=10.0,
            pdm_trajectory=pdm_trajectory,
            pdm_progress=11.0,
            pdm_masked_progress=9.0,
        )

        assert np.array_equal(scene.gt_trajectory, gt_trajectory)
        assert np.array_equal(scene.pdm_trajectory, pdm_trajectory)
        assert scene.gt_trajectory is not scene.pdm_trajectory
        assert scene.gt_progress == 12.0
        assert scene.pdm_progress == 11.0
        assert scene.gt_masked_progress == 10.0
        assert scene.pdm_masked_progress == 9.0

    def test_load_scene_context_uses_metric_cache_trajectory_as_pdm_reference(self, tmp_path: Path):
        if not METRIC_CACHE_SAMPLE_PATH.is_file():
            pytest.skip(
                f"integration sample missing: {METRIC_CACHE_SAMPLE_PATH}"
            )

        with lzma.open(METRIC_CACHE_SAMPLE_PATH, "rb") as file_obj:
            metric_cache = _load_pickle(file_obj)

        log_name = METRIC_CACHE_SAMPLE_PATH.parents[2].name
        scene_token = METRIC_CACHE_SAMPLE_PATH.parent.name
        cache_dir = METRIC_CACHE_SAMPLE_PATH.parents[3]
        boost_cache_dir = tmp_path / "boost_cache"
        previous_boost_cache_dir = get_boost_cache_dir()
        load_scene_context.cache_clear()
        try:
            set_boost_cache_dir(str(boost_cache_dir))
            scene = load_scene_context(cache_dir, log_name, scene_token)
        finally:
            load_scene_context.cache_clear()
            set_boost_cache_dir(previous_boost_cache_dir)

        sampled = list(metric_cache.trajectory.get_sampled_trajectory())
        ego_x = float(metric_cache.ego_state.rear_axle.x)
        ego_y = float(metric_cache.ego_state.rear_axle.y)
        ego_heading = float(metric_cache.ego_state.rear_axle.heading)
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        expected_pdm = []
        for state in sampled[: scene.pdm_trajectory.shape[0]]:
            dx = float(state.rear_axle.x) - ego_x
            dy = float(state.rear_axle.y) - ego_y
            expected_pdm.append(
                [
                    dx * cos_h - dy * sin_h,
                    dx * sin_h + dy * cos_h,
                    float(state.rear_axle.heading) - ego_heading,
                ]
            )
        expected_pdm = np.asarray(expected_pdm, dtype=np.float64)

        assert scene.pdm_trajectory is not None
        np.testing.assert_allclose(scene.pdm_trajectory, expected_pdm, atol=1e-6)
        # Real v1 caches have no human_trajectory. Do not invent an analysis GT
        # from the PDM reference just to satisfy the richer v2 cache schema.
        if getattr(metric_cache, "human_trajectory", None) is None:
            assert scene.gt_trajectory is None
            return
        assert scene.gt_trajectory is not None
        assert scene.pdm_trajectory.shape[0] > scene.gt_trajectory.shape[0]
        assert scene.pdm_trajectory.shape[1] == 3
        assert scene.gt_trajectory.shape[1] >= 2
        assert not np.shares_memory(scene.pdm_trajectory, scene.gt_trajectory)
        assert not np.allclose(
            scene.pdm_trajectory[: scene.gt_trajectory.shape[0], :2],
            scene.gt_trajectory[:, :2],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            scene.gt_trajectory[:2, :2],
            np.asarray(metric_cache.human_trajectory.poses[:2, :2], dtype=np.float64),
            atol=1e-6,
        )

    def test_load_scene_context_isolated_by_boost_cache_dir(self, tmp_path: Path):
        cache_dir = tmp_path / "metric_cache"
        log_name = "log_a"
        token = "scene_a"
        boost_dir_a = tmp_path / "boost_a"
        boost_dir_b = tmp_path / "boost_b"
        boost_dir_a_scene = boost_dir_a / log_name / f"{token}.pkl"
        boost_dir_b_scene = boost_dir_b / log_name / f"{token}.pkl"
        previous_boost_cache_dir = get_boost_cache_dir()
        load_scene_context.cache_clear()
        try:
            boost_dir_a_scene.parent.mkdir(parents=True, exist_ok=True)
            boost_dir_b_scene.parent.mkdir(parents=True, exist_ok=True)
            with boost_dir_a_scene.open("wb") as file_obj:
                pickle.dump(_build_scene(gt_progress=11.0), file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            with boost_dir_b_scene.open("wb") as file_obj:
                pickle.dump(_build_scene(gt_progress=22.0), file_obj, protocol=pickle.HIGHEST_PROTOCOL)

            set_boost_cache_dir(str(boost_dir_a))
            scene_a = load_scene_context(cache_dir, log_name, token)

            set_boost_cache_dir(str(boost_dir_b))
            scene_b = load_scene_context(cache_dir, log_name, token)
        finally:
            load_scene_context.cache_clear()
            set_boost_cache_dir(previous_boost_cache_dir)

        assert scene_a.gt_progress == 11.0
        assert scene_b.gt_progress == 22.0
        assert scene_a.gt_progress != scene_b.gt_progress

    def test_warmup_stats_reads_progress_files_without_scanning_pickles(self, tmp_path: Path):
        boost_dir = tmp_path / "boost"
        previous_boost_cache_dir = get_boost_cache_dir()
        try:
            set_boost_cache_dir(str(boost_dir))
            _write_progress(
                boost_dir / ".warmup" / "progress" / "a.json",
                source_dir="/cache/a",
                status="running",
                total=10,
                converted=3,
                skipped=2,
                failed=1,
            )
            _write_progress(
                boost_dir / ".warmup" / "progress" / "b.json",
                source_dir="/cache/b",
                status="done",
                total=5,
                converted=4,
                skipped=1,
                failed=0,
            )

            stats = get_warmup_stats()
        finally:
            set_boost_cache_dir(previous_boost_cache_dir)

        assert stats["total"] == 15
        assert stats["converted"] == 7
        assert stats["skipped"] == 3
        assert stats["failed"] == 1

    def test_warmup_can_be_disabled(self, tmp_path: Path):
        source_dir = tmp_path / "metric_cache"
        boost_dir = tmp_path / "boost"
        source_dir.mkdir()
        previous_boost_cache_dir = get_boost_cache_dir()
        try:
            set_boost_cache_dir(str(boost_dir))
            stats = warmup_boost_cache(str(source_dir), str(boost_dir), num_workers=0)
        finally:
            set_boost_cache_dir(previous_boost_cache_dir)

        assert stats["total"] == 0
        assert stats["converted"] == 0

    def test_warmup_lock_prevents_duplicate_conversion(self, tmp_path: Path):
        source_dir = tmp_path / "metric_cache"
        boost_dir = tmp_path / "boost"
        source_dir.mkdir()
        lock_path = _lock_path(boost_dir, source_dir)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("already running")

        previous_boost_cache_dir = get_boost_cache_dir()
        try:
            set_boost_cache_dir(str(boost_dir))
            stats = warmup_boost_cache(str(source_dir), str(boost_dir), num_workers=4)
        finally:
            set_boost_cache_dir(previous_boost_cache_dir)

        assert lock_path.exists()
        assert stats["total"] == 0



# ── Helpers ──────────────────────────────────────────────────────────────

NUM_WP = 8  # 8 waypoints @ 0.5s = 4s horizon


def _traj(xs: list[float], ys: list[float]) -> np.ndarray:
    """Build (8, 2) trajectory from x/y lists."""
    assert len(xs) == len(ys) == NUM_WP
    return np.column_stack([xs, ys])


def _straight(dx_per_step: float = 2.5, y: float = 0.0) -> np.ndarray:
    """Straight trajectory along x-axis."""
    xs = [dx_per_step * i for i in range(1, NUM_WP + 1)]
    return _traj(xs, [y] * NUM_WP)


def _build_scene(
    *,
    ego_speed: float = 5.0,
    obstacle_x: float = 30.0,
    obstacle_y: float = 0.0,
    obstacle_vx: float = 0.0,
    road_half_width: float = 4.0,
    road_length: float = 100.0,
    static_obstacle: bool = False,
    no_obstacle: bool = False,
    num_steps: int = 50,
    gt_progress: float | None = None,
) -> SceneContext:
    """Minimal scene builder with configurable obstacles."""
    from shapely.geometry import box

    ego_state = np.zeros(StateIndex.size(), dtype=np.float64)
    ego_state[StateIndex.VELOCITY_X] = ego_speed

    ego_past_states = np.zeros((20, StateIndex.size()), dtype=np.float64)
    for i in range(20):
        t = -(20 - i) * 0.1
        ego_past_states[i, StateIndex.X] = ego_speed * t
        ego_past_states[i, StateIndex.VELOCITY_X] = ego_speed

    road_polygon = box(-10.0, -road_half_width, road_length, road_half_width)
    drivable_map = DrivableMap(
        tokens=["lane_0"],
        types=[SemanticMapLayer.LANE],
        polygons=np.array([road_polygon], dtype=object),
    )
    centerline = PDMPath(np.array([[-10.0, 0.0], [road_length, 0.0]], dtype=np.float64))
    observation = PDMObservation(num_steps=num_steps, interval_time=0.1)

    dynamic_coords = np.empty((0, 4, 2), dtype=np.float64)
    dynamic_tokens: list[str] = []
    dynamic_vels = np.empty((0, 2), dtype=np.float64)
    static_coords = np.empty((0, 4, 2), dtype=np.float64)
    static_tokens: list[str] = []
    track_types: dict[str, str] = {}

    if not no_obstacle:
        obs_corners = np.array([
            [obstacle_x - 2.0, obstacle_y + 1.0],
            [obstacle_x + 2.0, obstacle_y + 1.0],
            [obstacle_x + 2.0, obstacle_y - 1.0],
            [obstacle_x - 2.0, obstacle_y - 1.0],
        ], dtype=np.float64)[None, :, :]

        if static_obstacle:
            static_coords = obs_corners
            static_tokens = ["static_1"]
            track_types["static_1"] = "static"
        else:
            dynamic_coords = obs_corners
            dynamic_tokens = ["vehicle_1"]
            dynamic_vels = np.array([[obstacle_vx, 0.0]], dtype=np.float64)
            track_types["vehicle_1"] = "agent"

    observation.update(
        static_coords=static_coords,
        static_tokens=static_tokens,
        dynamic_coords=dynamic_coords,
        dynamic_tokens=dynamic_tokens,
        dynamic_velocities=dynamic_vels,
        red_light_coords=np.empty((0, 4, 2), dtype=np.float64),
        red_light_tokens=[],
    )

    ctx = SceneContext(
        scene_token="test",
        log_name="test",
        ego_state=ego_state,
        ego_past_states=ego_past_states,
        observation=observation,
        drivable_area_map=drivable_map,
        route_lane_ids={"lane_0"},
        centerline=centerline,
        track_object_types=track_types,
    )
    if gt_progress is not None:
        ctx.gt_progress = gt_progress
        ctx.gt_masked_progress = gt_progress  # populate legacy analysis fields for normalization tests
    return ctx


def _score_rl(scene: SceneContext, traj: np.ndarray, config: RLScorerConfig | None = None):
    scorer = PDMScorer()
    return scorer.score_batch_for_rl(traj[None], scene, config)[0]


# ═════════════════════════════════════════════════════════════════════════
# 1. NC Continuous — collision classification + overlap severity
# ═════════════════════════════════════════════════════════════════════════


class TestNCContinuous:
    """Test continuous no-collision metric with full at-fault classification."""

    def test_no_collision_returns_1(self):
        """No collision → NC = 1.0"""
        scene = _build_scene(obstacle_x=50.0)
        result = _score_rl(scene, _straight(2.0))
        assert result.no_at_fault_collisions == 1.0

    def test_head_on_collision_returns_below_1(self):
        """Head-on into obstacle → at-fault, NC < 1.0"""
        scene = _build_scene(obstacle_x=10.0, obstacle_vx=0.0)
        traj = _straight(3.0)  # drives fast toward obstacle at x=10
        result = _score_rl(scene, traj)
        assert result.no_at_fault_collisions < 1.0, (
            f"Expected NC penalty for head-on collision, got {result.no_at_fault_collisions}"
        )

    def test_static_obstacle_max_penalty_is_05(self):
        """Static obstacle → at-fault max penalty capped at 0.5"""
        scene = _build_scene(obstacle_x=15.0, static_obstacle=True)
        traj = _straight(2.5)
        result = _score_rl(scene, traj)
        assert result.no_at_fault_collisions >= 0.4, (
            f"Static obstacle NC should have floor ~0.5, got {result.no_at_fault_collisions}"
        )

    def test_nc_is_monotonic_with_overlap(self):
        """Larger overlap → lower NC: direct hit worse than glancing."""
        scene = _build_scene(obstacle_x=15.0, obstacle_y=0.0)
        # Direct: y=0 → full overlap
        direct = _score_rl(scene, _straight(2.5, y=0.0))
        # Glancing: y=1.5 → partial overlap
        glancing = _score_rl(scene, _straight(2.5, y=1.5))
        assert glancing.no_at_fault_collisions >= direct.no_at_fault_collisions, (
            f"Glancing ({glancing.no_at_fault_collisions}) should be >= direct ({direct.no_at_fault_collisions})"
        )

    def test_nc_discrete_is_binary(self):
        """Discrete NC must be exactly 0.0 or 0.5 or 1.0."""
        config = RLScorerConfig(safety_mode="discrete")
        scene = _build_scene(obstacle_x=15.0)
        result = _score_rl(scene, _straight(2.5), config)
        assert result.no_at_fault_collisions in (0.0, 0.5, 1.0), (
            f"Discrete NC should be binary-ish, got {result.no_at_fault_collisions}"
        )

    def test_continuous_mode_also_reports_standard_pdms(self):
        """Continuous RL response carries standard v1 PDMS monitoring fields."""
        scene = _build_scene(obstacle_x=50.0)
        traj = _straight(2.0)
        result = _score_rl(scene, traj)
        pdm_result = PDMScorer().score(traj, scene)
        expected_pdms = (
            result.pdms_no_at_fault_collisions
            * result.pdms_drivable_area_compliance
            * (
                5.0 * result.pdms_ego_progress
                + 5.0 * result.pdms_time_to_collision
                + 2.0 * result.pdms_history_comfort
            )
            / 12.0
        )
        assert result.pdms_score == pytest.approx(expected_pdms)
        assert result.pdms_score == pytest.approx(pdm_result.pdm_score)
        assert result.pdms_no_at_fault_collisions == pytest.approx(
            pdm_result.no_at_fault_collisions
        )
        assert result.pdms_drivable_area_compliance == pytest.approx(
            pdm_result.drivable_area_compliance
        )
        assert result.pdms_ego_progress == pytest.approx(pdm_result.ego_progress)
        assert result.pdms_time_to_collision == pytest.approx(pdm_result.time_to_collision)
        assert result.pdms_history_comfort == pytest.approx(pdm_result.history_comfort)
        assert result.pdms_score > 0.0
        assert result.pdms_no_at_fault_collisions in (0.0, 0.5, 1.0)
        assert result.pdms_drivable_area_compliance in (0.0, 1.0)

    def test_nc_range_is_01(self):
        """NC must be in [0, 1] for all trajectories."""
        scene = _build_scene(obstacle_x=15.0)
        for y_off in np.linspace(-3.0, 3.0, 7):
            result = _score_rl(scene, _straight(2.5, y=float(y_off)))
            assert 0.0 <= result.no_at_fault_collisions <= 1.0, (
                f"NC out of range for y_off={y_off}: {result.no_at_fault_collisions}"
            )


# ═════════════════════════════════════════════════════════════════════════
# 2. DAC Continuous — sweep area ratio
# ═════════════════════════════════════════════════════════════════════════


class TestDACContinuous:
    """Test continuous drivable area compliance with sweep-area algorithm."""

    def test_on_road_returns_1(self):
        """Fully on-road trajectory → DAC = 1.0"""
        scene = _build_scene(no_obstacle=True)
        result = _score_rl(scene, _straight(2.0, y=0.0))
        assert result.drivable_area_compliance == 1.0

    def test_fully_offroad_returns_low(self):
        """Trajectory far off road → DAC < 0.5"""
        scene = _build_scene(road_half_width=4.0, no_obstacle=True)
        # y offset = 20m → mostly outside road [-4, 4]
        # Note: LQR starts from y=0, so early sweep is on-road; expect DAC < 0.5
        traj = _straight(2.0, y=20.0)
        result = _score_rl(scene, traj)
        assert result.drivable_area_compliance < 0.5, (
            f"Expected low DAC for far offroad, got {result.drivable_area_compliance}"
        )

    def test_partial_offroad_is_intermediate(self):
        """Trajectory with edge off road → 0 < DAC < 1"""
        scene = _build_scene(road_half_width=4.0, no_obstacle=True)
        # y=3.5 → vehicle corner brushes the edge (vehicle ~2m wide)
        traj = _straight(2.0, y=3.5)
        result = _score_rl(scene, traj)
        assert 0.0 < result.drivable_area_compliance < 1.0, (
            f"Expected intermediate DAC for edge case, got {result.drivable_area_compliance}"
        )

    def test_dac_is_monotonic_with_offroad_distance(self):
        """More off-road → lower DAC."""
        scene = _build_scene(road_half_width=4.0, no_obstacle=True)
        dac_near = _score_rl(scene, _straight(2.0, y=3.0)).drivable_area_compliance
        dac_far = _score_rl(scene, _straight(2.0, y=6.0)).drivable_area_compliance
        assert dac_near > dac_far, f"Near ({dac_near}) should be > far ({dac_far})"

    def test_dac_is_continuous(self):
        """Small y change → small DAC change (no large jumps)."""
        scene = _build_scene(road_half_width=4.0, no_obstacle=True)
        y_values = np.linspace(0.0, 6.0, 13)
        dac_values = []
        for y in y_values:
            r = _score_rl(scene, _straight(2.0, y=float(y)))
            dac_values.append(r.drivable_area_compliance)

        # Check no jump > 0.5 between adjacent y values (0.5m spacing)
        for i in range(1, len(dac_values)):
            diff = abs(dac_values[i] - dac_values[i - 1])
            assert diff < 0.5, (
                f"DAC jump too large between y={y_values[i-1]:.1f} ({dac_values[i-1]:.3f}) "
                f"and y={y_values[i]:.1f} ({dac_values[i]:.3f}), diff={diff:.3f}"
            )

    def test_dac_range_is_01(self):
        """DAC must be in [0, 1] for all trajectories."""
        scene = _build_scene(no_obstacle=True)
        for y in [-10.0, -4.0, 0.0, 4.0, 10.0]:
            result = _score_rl(scene, _straight(2.0, y=y))
            assert 0.0 <= result.drivable_area_compliance <= 1.0


# ═════════════════════════════════════════════════════════════════════════
# 3. Config alignment: v1 defaults
# ═════════════════════════════════════════════════════════════════════════


class TestRLConfig:
    """Test RLScorerConfig.v1() defaults and weight-zero gating."""

    def test_v1_inactive_weights_are_zero(self):
        cfg = RLScorerConfig.v1()
        assert cfg.ddc_weight == 0.0
        assert cfg.tlc_weight == 0.0
        assert cfg.lk_weight == 0.0

    def test_v1_active_weights_match_plan(self):
        cfg = RLScorerConfig.v1()
        assert cfg.nc_weight == 5.0
        assert cfg.dac_weight == 3.0
        assert cfg.ep_weight == 5.0
        assert cfg.ttc_weight == 5.0
        assert cfg.hc_weight == 2.0

    def test_weight_zero_skips_computation(self):
        """DDC/TLC/LK = 1.0 when weight = 0 (skipped)."""
        scene = _build_scene(no_obstacle=True)
        config = RLScorerConfig.v1()
        result = _score_rl(scene, _straight(2.0), config)
        assert result.driving_direction_compliance == 1.0
        assert result.traffic_light_compliance == 1.0
        assert result.lane_keeping == 1.0

    def test_default_uses_v1(self):
        """Default RLScorerConfig matches v1() for inactive weights."""
        default = RLScorerConfig()
        assert default.ddc_weight == 0.0
        assert default.tlc_weight == 0.0
        assert default.lk_weight == 0.0


# ═════════════════════════════════════════════════════════════════════════
# 5. Pipeline alignment: RL uses same simulation as PDMS
# ═════════════════════════════════════════════════════════════════════════


class TestPipelineAlignment:
    """Verify RL scoring uses the same _build_proposals path as PDMS."""

    def test_discrete_nc_matches_pdms(self):
        """Discrete RL NC should match PDMS NC (same simulation pipeline)."""
        scene = _build_scene(obstacle_x=50.0)
        traj = _straight(2.0)
        scorer = PDMScorer()

        pdms = scorer.score(traj, scene)
        rl_discrete = scorer.score_batch_for_rl(
            traj[None], scene, RLScorerConfig(safety_mode="discrete")
        )[0]

        assert abs(pdms.no_at_fault_collisions - rl_discrete.no_at_fault_collisions) < 1e-6
        assert abs(pdms.drivable_area_compliance - rl_discrete.drivable_area_compliance) < 1e-6

    def test_all_sub_rewards_in_01(self):
        """All sub-rewards must be in [0, 1] for any trajectory."""
        scene = _build_scene()
        for y in [0.0, 2.0, -2.0, 5.0, -5.0]:
            result = _score_rl(scene, _straight(2.0, y=float(y)))
            for key, val in result.sub_rewards().items():
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1] for y={y}"

    def test_grounding_fields_centered(self):
        scene = _build_scene(no_obstacle=True)
        result = _score_rl(scene, _straight(2.0, y=0.0))
        assert abs(result.centerline_lateral_offset_start_signed) < 1e-6
        assert abs(result.centerline_lateral_offset_end_signed) < 1e-6
        assert result.centerline_distance_mean >= 0.0
        assert result.centerline_distance_max >= result.centerline_distance_mean
        assert 0.0 <= result.in_intersection_fraction <= 1.0
        assert 0.0 <= result.oncoming_fraction <= 1.0
        assert 0.0 <= result.non_drivable_fraction <= 1.0
        assert 0.0 <= result.multiple_lanes_fraction <= 1.0

    def test_grounding_fields_offset_sign(self):
        scene = _build_scene(no_obstacle=True)
        left = _score_rl(scene, _straight(2.0, y=1.0))
        right = _score_rl(scene, _straight(2.0, y=-1.0))
        assert left.centerline_lateral_offset_start_signed * right.centerline_lateral_offset_start_signed <= 0.0
        assert left.centerline_lateral_offset_end_signed * right.centerline_lateral_offset_end_signed <= 0.0

    def test_boundary_distance_ordering(self):
        scene = _build_scene(road_half_width=4.0, no_obstacle=True)
        centered = _score_rl(scene, _straight(2.0, y=0.0))
        edge = _score_rl(scene, _straight(2.0, y=3.0))
        assert centered.boundary_distance_min >= edge.boundary_distance_min


    def test_rl_score_is_soft_gated(self):
        """rl_score = (NC*DAC*DDC*TLC)^alpha × weighted_avg(EP, TTC, LK, HC)."""
        scene = _build_scene(no_obstacle=True)
        config = RLScorerConfig.v1()
        result = _score_rl(scene, _straight(2.0), config)

        sub = result.sub_rewards()
        safety = sub["nc"] * sub["dac"] * sub["ddc"] * sub["tlc"]
        safety_gate = safety ** config.safety_gate_alpha

        perf_w = config.performance_weights
        perf_vals = np.array([sub["ep"], sub["ttc"], sub["lk"], sub["hc"]])
        performance = (perf_vals * perf_w).sum() / perf_w.sum()

        expected = safety_gate * performance
        assert abs(result.rl_score - expected) < 1e-6, (
            f"rl_score ({result.rl_score}) != soft_gate ({expected})"
        )
