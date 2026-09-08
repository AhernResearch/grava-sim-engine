from __future__ import annotations

import hashlib
import json
import logging
import lzma
import os
import pickle
import socket
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from grava_sim_engine.core.geometry import PDMPath
from grava_sim_engine.core.observation import PDMObservation
from grava_sim_engine.core.occupancy import DrivableMap, OccupancyMap
from grava_sim_engine.core.enums import MultiMetricIndex
from grava_sim_engine.core.scoring.base import ScorerBase
from grava_sim_engine.core.types import SceneContext

logger = logging.getLogger(__name__)

RED_LIGHT_TOKEN_PREFIX = "red_light"
SCENARIO_TYPES = ("unknown", "original", "synthetic")

# ── Boost cache state ──────────────────────────────────────────────────

_boost_cache_dir: str | None = None
_warmup_stats: dict[str, int] = {"converted": 0, "skipped": 0, "failed": 0, "total": 0}

try:
    from navsim.planning.metric_caching.metric_cache import MetricCache

    _HAS_NAVSIM = True
except ImportError:  # pragma: no cover - exercised only without NavSim env
    MetricCache = Any  # type: ignore[assignment]
    _HAS_NAVSIM = False

from grava_sim_engine.adapters.navsim._stubs import MetricCacheUnpickler


def _load_pickle(file_obj: Any) -> Any:
    """Load a pickle, using stub unpickler when navsim is not installed."""
    if _HAS_NAVSIM:
        return pickle.load(file_obj)
    return MetricCacheUnpickler(file_obj).load()


def _candidate_metric_cache_paths(cache_root: Path, log_name: str, token: str) -> list[Path]:
    candidates = [
        cache_root / log_name / scenario_type / token / "metric_cache.pkl"
        for scenario_type in SCENARIO_TYPES
    ]
    if any(path.is_file() for path in candidates):
        return candidates
    candidates.extend(sorted(cache_root.glob(f"**/{token}/metric_cache.pkl")))
    return candidates


def load_metric_cache(cache_dir: str | Path, log_name: str, token: str) -> MetricCache:
    """Load a NavSim MetricCache pickle from the standard cache layout."""
    cache_root = Path(cache_dir)
    for pkl_path in _candidate_metric_cache_paths(cache_root, log_name, token):
        if not pkl_path.exists():
            continue
        with lzma.open(pkl_path, "rb") as file_obj:
            return _load_pickle(file_obj)

    raise FileNotFoundError(
        f"MetricCache not found for token={token}, log_name={log_name} under {cache_root}"
    )


def _build_ego_state_array(ego_state: Any) -> np.ndarray:
    state = np.zeros(11, dtype=np.float64)
    rear_axle = ego_state.rear_axle
    dynamic = ego_state.dynamic_car_state

    state[0] = rear_axle.x
    state[1] = rear_axle.y
    state[2] = rear_axle.heading
    state[3] = dynamic.rear_axle_velocity_2d.x
    state[4] = dynamic.rear_axle_velocity_2d.y
    state[5] = dynamic.rear_axle_acceleration_2d.x
    state[6] = dynamic.rear_axle_acceleration_2d.y
    state[7] = ego_state.tire_steering_angle
    state[8] = dynamic.tire_steering_rate
    state[9] = dynamic.angular_velocity
    state[10] = dynamic.angular_acceleration
    return state


def _state_time_seconds(state: Any, fallback_index: int, fallback_dt: float) -> float:
    time_point = getattr(state, "time_point", None)
    if time_point is None:
        return fallback_index * fallback_dt
    return float(time_point.time_us) / 1e6


def _estimate_xy_velocities(positions: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    if len(positions) <= 1:
        return np.zeros((len(positions), 2), dtype=np.float64)

    if np.any(np.diff(times_s) <= 0.0):
        times_s = np.arange(len(positions), dtype=np.float64) * 0.1

    velocities = np.zeros((len(positions), 2), dtype=np.float64)
    velocities[:, 0] = np.gradient(positions[:, 0], times_s, edge_order=1)
    velocities[:, 1] = np.gradient(positions[:, 1], times_s, edge_order=1)
    return velocities


def _sample_past_states(past_human_trajectory: Any, interval_time: float) -> np.ndarray:
    if past_human_trajectory is None:
        return np.zeros((0, 11), dtype=np.float64)

    sampled_states = list(past_human_trajectory.get_sampled_trajectory())
    if len(sampled_states) == 0:
        return np.zeros((0, 11), dtype=np.float64)

    positions = np.array(
        [[state.rear_axle.x, state.rear_axle.y] for state in sampled_states],
        dtype=np.float64,
    )
    headings = np.array([state.rear_axle.heading for state in sampled_states], dtype=np.float64)
    times_s = np.array(
        [_state_time_seconds(state, idx, interval_time) for idx, state in enumerate(sampled_states)],
        dtype=np.float64,
    )
    velocities = _estimate_xy_velocities(positions, times_s)

    state_array = np.zeros((len(sampled_states), 11), dtype=np.float64)
    state_array[:, 0:2] = positions
    state_array[:, 2] = headings
    state_array[:, 3:5] = velocities
    return state_array


def _build_occupancy_map(tokens: list[str], geometries: np.ndarray) -> OccupancyMap | None:
    if len(tokens) == 0:
        return None
    return OccupancyMap(tokens=tokens, polygons=geometries)


def _convert_navsim_observation(navsim_observation: Any) -> PDMObservation:
    navsim_maps = getattr(navsim_observation, "_occupancy_maps", None)
    if navsim_maps is None or len(navsim_maps) == 0:
        raise ValueError("MetricCache observation does not contain occupancy maps")

    interval_time = float(getattr(navsim_observation, "_sample_interval", 0.1))
    observation = PDMObservation(num_steps=len(navsim_maps), interval_time=interval_time)

    occupancy_maps: list[OccupancyMap | None] = []
    red_light_maps: list[OccupancyMap | None] = []
    for navsim_map in navsim_maps:
        tokens = [str(token) for token in getattr(navsim_map, "_tokens")]
        geometries = np.asarray(getattr(navsim_map, "_geometries"), dtype=object)

        occupancy_maps.append(_build_occupancy_map(tokens, geometries))

        red_light_indices = [
            index for index, token in enumerate(tokens) if token.startswith(RED_LIGHT_TOKEN_PREFIX)
        ]
        red_light_tokens = [tokens[index] for index in red_light_indices]
        red_light_geometries = geometries[red_light_indices]
        red_light_maps.append(_build_occupancy_map(red_light_tokens, red_light_geometries))

    global_to_local_idcs = getattr(navsim_observation, "_global_to_local_idcs", None)
    if global_to_local_idcs is None:
        global_to_local = list(range(len(occupancy_maps)))
    else:
        global_to_local = [int(index) for index in global_to_local_idcs]

    if global_to_local and max(global_to_local) >= len(occupancy_maps):
        raise ValueError("Observation global_to_local_idcs contains indices outside occupancy map range")

    observation._occupancy_maps = occupancy_maps
    observation._red_light_maps = red_light_maps
    observation._global_to_local_idcs = global_to_local
    observation._observation_sample_res = int(
        getattr(navsim_observation, "_observation_sample_res", observation._observation_sample_res)
    )
    return observation


def _convert_drivable_area_map(navsim_drivable_map: Any) -> DrivableMap:
    tokens = [str(token) for token in getattr(navsim_drivable_map, "_tokens")]
    types = [
        layer.name if hasattr(layer, "name") else str(layer)
        for layer in getattr(navsim_drivable_map, "_map_types")
    ]
    polygons = list(getattr(navsim_drivable_map, "_geometries"))
    return DrivableMap(tokens=tokens, types=types, polygons=polygons)


def _convert_centerline(navsim_centerline: Any) -> PDMPath:
    xy_points = np.array(
        [[point.x, point.y] for point in getattr(navsim_centerline, "_discrete_path")],
        dtype=np.float64,
    )
    return PDMPath(xy_points)


def _extract_collided_track_ids(metric_cache: Any) -> set[str]:
    collided_track_ids = getattr(metric_cache, "collided_track_ids", None)
    if collided_track_ids is None:
        observation = getattr(metric_cache, "observation", None)
        collided_track_ids = getattr(observation, "_collided_track_ids", None)
        if collided_track_ids is None and observation is not None:
            collided_track_ids = getattr(observation, "collided_track_ids", [])
    return {str(track_id) for track_id in (collided_track_ids or [])}


def _extract_log_name_from_path(file_path: str | Path) -> str | None:
    """Extract log_name from MetricCache file_path.

    Path format: .../metric_cache*/{log_name}/{scenario_type}/{token}/metric_cache.pkl
    """
    parts = Path(file_path).parts
    # Find 'metric_cache' in path, the next part is log_name
    for i, p in enumerate(parts):
        if "metric_cache" in p and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _extract_pdm_trajectory_xy(
    metric_cache: Any, ego_state_array: np.ndarray,
    num_future_steps: int = 40,
) -> np.ndarray | None:
    """Extract MetricCache.trajectory as ego-relative PDM reference waypoints.

    MetricCache.trajectory is the explicit PDM closed-loop reference used by V1 scoring.
    We keep sampled[0:num_future_steps+1] so t=0 remains the ego anchor.
    Includes heading (x, y, heading) to avoid heading derivation errors.
    """
    trajectory = getattr(metric_cache, "trajectory", None)
    if trajectory is None:
        return None
    sampled = list(trajectory.get_sampled_trajectory())
    if len(sampled) < 2:
        return None

    end_idx = min(num_future_steps + 1, len(sampled))
    global_xyh = np.array(
        [[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading] for s in sampled[0:end_idx]],
        dtype=np.float64,
    )

    ego_x = ego_state_array[0]
    ego_y = ego_state_array[1]
    ego_h = ego_state_array[2]
    cos_h = np.cos(-ego_h)
    sin_h = np.sin(-ego_h)

    dx = global_xyh[:, 0] - ego_x
    dy = global_xyh[:, 1] - ego_y
    local_x = dx * cos_h - dy * sin_h
    local_y = dx * sin_h + dy * cos_h
    local_heading = global_xyh[:, 2] - ego_h

    return np.stack([local_x, local_y, local_heading], axis=1)


def _extract_gt_trajectory_xy(metric_cache: Any) -> np.ndarray | None:
    """Extract human/open-loop GT trajectory as ego-relative waypoints.

    NavSim MetricCache.human_trajectory stores local poses at 0.5s spacing.
    Returns (N, 3) with [x, y, heading] when heading is available, (N, 2) otherwise.
    """
    human_trajectory = getattr(metric_cache, "human_trajectory", None)
    poses = getattr(human_trajectory, "poses", None)
    if poses is None:
        return None

    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] < 2:
        return None

    return poses.copy()


_AGENT_TYPE_NAMES = {"VEHICLE", "PEDESTRIAN", "BICYCLE", "EGO"}


def _extract_track_object_types(metric_cache: Any) -> dict[str, str]:
    """Build token → 'agent'|'static' mapping.

    Sources (in priority order):
    1. observation.unique_objects — covers all tokens the scorer might encounter
    2. future_tracked_objects — supplementary coverage
    """
    result: dict[str, str] = {}

    # Primary: observation.unique_objects (same source NavSim uses for collision type)
    unique_objects = getattr(getattr(metric_cache, "observation", None), "unique_objects", None)
    if unique_objects is not None:
        for token, obj in unique_objects.items():
            obj_type = getattr(obj, "tracked_object_type", None)
            type_name = obj_type.name if obj_type is not None else "UNKNOWN"
            result[str(token)] = "agent" if type_name in _AGENT_TYPE_NAMES else "static"

    # Supplementary: future_tracked_objects
    for dt in getattr(metric_cache, "future_tracked_objects", []):
        tracked_objects = getattr(dt, "tracked_objects", None)
        if tracked_objects is None:
            continue
        for obj in tracked_objects.tracked_objects:
            token = str(getattr(obj, "track_token", ""))
            if not token:
                continue
            obj_type = getattr(obj, "tracked_object_type", None)
            type_name = obj_type.name if obj_type is not None else "UNKNOWN"
            result[token] = "agent" if type_name in _AGENT_TYPE_NAMES else "static"
    return result


def _extract_track_speeds(metric_cache: Any) -> dict[str, float]:
    """Build token → GT speed (m/s) mapping from observation.unique_objects.

    NavSim uses tracked_object.velocity.magnitude() to determine if a track is
    stopped (threshold 0.05 m/s). We extract this GT value so our collision
    classification matches NavSim exactly.
    """
    result: dict[str, float] = {}
    unique_objects = getattr(getattr(metric_cache, "observation", None), "unique_objects", None)
    if unique_objects is None:
        return result
    for token, obj in unique_objects.items():
        velocity = getattr(obj, "velocity", None)
        if velocity is not None and hasattr(velocity, "magnitude"):
            result[str(token)] = float(velocity.magnitude())
        else:
            # Non-Agent types (StaticObject) are always "stopped"
            result[str(token)] = 0.0
    return result


def _attach_rl_precompute(ctx: SceneContext) -> bool:
    """Populate optional reference progress fields for warm and lazy paths.

    Returns True when the context was mutated.  The boost-cache caller uses this
    to backfill old pickles that were created before PDM reference fields were
    persisted.
    """
    changed = False
    scorer = ScorerBase()
    safety_indices = [MultiMetricIndex.NO_COLLISION, MultiMetricIndex.DRIVABLE_AREA]

    if (
        getattr(ctx, "gt_trajectory", None) is not None
        and (getattr(ctx, "gt_progress", None) is None or getattr(ctx, "gt_masked_progress", None) is None)
    ):
        gt_result = scorer._simulate_and_score_gt(
            ctx,
            multi_indices=safety_indices,
            weighted_indices=[],
        )
        if gt_result is not None:
            if ctx.gt_progress is None:
                ctx.gt_progress = gt_result.progress
                changed = True
            if getattr(ctx, "gt_masked_progress", None) is None:
                gt_nc = float(gt_result.multi_metrics[MultiMetricIndex.NO_COLLISION])
                gt_dac = float(gt_result.multi_metrics[MultiMetricIndex.DRIVABLE_AREA])
                ctx.gt_masked_progress = gt_result.progress * gt_nc * gt_dac
                changed = True

    if (
        getattr(ctx, "pdm_trajectory", None) is not None
        and (getattr(ctx, "pdm_progress", None) is None or getattr(ctx, "pdm_masked_progress", None) is None)
    ):
        pdm_result = scorer._simulate_and_score_pdm(
            ctx,
            multi_indices=safety_indices,
            weighted_indices=[],
        )
        if pdm_result is not None:
            if ctx.pdm_progress is None:
                ctx.pdm_progress = pdm_result.progress
                changed = True
            if getattr(ctx, "pdm_masked_progress", None) is None:
                pdm_nc = float(pdm_result.multi_metrics[MultiMetricIndex.NO_COLLISION])
                pdm_dac = float(pdm_result.multi_metrics[MultiMetricIndex.DRIVABLE_AREA])
                ctx.pdm_masked_progress = pdm_result.progress * pdm_nc * pdm_dac
                changed = True

    return changed


def metric_cache_to_scene_context(metric_cache: MetricCache, scene_token: str) -> SceneContext:
    """Convert a NavSim MetricCache object into a grava-sim-engine SceneContext."""
    interval_time = float(getattr(metric_cache.observation, "_sample_interval", 0.1))

    # Get log_name: prefer attribute, fallback to extracting from file_path
    log_name = getattr(metric_cache, "log_name", None)
    if log_name is None and hasattr(metric_cache, "file_path"):
        log_name = _extract_log_name_from_path(metric_cache.file_path)
    if log_name is None:
        raise ValueError("log_name not found in MetricCache and could not extract from file_path")

    ego_state_array = _build_ego_state_array(metric_cache.ego_state)

    ctx = SceneContext(
        scene_token=scene_token,
        log_name=str(log_name),
        ego_state=ego_state_array,
        ego_past_states=_sample_past_states(getattr(metric_cache, "past_human_trajectory", None), interval_time),
        observation=_convert_navsim_observation(metric_cache.observation),
        drivable_area_map=_convert_drivable_area_map(metric_cache.drivable_area_map),
        route_lane_ids={str(lane_id) for lane_id in metric_cache.route_lane_ids},
        centerline=_convert_centerline(metric_cache.centerline),
        collided_track_ids=_extract_collided_track_ids(metric_cache),
        gt_trajectory=_extract_gt_trajectory_xy(metric_cache),
        pdm_trajectory=_extract_pdm_trajectory_xy(metric_cache, ego_state_array),
        track_object_types=_extract_track_object_types(metric_cache),
        track_speeds=_extract_track_speeds(metric_cache),
    )

    _attach_rl_precompute(ctx)
    return ctx


@lru_cache(maxsize=256)
def _load_scene_context_cached(
    cache_dir: str | Path,
    log_name: str,
    token: str,
    boost_cache_dir: str | None,
) -> SceneContext:
    # L2: boost cache (fast pickle, ~24ms)
    if boost_cache_dir is not None:
        boost_path = _boost_cache_path(boost_cache_dir, log_name, token)
        if boost_path.exists():
            ctx = _load_from_boost(boost_path)
            if _attach_rl_precompute(ctx):
                try:
                    _save_to_boost(ctx, boost_path)
                except OSError:
                    logger.warning(
                        "Failed to update boost cache with backfilled reference metrics: %s",
                        boost_path,
                        exc_info=True,
                    )
            return ctx

    # L3: original LZMA MetricCache (~1s)
    metric_cache = load_metric_cache(cache_dir=cache_dir, log_name=log_name, token=token)
    ctx = metric_cache_to_scene_context(metric_cache=metric_cache, scene_token=token)

    # Lazy write-back to boost cache
    if boost_cache_dir is not None:
        try:
            _save_to_boost(ctx, _boost_cache_path(boost_cache_dir, log_name, token))
        except OSError:
            logger.warning(
                "Failed to write boost cache for %s/%s",
                log_name,
                token,
                exc_info=True,
            )

    return ctx


def load_scene_context(cache_dir: str | Path, log_name: str, token: str) -> SceneContext:
    return _load_scene_context_cached(cache_dir, log_name, token, _boost_cache_dir)


load_scene_context.cache_clear = _load_scene_context_cached.cache_clear
load_scene_context.cache_info = _load_scene_context_cached.cache_info
load_scene_context.cache_parameters = _load_scene_context_cached.cache_parameters


# ── Boost cache layer ──────────────────────────────────────────────────


def set_boost_cache_dir(path: str | None) -> None:
    global _boost_cache_dir
    _boost_cache_dir = path


def get_boost_cache_dir() -> str | None:
    return _boost_cache_dir


def get_warmup_stats() -> dict[str, int]:
    """Return boost cache stats without recursively scanning boost files.

    Warmup workers write one small JSON progress file per source dataset.  Health checks
    read those files instead of globbing tens of thousands of boost pickles.
    """
    boost_dir = _boost_cache_dir
    if boost_dir is None:
        return {"converted": 0, "total": 0}
    boost_root = Path(boost_dir)
    if not boost_root.exists():
        return {"converted": 0, "total": _warmup_stats.get("total", 0)}

    aggregate = {"converted": 0, "skipped": 0, "failed": 0, "total": 0}
    progress_dir = _boost_progress_dir(boost_root)
    try:
        progress_files = list(progress_dir.glob("*.json"))
    except OSError:
        progress_files = []

    for progress_file in progress_files:
        try:
            data = json.loads(progress_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for key in aggregate:
            aggregate[key] += int(data.get(key, 0) or 0)

    if aggregate["total"] == 0:
        aggregate.update({k: int(v) for k, v in _warmup_stats.items()})
    return aggregate


def _boost_cache_path(boost_dir: str, log_name: str, token: str) -> Path:
    return Path(boost_dir) / log_name / f"{token}.pkl"


def _boost_meta_dir(boost_root: Path) -> Path:
    return boost_root / ".warmup"


def _boost_progress_dir(boost_root: Path) -> Path:
    return _boost_meta_dir(boost_root) / "progress"


def _source_key(source_dir: str | Path) -> str:
    source = str(Path(source_dir).resolve())
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def _lock_path(boost_dir: str | Path, source_dir: str | Path) -> Path:
    return _boost_meta_dir(Path(boost_dir)) / f"{_source_key(source_dir)}.lock"


def _progress_path(boost_dir: str | Path, source_dir: str | Path) -> Path:
    return _boost_progress_dir(Path(boost_dir)) / f"{_source_key(source_dir)}.json"


def _write_progress(
    progress_path: Path,
    *,
    source_dir: str,
    status: str,
    total: int,
    converted: int,
    skipped: int,
    failed: int,
) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_dir": source_dir,
        "status": status,
        "total": int(total),
        "converted": int(converted),
        "skipped": int(skipped),
        "failed": int(failed),
        "updated_at": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    tmp = progress_path.with_suffix(progress_path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    os.replace(tmp, progress_path)


def _save_to_boost(ctx: SceneContext, boost_path: Path) -> None:
    boost_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=boost_path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            pickle.dump(ctx, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(tmp_path, boost_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.debug("Failed to remove temporary boost cache file %s", tmp_path, exc_info=True)
        raise


def _load_from_boost(boost_path: Path) -> SceneContext:
    with open(boost_path, "rb") as f:
        return pickle.load(f)


def _convert_one_scene(args: tuple) -> str:
    """Convert a single LZMA MetricCache to boost pickle. Returns status string."""
    source_path, boost_dir = args
    try:
        # Extract log_name and token from path:
        # .../metric_cache*/{log_name}/{scenario_type}/{token}/metric_cache.pkl
        parts = source_path.parts
        token = parts[-2]
        log_name = parts[-4]

        boost_path = _boost_cache_path(boost_dir, log_name, token)
        if boost_path.exists():
            return "skipped"

        with lzma.open(source_path, "rb") as f:
            mc = _load_pickle(f)
        ctx = metric_cache_to_scene_context(mc, token)
        _save_to_boost(ctx, boost_path)
        return "converted"
    except Exception:
        logger.warning("Failed to convert %s", source_path, exc_info=True)
        return "failed"


def warmup_boost_cache(source_dir: str, boost_dir: str, num_workers: int = 32) -> dict[str, int]:
    """Scan source_dir for LZMA MetricCache files, convert to boost pickle in parallel.

    Uses subprocess to avoid GIL contention with the server's main thread.
    """
    source_root = Path(source_dir)
    boost_root = Path(boost_dir)
    boost_root.mkdir(parents=True, exist_ok=True)
    progress_path = _progress_path(boost_root, source_root)

    if num_workers <= 0:
        _write_progress(
            progress_path,
            source_dir=str(source_root),
            status="disabled",
            total=0,
            converted=0,
            skipped=0,
            failed=0,
        )
        logger.info("Boost warmup disabled for %s (workers=%d)", source_dir, num_workers)
        return get_warmup_stats()

    lock_path = _lock_path(boost_root, source_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        logger.info("Boost warmup already running for %s (lock=%s)", source_dir, lock_path)
        return get_warmup_stats()

    with os.fdopen(lock_fd, "w") as lock_file:
        lock_file.write(f"pid={os.getpid()} host={socket.gethostname()} source={source_root}\n")

    # Fast estimate: count log-level subdirs (avoid slow AFS glob in main thread)
    try:
        log_dirs = [p for p in source_root.iterdir() if p.is_dir()]
        estimated_total = len(log_dirs) * 80  # rough estimate: ~80 scenes per log
        _warmup_stats["total"] = _warmup_stats.get("total", 0) + estimated_total
        logger.info("Boost warmup: ~%d scenes estimated (subprocess, %d workers)", estimated_total, num_workers)
    except OSError:
        _warmup_stats["total"] = 0
        logger.warning("Boost warmup: cannot list source_dir %s", source_dir)
        try:
            lock_path.unlink()
        except OSError:
            logger.debug("Failed to remove warmup lock %s", lock_path, exc_info=True)
        return get_warmup_stats()

    # Run conversion in a separate process to avoid GIL contention
    # The subprocess does its own glob + conversion
    script = f"""
import sys, logging, json, os, time, socket
sys.path.insert(0, '{Path(__file__).resolve().parents[3]}')
from grava_sim_engine.adapters.navsim.cache_loader import (
    _convert_one_scene, _warmup_stats, metric_cache_to_scene_context,
)
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('warmup')

source_root = Path({str(source_root)!r})
boost_dir = {str(boost_root)!r}
progress_path = Path({str(progress_path)!r})
all_pkls = sorted(source_root.glob('**/metric_cache.pkl'))
total = len(all_pkls)
log.info('Warmup subprocess: %d scenes, %d workers', total, {num_workers})

converted = skipped = failed = 0

def write_progress(status):
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {{
        'source_dir': str(source_root),
        'status': status,
        'total': total,
        'converted': converted,
        'skipped': skipped,
        'failed': failed,
        'updated_at': time.time(),
        'host': socket.gethostname(),
        'pid': os.getpid(),
    }}
    tmp = progress_path.with_suffix(progress_path.suffix + f'.tmp.{{os.getpid()}}')
    tmp.write_text(json.dumps(payload, sort_keys=True))
    os.replace(tmp, progress_path)

write_progress('running')
tasks = ((p, boost_dir) for p in all_pkls)
chunksize = max(1, min(64, total // max(1, {num_workers} * 8) or 1))
with ProcessPoolExecutor(max_workers={num_workers}) as pool:
    for i, status in enumerate(pool.map(_convert_one_scene, tasks, chunksize=chunksize), 1):
        if status == 'converted':
            converted += 1
        elif status == 'skipped':
            skipped += 1
        else:
            failed += 1
        if i % 2000 == 0 or i == total:
            write_progress('running')
            log.info('Progress: %d/%d (converted=%d, skipped=%d, failed=%d)', i, total, converted, skipped, failed)

write_progress('done')
log.info('Done: converted=%d, skipped=%d, failed=%d', converted, skipped, failed)
"""
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream output and update stats
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logger.info("[warmup] %s", line)
                # Parse exact total from subprocess "Warmup subprocess: N scenes" line
                if "Warmup subprocess:" in line:
                    try:
                        n_str = line.split("Warmup subprocess:")[1].split("scenes")[0].strip()
                        # Correct the estimate with the actual count from glob
                        _warmup_stats["total"] = _warmup_stats.get("total", 0) - estimated_total + int(n_str)
                    except (IndexError, ValueError):
                        logger.debug("Failed to parse warmup total from line: %s", line, exc_info=True)

        proc.wait()
        if proc.returncode != 0:
            logger.warning("Boost warmup subprocess exited with code %s for %s", proc.returncode, source_dir)
        logger.info("Boost warmup finished for %s.", source_dir)
        return get_warmup_stats()
    finally:
        try:
            lock_path.unlink()
        except OSError:
            logger.debug("Failed to remove warmup lock %s", lock_path, exc_info=True)


__all__ = [
    "load_metric_cache",
    "metric_cache_to_scene_context",
    "load_scene_context",
    "set_boost_cache_dir",
    "get_boost_cache_dir",
    "get_warmup_stats",
    "warmup_boost_cache",
]
