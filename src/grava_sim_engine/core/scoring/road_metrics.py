"""Road metrics shared by the driving scorers."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from ..enums import BBCoordsIndex, EgoAreaIndex
from ..types import SceneContext
from . import scene_geometry
from .config import RLScorerConfig

RED_LIGHT_TOKEN_PREFIX = "red_light"
_DAC_STRUCTURAL_LAYERS = frozenset({"roadblock", "intersection"})


def drivable_area_compliance(ego_areas: np.ndarray) -> np.ndarray:
    off_road = ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].any(axis=1)
    scores = np.ones(len(ego_areas), dtype=np.float64)
    scores[off_road] = 0.0
    return scores


def driving_direction_compliance(
    ego_coords: np.ndarray,
    ego_areas: np.ndarray,
    scene: SceneContext,
    *,
    horizon: float = 1.0,
    compliance_threshold: float = 2.0,
    violation_threshold: float = 6.0,
) -> np.ndarray:
    centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
    oncoming_progress = np.zeros(centers.shape[:2], dtype=np.float64)
    oncoming_progress[:, 1:] = np.linalg.norm(centers[:, 1:] - centers[:, :-1], axis=-1)
    not_oncoming = ~ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
    oncoming_progress[not_oncoming] = 0.0
    horizon_steps = max(int(round(horizon / scene_geometry.dt(scene))), 1)
    rolling_progress = np.zeros_like(oncoming_progress)
    for time_idx in range(oncoming_progress.shape[1]):
        start_idx = max(0, time_idx - horizon_steps)
        rolling_progress[:, time_idx] = oncoming_progress[:, start_idx : time_idx + 1].sum(axis=1)
    max_progress = rolling_progress.max(axis=1)
    scores = np.ones(len(ego_coords), dtype=np.float64)
    medium_mask = (max_progress >= compliance_threshold) & (max_progress < violation_threshold)
    severe_mask = max_progress >= violation_threshold
    scores[medium_mask] = 0.5
    scores[severe_mask] = 0.0
    return scores


def traffic_light_compliance(ego_polygons: np.ndarray, scene: SceneContext) -> np.ndarray:
    scores = np.ones(len(ego_polygons), dtype=np.float64)
    for time_idx in range(ego_polygons.shape[1]):
        red_light_map = scene_geometry.get_red_light_map(scene, time_idx)
        if red_light_map is None:
            continue
        for proposal_idx in range(len(ego_polygons)):
            if scores[proposal_idx] == 0.0:
                continue
            ego_polygon = ego_polygons[proposal_idx, time_idx]
            if not red_light_map.intersects(ego_polygon)[0]:
                continue
            for token in red_light_map.get_colliding_tokens(ego_polygon):
                if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                    scores[proposal_idx] = 0.0
                    break
    return scores


def lane_keeping(
    ego_coords: np.ndarray,
    ego_areas: np.ndarray,
    scene: SceneContext,
    *,
    deviation: float = 0.5,
    horizon: float = 2.0,
) -> np.ndarray:
    centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
    horizon_steps = max(int(round(horizon / scene_geometry.dt(scene))), 1)
    scores = np.ones(len(ego_coords), dtype=np.float64)
    for proposal_idx in range(len(ego_coords)):
        consecutive_exceeds = 0
        violated = False
        for time_idx in range(ego_coords.shape[1]):
            if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                continue
            dev = Point(*centers[proposal_idx, time_idx]).distance(scene.centerline.linestring)
            if dev > deviation:
                consecutive_exceeds += 1
                if consecutive_exceeds >= horizon_steps:
                    violated = True
                    break
            else:
                consecutive_exceeds = 0
        if violated:
            scores[proposal_idx] = 0.0
    return scores


def dac_continuous(ego_polygons: np.ndarray, scene: SceneContext) -> np.ndarray:
    dm = scene.drivable_area_map
    eligible = frozenset(
        (
            i
            for i, (tok, typ) in enumerate(zip(dm.tokens, dm.types))
            if tok in scene.route_lane_ids or typ in _DAC_STRUCTURAL_LAYERS
        )
    )
    batch_size, num_steps = ego_polygons.shape[:2]
    scores = np.ones(batch_size, dtype=np.float64)
    if dm._tree is None:
        return np.zeros(batch_size, dtype=np.float64)
    use_drivable_fallback = len(eligible) == 0
    drivable_union = dm.drivable_union if use_drivable_fallback else None
    for proposal_idx in range(batch_size):
        for time_idx in range(num_steps):
            ego_poly = ego_polygons[proposal_idx, time_idx]
            ego_area = float(ego_poly.area)
            if ego_area < 1e-09:
                continue
            if use_drivable_fallback:
                if drivable_union is None or drivable_union.is_empty:
                    scores[proposal_idx] = 0.0
                    break
                inside = float(ego_poly.intersection(drivable_union).area)
            else:
                nearby = dm._tree.query(ego_poly, predicate="intersects")
                local = [j for j in nearby.tolist() if j in eligible]
                if not local:
                    scores[proposal_idx] = 0.0
                    break
                local_union = unary_union([dm._polygons[j] for j in local])
                inside = float(ego_poly.intersection(local_union).area)
            coverage = min(inside / ego_area, 1.0)
            scores[proposal_idx] = min(scores[proposal_idx], coverage)
            if scores[proposal_idx] == 0.0:
                break
    return scores


def ddc_continuous(
    ego_coords: np.ndarray, ego_areas: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig
) -> np.ndarray:
    centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
    oncoming_progress = np.zeros(centers.shape[:2], dtype=np.float64)
    oncoming_progress[:, 1:] = np.linalg.norm(centers[:, 1:] - centers[:, :-1], axis=-1)
    not_oncoming = ~ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
    oncoming_progress[not_oncoming] = 0.0
    horizon_steps = max(
        int(round(rl_config.driving_direction_horizon / scene_geometry.dt(scene))), 1
    )
    rolling_progress = np.zeros_like(oncoming_progress)
    for time_idx in range(oncoming_progress.shape[1]):
        start_idx = max(0, time_idx - horizon_steps)
        rolling_progress[:, time_idx] = oncoming_progress[:, start_idx : time_idx + 1].sum(axis=1)
    max_progress = rolling_progress.max(axis=1)
    lo = rl_config.driving_direction_compliance_threshold
    hi = rl_config.driving_direction_violation_threshold
    if hi <= lo:
        return np.where(max_progress <= lo, 1.0, 0.0).astype(np.float64)
    return np.clip(1.0 - (max_progress - lo) / (hi - lo), 0.0, 1.0)


def tlc_continuous(
    ego_polygons: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig
) -> np.ndarray:
    batch_size = len(ego_polygons)
    min_distances = np.full(batch_size, np.inf, dtype=np.float64)
    margin = rl_config.tlc_margin
    for time_idx in range(ego_polygons.shape[1]):
        red_light_map = scene_geometry.get_red_light_map(scene, time_idx)
        if red_light_map is None:
            continue
        for proposal_idx in range(batch_size):
            if min_distances[proposal_idx] == 0.0:
                continue
            ego_poly = ego_polygons[proposal_idx, time_idx]
            if red_light_map.intersects(ego_poly)[0]:
                for token in red_light_map.get_colliding_tokens(ego_poly):
                    if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                        min_distances[proposal_idx] = 0.0
                        break
            else:
                buffered = ego_poly.buffer(margin)
                if red_light_map.intersects(buffered)[0]:
                    for token in red_light_map.get_colliding_tokens(buffered):
                        if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                            dist = ego_poly.distance(red_light_map[token])
                            min_distances[proposal_idx] = min(min_distances[proposal_idx], dist)
    min_distances = np.where(np.isinf(min_distances), margin * 2.0, min_distances)
    return np.clip(min_distances / margin, 0.0, 1.0)


def ep_continuous(
    ego_coords: np.ndarray,
    scene: SceneContext,
    rl_config: RLScorerConfig,
    *,
    reference_masked_progress: float | None = None,
) -> np.ndarray:
    """Continuous ego progress normalized by a caller-provided denominator.

    Fallback chain:
        1. reference_masked_progress param (typically official PDM masked progress)
        2. progress_distance_threshold (5m, no reference available)

    This helper no longer assigns any implicit official-reference meaning to GT.
    If a caller wants to analyze GT-normalized progress for diagnostics, it must
    resolve and pass that denominator explicitly as an analysis choice.
    """
    raw_progress = scene_geometry.progress(ego_coords, scene)
    denominator = reference_masked_progress
    if denominator is not None and denominator > rl_config.progress_distance_threshold:
        return np.clip(raw_progress / denominator, 0.0, 1.0)
    return np.clip(raw_progress / rl_config.progress_distance_threshold, 0.0, 1.0)


def lk_continuous(
    ego_coords: np.ndarray, ego_areas: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig
) -> np.ndarray:
    centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
    batch_size = len(ego_coords)
    scores = np.ones(batch_size, dtype=np.float64)
    for proposal_idx in range(batch_size):
        deviations = []
        for time_idx in range(ego_coords.shape[1]):
            if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                continue
            dev = Point(*centers[proposal_idx, time_idx]).distance(scene.centerline.linestring)
            deviations.append(dev)
        if not deviations:
            continue
        mean_dev = float(np.mean(deviations))
        lo = rl_config.lane_keeping_deviation
        hi = rl_config.lane_keeping_max_deviation
        if mean_dev <= lo:
            scores[proposal_idx] = 1.0
        elif mean_dev >= hi:
            scores[proposal_idx] = 0.0
        else:
            scores[proposal_idx] = 1.0 - (mean_dev - lo) / (hi - lo)
    return scores
