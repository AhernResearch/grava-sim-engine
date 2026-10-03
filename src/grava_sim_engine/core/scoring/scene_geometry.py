"""Scene geometry shared by the driving scorers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import shapely
from shapely.geometry import Point

from ..enums import DRIVABLE_LAYERS, BBCoordsIndex, EgoAreaIndex, SemanticMapLayer
from ..occupancy import (
    _LAYER_NAME_TO_ENUM,
    DrivableMap,
    OccupancyMap,
    _normalize_layer_name,
)
from ..types import SceneContext


def calculate_ego_areas(ego_coords: np.ndarray, scene: SceneContext) -> np.ndarray:
    batch_size, horizon, _, _ = ego_coords.shape
    centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
    corners = ego_coords[:, :, :4, :]
    ego_areas = np.zeros((batch_size, horizon, len(EgoAreaIndex)), dtype=bool)
    lane_membership = points_in_map_tokens(
        corners, scene.drivable_area_map, {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR}
    )
    if len(lane_membership) > 0:
        lane_corner_counts = lane_membership.sum(axis=-1)
        multiple_lane_mask = (lane_corner_counts > 0).sum(axis=0) > 1
        single_lane_mask = np.any(lane_corner_counts == 4, axis=0)
        ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES] = multiple_lane_mask & ~single_lane_mask
    center_membership = scene.drivable_area_map.points_in_polygons(centers)
    corner_membership = scene.drivable_area_map.points_in_polygons(corners)
    drivable_layers_all = list(DRIVABLE_LAYERS)
    corner_in_drivable = corner_membership[..., drivable_layers_all[0]]
    for layer in drivable_layers_all[1:]:
        corner_in_drivable = corner_in_drivable | corner_membership[..., layer]
    ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA] = ~corner_in_drivable.all(axis=-1)
    route_layers = points_in_route_lanes(centers, scene.drivable_area_map, scene.route_lane_ids)
    ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC] = ~route_layers
    ego_areas[:, :, EgoAreaIndex.IN_INTERSECTION] = center_membership[
        ..., SemanticMapLayer.INTERSECTION
    ]
    return ego_areas


def progress(ego_coords: np.ndarray, scene: SceneContext) -> np.ndarray:
    centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
    progress = np.zeros(len(ego_coords), dtype=np.float64)
    for proposal_idx in range(len(ego_coords)):
        start_progress = scene.centerline.project(Point(*centers[proposal_idx, 0]))
        end_progress = scene.centerline.project(Point(*centers[proposal_idx, -1]))
        progress[proposal_idx] = max(0.0, end_progress - start_progress)
    return progress


def points_in_map_tokens(
    points: np.ndarray, drivable_area_map: DrivableMap, include_layers: Iterable[SemanticMapLayer]
) -> np.ndarray:
    include_values = set(include_layers)
    selected_indices = [
        idx
        for idx, layer_name in enumerate(drivable_area_map.types)
        if _layer_name_to_enum(layer_name) in include_values
    ]
    if not selected_indices:
        return np.zeros((0, *points.shape[:-1]), dtype=bool)
    flat_points = points.reshape(-1, 2)
    membership = np.zeros((len(selected_indices), len(flat_points)), dtype=bool)
    x_coords = flat_points[:, 0]
    y_coords = flat_points[:, 1]
    for row_idx, polygon_idx in enumerate(selected_indices):
        membership[row_idx] = shapely.contains_xy(
            drivable_area_map[drivable_area_map.tokens[polygon_idx]], x_coords, y_coords
        )
    return membership.reshape((len(selected_indices), *points.shape[:-1]))


def points_in_route_lanes(
    points: np.ndarray, drivable_area_map: DrivableMap, route_lane_ids: Sequence[str]
) -> np.ndarray:
    route_lane_id_set = set(route_lane_ids)
    if not route_lane_id_set:
        return np.zeros(points.shape[:-1], dtype=bool)
    selected_indices = [
        idx
        for idx, (token, layer_name) in enumerate(
            zip(drivable_area_map.tokens, drivable_area_map.types)
        )
        if token in route_lane_id_set
        and _layer_name_to_enum(layer_name)
        in {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR}
    ]
    if not selected_indices:
        return np.zeros(points.shape[:-1], dtype=bool)
    flat_points = points.reshape(-1, 2)
    x_coords = flat_points[:, 0]
    y_coords = flat_points[:, 1]
    membership = np.zeros((len(selected_indices), len(flat_points)), dtype=bool)
    for row_idx, polygon_idx in enumerate(selected_indices):
        membership[row_idx] = shapely.contains_xy(
            drivable_area_map[drivable_area_map.tokens[polygon_idx]], x_coords, y_coords
        )
    return membership.any(axis=0).reshape(points.shape[:-1])


def dt(scene: SceneContext) -> float:
    return float(scene.observation.interval_time)


def local_time_idx(scene: SceneContext, time_idx: int) -> int:
    global_to_local = scene.observation.global_to_local_idcs
    clamped = min(max(time_idx, 0), len(global_to_local) - 1)
    return int(global_to_local[clamped])


def get_occupancy_map(scene: SceneContext, time_idx: int) -> OccupancyMap | None:
    return scene.observation.get_occupancy_map(local_time_idx(scene, time_idx))


def get_red_light_map(scene: SceneContext, time_idx: int) -> OccupancyMap | None:
    return scene.observation.get_red_light_map(local_time_idx(scene, time_idx))


def _layer_name_to_enum(layer_name: str) -> SemanticMapLayer | None:
    return _LAYER_NAME_TO_ENUM.get(_normalize_layer_name(layer_name))
