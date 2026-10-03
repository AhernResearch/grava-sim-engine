"""Road geometry shared by the driving scorers."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from ..enums import BBCoordsIndex, EgoAreaIndex, SemanticMapLayer, StateIndex
from ..geometry import state_to_coords
from ..types import SceneContext, VehicleParams
from . import scene_geometry
from .config import RLScorerConfig


def half_lane_width(scene: SceneContext) -> float:
    """Perpendicular distance from centerline to ego's lane polygon boundary.

    Uses a ray perpendicular to the centerline tangent at ego position to find
    the intersection with the lane polygon boundary. Falls back to shortest
    distance if the perpendicular ray does not intersect.
    """
    ego_x = scene.ego_state[StateIndex.X]
    ego_y = scene.ego_state[StateIndex.Y]
    ego_pt = Point(ego_x, ego_y)
    cl_ls = scene.centerline.linestring
    proj_dist = cl_ls.project(ego_pt)
    cl_pt = cl_ls.interpolate(proj_dist)
    eps = 0.1
    length = cl_ls.length
    p_before = cl_ls.interpolate(max(proj_dist - eps, 0.0))
    p_after = cl_ls.interpolate(min(proj_dist + eps, length))
    tx = p_after.x - p_before.x
    ty = p_after.y - p_before.y
    tn = max(np.hypot(tx, ty), 1e-09)
    nx, ny = (-ty / tn, tx / tn)
    lane_layers = {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR}
    dm = scene.drivable_area_map
    for token, layer_name in zip(dm.tokens, dm.types):
        if scene_geometry._layer_name_to_enum(layer_name) not in lane_layers:
            continue
        poly = dm[token]
        if not poly.contains(ego_pt):
            continue
        ray_len = 20.0
        ray_line = LineString(
            [
                (cl_pt.x - nx * ray_len, cl_pt.y - ny * ray_len),
                (cl_pt.x + nx * ray_len, cl_pt.y + ny * ray_len),
            ]
        )
        boundary_intersection = ray_line.intersection(poly.boundary)
        if not boundary_intersection.is_empty:
            half_w = cl_pt.distance(boundary_intersection)
            if half_w > 0.1:
                return max(half_w, 0.5)
        return max(cl_pt.distance(poly.boundary), 0.5)
    drivable = dm.drivable_union
    if drivable.is_empty:
        return 2.0
    return max(cl_pt.distance(drivable.boundary), 0.5)


def boundary_distance_series_raw(ego_coords: np.ndarray, scene: SceneContext) -> np.ndarray:
    """Per-step signed min distance from ego corners to drivable boundary. Shape: (B, T).

    Positive = inside drivable area (clearance to boundary).
    Negative = outside drivable area (penetration depth).
    Zero = exactly on boundary.
    """
    corners = ego_coords[:, :, :4, :]
    B, T = corners.shape[:2]
    drivable = scene.drivable_area_map.drivable_union
    if drivable.is_empty:
        return np.zeros((B, T), dtype=np.float64)
    boundary = drivable.boundary
    flat_points = shapely.points(corners.reshape(-1, 2))
    flat_inside = shapely.contains(drivable, flat_points)
    flat_dists = shapely.distance(flat_points, boundary)
    flat_dists[~flat_inside] = -flat_dists[~flat_inside]
    dists = flat_dists.reshape(B, T, 4).min(axis=2)
    return dists


def signed_lateral_offset_at(center_xy, centerline_ls) -> float:
    """Compute signed lateral offset. Positive = right of centerline."""
    ego_pt = Point(*center_xy)
    proj_dist = centerline_ls.project(ego_pt)
    cl_pt = centerline_ls.interpolate(proj_dist)
    dx = ego_pt.x - cl_pt.x
    dy = ego_pt.y - cl_pt.y
    eps = 0.1
    length = centerline_ls.length
    p_before = centerline_ls.interpolate(max(proj_dist - eps, 0.0))
    p_after = centerline_ls.interpolate(min(proj_dist + eps, length))
    tangent_x = p_after.x - p_before.x
    tangent_y = p_after.y - p_before.y
    cross = tangent_x * dy - tangent_y * dx
    norm = max(np.hypot(tangent_x, tangent_y), 1e-09)
    return -cross / norm


def lateral_offset_signed(
    ego_coords: np.ndarray, ego_areas: np.ndarray, scene: SceneContext
) -> np.ndarray:
    """Mean signed lateral offset from centerline per proposal. Shape: (batch,)"""
    centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
    batch_size = len(ego_coords)
    result = np.zeros(batch_size, dtype=np.float64)
    cl_ls = scene.centerline.linestring
    for proposal_idx in range(batch_size):
        offsets = []
        for time_idx in range(ego_coords.shape[1]):
            if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                continue
            offsets.append(signed_lateral_offset_at(centers[proposal_idx, time_idx], cl_ls))
        if offsets:
            result[proposal_idx] = float(np.mean(offsets))
    return result


def lateral_offset_change(
    ego_coords: np.ndarray, ego_areas: np.ndarray, scene: SceneContext
) -> np.ndarray:
    """Signed offset at last non-intersection timestep minus first. Shape: (batch,)"""
    centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
    batch_size = len(ego_coords)
    result = np.zeros(batch_size, dtype=np.float64)
    cl_ls = scene.centerline.linestring
    for proposal_idx in range(batch_size):
        first_offset = None
        last_offset = None
        for time_idx in range(ego_coords.shape[1]):
            if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                continue
            offset = signed_lateral_offset_at(centers[proposal_idx, time_idx], cl_ls)
            if first_offset is None:
                first_offset = offset
            last_offset = offset
        if first_offset is not None and last_offset is not None:
            result[proposal_idx] = last_offset - first_offset
    return result


def centerline_geometry(
    ego_coords: np.ndarray,
    ego_areas: np.ndarray,
    scene: SceneContext,
    *,
    include_details: bool = False,
) -> dict:
    """Centerline-relative geometry diagnostics.

    Note: start_signed/end_signed are the signed lateral offsets at the
    first/last NON-INTERSECTION timestep, not the trajectory start/end.
    Intersection steps are excluded because centerline distance is ill-defined there.
    """
    centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
    batch_size, num_steps = centers.shape[:2]
    cl_ls = scene.centerline.linestring
    start_signed = np.zeros(batch_size, dtype=np.float64)
    end_signed = np.zeros(batch_size, dtype=np.float64)
    mean_dist = np.zeros(batch_size, dtype=np.float64)
    max_dist = np.zeros(batch_size, dtype=np.float64)
    local_points = sample_local_centerline_points(scene) if include_details else []
    for proposal_idx in range(batch_size):
        signed_offsets = []
        abs_dists = []
        for time_idx in range(num_steps):
            if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                continue
            center_xy = centers[proposal_idx, time_idx]
            signed = signed_lateral_offset_at(center_xy, cl_ls)
            signed_offsets.append(signed)
            abs_dists.append(Point(*center_xy).distance(cl_ls))
        if signed_offsets:
            start_signed[proposal_idx] = float(signed_offsets[0])
            end_signed[proposal_idx] = float(signed_offsets[-1])
            mean_dist[proposal_idx] = float(np.mean(abs_dists))
            max_dist[proposal_idx] = float(np.max(abs_dists))
    return {
        "start_signed": start_signed,
        "end_signed": end_signed,
        "mean_distance": mean_dist,
        "max_distance": max_dist,
        "local_centerline_points": local_points,
    }


def sample_local_centerline_points(
    scene: SceneContext, num_points: int = 6, horizon_m: float = 30.0
) -> list[list[float]]:
    cl = scene.centerline
    ego_x = float(scene.ego_state[StateIndex.X])
    ego_y = float(scene.ego_state[StateIndex.Y])
    ego_h = float(scene.ego_state[StateIndex.HEADING])
    ego_pt = Point(ego_x, ego_y)
    start_progress = cl.project(ego_pt)
    sample_ds = np.linspace(start_progress, min(start_progress + horizon_m, cl.length), num_points)
    cos_h = np.cos(-ego_h)
    sin_h = np.sin(-ego_h)
    points = []
    for d in sample_ds:
        p = cl.interpolate(float(d))
        dx = float(p.x) - ego_x
        dy = float(p.y) - ego_y
        local_x = dx * cos_h - dy * sin_h
        local_y = dx * sin_h + dy * cos_h
        points.append([local_x, local_y])
    return points


def boundary_geometry(
    ego_coords: np.ndarray,
    scene: SceneContext,
    rl_config: RLScorerConfig,
    *,
    dists: np.ndarray | None = None,
    vehicle: VehicleParams,
) -> dict:
    if dists is None:
        dists = boundary_distance_series_raw(ego_coords, scene)
    centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
    sides: list[str | None] = []
    nearest_sides: list[str | None] = []
    nearest_dists: list[float] = []
    cl_ls = scene.centerline.linestring
    drivable = scene.drivable_area_map.drivable_union
    boundary = drivable.boundary if not drivable.is_empty else None
    ego_pos = Point(*scene.ego_state[:2])
    B, T = dists.shape
    sides_series: list[list[str | None]] = [[None] * T for _ in range(B)]
    for pi in range(B):
        for ti in range(T):
            if dists[pi, ti] < 0.0:
                signed = signed_lateral_offset_at(centers[pi, ti], cl_ls)
                sides_series[pi][ti] = "right" if signed > 0 else "left"
    for proposal_idx in range(len(ego_coords)):
        closest_idx = int(np.argmin(dists[proposal_idx]))
        center_xy = centers[proposal_idx, closest_idx]
        signed = signed_lateral_offset_at(center_xy, cl_ls)
        if signed > 0:
            sides.append("right")
        elif signed < 0:
            sides.append("left")
        else:
            sides.append(None)
        if boundary is not None:
            nearest_pt = nearest_points(ego_pos, boundary)[1]
            ego_state_arr = scene.ego_state.reshape(1, 1, -1)
            ego_now_coords = state_to_coords(ego_state_arr, vehicle)
            ego_now_corners = ego_now_coords[0, 0, :4, :]
            corner_pts = shapely.points(ego_now_corners)
            corner_dists = shapely.distance(corner_pts, boundary)
            nearest_dists.append(float(corner_dists.min()))
            nearest_signed = signed_lateral_offset_at(np.array([nearest_pt.x, nearest_pt.y]), cl_ls)
            if nearest_signed > 0:
                nearest_sides.append("right")
            elif nearest_signed < 0:
                nearest_sides.append("left")
            else:
                nearest_sides.append(None)
        else:
            nearest_sides.append(None)
            nearest_dists.append(0.0)
    return {
        "distances": dists,
        "start": dists[:, 0],
        "end": dists[:, -1],
        "min": dists.min(axis=1),
        "mean": dists.mean(axis=1),
        "side": sides,
        "nearest_side": nearest_sides,
        "nearest_distance": nearest_dists,
        "sides_series": sides_series,
    }


def topology_occupancy(ego_areas: np.ndarray) -> dict:
    in_intersection = ego_areas[:, :, EgoAreaIndex.IN_INTERSECTION].astype(bool)
    oncoming = ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC].astype(bool)
    non_drivable = ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].astype(bool)
    multiple_lanes = ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES].astype(bool)
    n_steps = ego_areas.shape[1]
    if n_steps == 0:
        batch_size = ego_areas.shape[0]
        zeros = np.zeros(batch_size, dtype=np.float64)
        return {
            "in_intersection_flags": in_intersection,
            "oncoming_flags": oncoming,
            "non_drivable_flags": non_drivable,
            "multiple_lanes_flags": multiple_lanes,
            "in_intersection_fraction": zeros,
            "oncoming_fraction": zeros,
            "non_drivable_fraction": zeros,
            "multiple_lanes_fraction": zeros,
        }
    return {
        "in_intersection_flags": in_intersection,
        "oncoming_flags": oncoming,
        "non_drivable_flags": non_drivable,
        "multiple_lanes_flags": multiple_lanes,
        "in_intersection_fraction": in_intersection.mean(axis=1),
        "oncoming_fraction": oncoming.mean(axis=1),
        "non_drivable_fraction": non_drivable.mean(axis=1),
        "multiple_lanes_fraction": multiple_lanes.mean(axis=1),
    }
