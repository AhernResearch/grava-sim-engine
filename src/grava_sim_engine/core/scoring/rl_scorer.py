from __future__ import annotations

from typing import List

import numpy as np

from ..enums import SemanticMapLayer
from ..geometry import calculate_progress, coords_to_polygons, state_to_coords
from ..types import RLScoringResult, SceneContext
from . import (
    collision_metrics,
    comfort_metrics,
    reference,
    road_geometry,
    road_metrics,
    scene_geometry,
)
from . import trajectory as trajectory_ops
from .base import ScorerBase
from .config import RLScorerConfig


class RLScorer(ScorerBase):
    """RL reward scorer with continuous and discrete modes.

    Continuous mode: independent continuous sub-rewards with soft safety gating.
    Discrete mode: all sub-metrics aligned with PDMSScorer (binary NC/TTC/HC,
    v1 progress normalization) for exact PDMS compatibility.

    Aggregation: safety_gate ^ alpha × weighted_avg(EP, TTC, LK, HC)
    where safety_gate = NC × DAC × DDC × TLC
    """

    def score(
        self,
        waypoints_xy: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig | None = None,
        *,
        include_details: bool = False,
    ) -> RLScoringResult:
        return self.score_batch(
            waypoints_xy[None, ...], scene, rl_config, include_details=include_details
        )[0]

    def _pdm_per_step_progress(self, scene: SceneContext) -> np.ndarray | None:
        """PDM per-step centerline arc-lengths, shape (1, T)."""
        if scene.pdm_trajectory is None:
            return None
        coords, _ = reference.simulate_reference(
            scene.pdm_trajectory, scene, simulator=self._simulator, vehicle=self._vehicle
        )
        return calculate_progress(coords, scene.centerline)

    def _progress_per_waypoint(
        self, ego_coords: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig
    ) -> List[List[float]]:
        """8 values per proposal: pred_cumulative[i] / pdm_cumulative[i] at each waypoint."""
        pred_projected = calculate_progress(ego_coords, scene.centerline)
        pdm_projected = self._pdm_per_step_progress(scene)
        all_wp = [5, 10, 15, 20, 25, 30, 35, 40]
        wp_indices = all_wp
        results = []
        for b in range(len(ego_coords)):
            start = float(pred_projected[b, 0])
            pred_cum = [max(0.0, float(pred_projected[b, idx]) - start) for idx in wp_indices]
            if pdm_projected is not None:
                pdm_start = float(pdm_projected[0, 0])
                pdm_cum = [max(0.0, float(pdm_projected[0, idx]) - pdm_start) for idx in wp_indices]
                normalized = []
                for i in range(len(wp_indices)):
                    denom = pdm_cum[i]
                    if denom < 0.1:
                        normalized.append(1.0 if abs(pred_cum[i]) < 0.1 else 0.0)
                    else:
                        normalized.append(float(np.clip(pred_cum[i] / denom, 0.0, 1.0)))
            else:
                threshold = rl_config.progress_distance_threshold
                normalized = [float(np.clip(p / threshold, 0.0, 1.0)) for p in pred_cum]
            results.append(normalized)
        return results

    def score_batch(
        self,
        trajectories_xy: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig | None = None,
        *,
        include_details: bool = False,
    ) -> List[RLScoringResult]:
        rl_config = rl_config or RLScorerConfig()
        batch_waypoints = trajectory_ops.coerce_trajectories(trajectories_xy)
        proposals = trajectory_ops.build_proposals(batch_waypoints, scene)
        simulated_states = self._simulator.simulate_proposals(
            ego_state=scene.ego_state, proposals=proposals, observation=scene.observation
        )
        ego_coords = state_to_coords(simulated_states, self._vehicle)
        ego_polygons = coords_to_polygons(ego_coords)
        ego_areas = scene_geometry.calculate_ego_areas(ego_coords, scene)
        if rl_config.safety_mode == "continuous":
            collision_result = collision_metrics.collision_metrics(
                simulated_states, ego_polygons, ego_areas, scene, include_details=include_details
            )
            nc = collision_result["nc"]
            max_collision_overlap = collision_result["max_collision_overlap"]
            max_collision_penetration_distance = collision_result[
                "max_collision_penetration_distance"
            ]
            per_step_collisions = collision_result["per_step_collisions"]
            dac = road_metrics.dac_continuous(ego_polygons, scene)
            ddc = (
                road_metrics.ddc_continuous(ego_coords, ego_areas, scene, rl_config)
                if rl_config.ddc_weight > 0.0
                else np.ones(len(batch_waypoints), dtype=np.float64)
            )
            tlc = (
                road_metrics.tlc_continuous(ego_polygons, scene, rl_config)
                if rl_config.tlc_weight > 0.0
                else np.ones(len(batch_waypoints), dtype=np.float64)
            )
            obstacle_dist_series = collision_metrics.obstacle_distance_series(
                ego_polygons, scene, rl_config
            )
            min_obstacle_dist = obstacle_dist_series.min(axis=1)
            obstacle_margin = rl_config.obstacle_clearance_margin
            obstacle_valid = obstacle_dist_series < obstacle_margin
            obstacle_counts = obstacle_valid.sum(axis=1)
            mean_obstacle_dist_5m = np.divide(
                np.where(obstacle_valid, obstacle_dist_series, 0.0).sum(axis=1),
                obstacle_counts,
                out=np.full(len(batch_waypoints), obstacle_margin, dtype=np.float64),
                where=obstacle_counts > 0,
            )
            boundary_dist_series = road_geometry.boundary_distance_series_raw(ego_coords, scene)
            half_lane_w = road_geometry.half_lane_width(scene)
        else:
            nc = collision_metrics.no_at_fault_collision(
                simulated_states, ego_polygons, ego_areas, scene
            )
            dac = road_metrics.drivable_area_compliance(ego_areas)
            max_collision_overlap = np.where(nc < 1.0, 1.0 - nc, 0.0)
            max_collision_penetration_distance = np.zeros(len(batch_waypoints), dtype=np.float64)
            per_step_collisions = [
                [None] * ego_polygons.shape[1] if include_details else []
                for _ in range(len(batch_waypoints))
            ]
            ddc = road_metrics.driving_direction_compliance(ego_coords, ego_areas, scene)
            tlc = road_metrics.traffic_light_compliance(ego_polygons, scene)
            min_obstacle_dist = np.zeros(len(batch_waypoints), dtype=np.float64)
            mean_obstacle_dist_5m = np.full(len(batch_waypoints), 5.0, dtype=np.float64)
            boundary_dist_series = None
            half_lane_w = 2.0
        pdm_masked = reference.pdm_masked_progress(
            scene, simulator=self._simulator, vehicle=self._vehicle
        )
        if rl_config.safety_mode == "discrete":
            progress_raw = scene_geometry.progress(ego_coords, scene)
            ep = reference.normalize_progress(progress_raw, nc, dac, pdm_masked)
            ttc = collision_metrics.time_to_collision(
                simulated_states, ego_coords, ego_areas, scene
            )
            hc = comfort_metrics.history_comfort(simulated_states, scene, use_past_states=False)
        else:
            ep = road_metrics.ep_continuous(
                ego_coords, scene, rl_config, reference_masked_progress=pdm_masked
            )
            ttc = collision_metrics.ttc_continuous(
                simulated_states, ego_coords, ego_areas, scene, rl_config
            )
            hc = comfort_metrics.hc_continuous(simulated_states, scene)
        if include_details:
            progress_per_wp = self._progress_per_waypoint(ego_coords, scene, rl_config)
        lk = (
            road_metrics.lk_continuous(ego_coords, ego_areas, scene, rl_config)
            if rl_config.lk_weight > 0.0
            else np.ones(len(batch_waypoints), dtype=np.float64)
        )
        if rl_config.safety_mode == "discrete":
            pdms_nc = nc
            pdms_dac = dac
            pdms_ddc = ddc
            pdms_tlc = tlc
            pdms_ep = ep
            pdms_ttc = ttc
            pdms_hc = hc
        else:
            pdms_nc = collision_metrics.no_at_fault_collision(
                simulated_states, ego_polygons, ego_areas, scene
            )
            pdms_dac = road_metrics.drivable_area_compliance(ego_areas)
            pdms_ddc = road_metrics.driving_direction_compliance(ego_coords, ego_areas, scene)
            pdms_tlc = road_metrics.traffic_light_compliance(ego_polygons, scene)
            pdms_progress_raw = scene_geometry.progress(ego_coords, scene)
            pdms_ep = reference.normalize_progress(pdms_progress_raw, pdms_nc, pdms_dac, pdm_masked)
            pdms_ttc = collision_metrics.time_to_collision(
                simulated_states, ego_coords, ego_areas, scene
            )
            pdms_hc = comfort_metrics.history_comfort(
                simulated_states, scene, use_past_states=False
            )
        pdms_lk = road_metrics.lane_keeping(ego_coords, ego_areas, scene)
        pdms_safety_gate = pdms_nc * pdms_dac
        pdms_perf = (5.0 * pdms_ep + 5.0 * pdms_ttc + 2.0 * pdms_hc) / 12.0
        pdm_scores = pdms_safety_gate * pdms_perf
        lat_offset_signed = road_geometry.lateral_offset_signed(ego_coords, ego_areas, scene)
        lat_offset_change = road_geometry.lateral_offset_change(ego_coords, ego_areas, scene)
        centerline_geom = road_geometry.centerline_geometry(
            ego_coords, ego_areas, scene, include_details=include_details
        )
        boundary_geom = road_geometry.boundary_geometry(
            ego_coords, scene, rl_config, dists=boundary_dist_series, vehicle=self._vehicle
        )
        topology_geom = road_geometry.topology_occupancy(ego_areas)
        ego_pos_global = scene.ego_state[:2].reshape(1, 1, 2)
        ego_now_membership = scene.drivable_area_map.points_in_polygons(ego_pos_global)
        in_intersection_now = bool(ego_now_membership[0, 0, SemanticMapLayer.INTERSECTION])
        if rl_config.safety_mode == "discrete":
            progress_meters = scene_geometry.progress(ego_coords, scene)
            safety_gate_arr = nc * dac
            gated_progress = progress_meters * safety_gate_arr * ttc * hc
        safety_product = nc * dac * ddc * tlc
        alpha = rl_config.safety_gate_alpha
        safety_gate = np.power(np.clip(safety_product, 0.0, 1.0), alpha)
        perf_metrics = np.stack([ep, ttc, lk, hc], axis=1)
        perf_weights = rl_config.performance_weights
        perf_sum = perf_weights.sum()
        if perf_sum > 0:
            performance = (perf_metrics * perf_weights[None, :]).sum(axis=1) / perf_sum
        else:
            performance = np.ones(len(batch_waypoints), dtype=np.float64)
        rl_scores = safety_gate * performance
        results: List[RLScoringResult] = []
        for i in range(len(batch_waypoints)):
            result_kwargs = dict(
                rl_score=float(rl_scores[i]),
                no_at_fault_collisions=float(nc[i]),
                drivable_area_compliance=float(dac[i]),
                driving_direction_compliance=float(ddc[i]),
                traffic_light_compliance=float(tlc[i]),
                ego_progress=float(ep[i]),
                time_to_collision=float(ttc[i]),
                lane_keeping=float(lk[i]),
                history_comfort=float(hc[i]),
                max_collision_overlap=float(max_collision_overlap[i]),
                max_collision_penetration_distance=float(max_collision_penetration_distance[i]),
                min_obstacle_distance=float(min_obstacle_dist[i]),
                min_boundary_distance=float(boundary_geom["min"][i]),
                mean_obstacle_distance_5m=float(mean_obstacle_dist_5m[i]),
                half_lane_width=float(half_lane_w),
                lateral_offset_signed=float(lat_offset_signed[i]),
                lateral_offset_change=float(lat_offset_change[i]),
                centerline_lateral_offset_start_signed=float(centerline_geom["start_signed"][i]),
                centerline_lateral_offset_end_signed=float(centerline_geom["end_signed"][i]),
                centerline_distance_mean=float(centerline_geom["mean_distance"][i]),
                centerline_distance_max=float(centerline_geom["max_distance"][i]),
                boundary_distance_start=float(boundary_geom["start"][i]),
                boundary_distance_end=float(boundary_geom["end"][i]),
                boundary_distance_mean=float(boundary_geom["mean"][i]),
                boundary_side=boundary_geom["side"][i],
                nearest_boundary_side=boundary_geom["nearest_side"][i],
                nearest_boundary_distance=float(boundary_geom["nearest_distance"][i]),
                in_intersection_fraction=float(topology_geom["in_intersection_fraction"][i]),
                in_intersection_now=in_intersection_now,
                oncoming_fraction=float(topology_geom["oncoming_fraction"][i]),
                non_drivable_fraction=float(topology_geom["non_drivable_fraction"][i]),
                multiple_lanes_fraction=float(topology_geom["multiple_lanes_fraction"][i]),
            )
            if include_details:
                result_kwargs.update(
                    local_centerline_points=centerline_geom["local_centerline_points"],
                    boundary_distances=[float(x) for x in boundary_geom["distances"][i].tolist()],
                    in_intersection_flags=[
                        bool(x) for x in topology_geom["in_intersection_flags"][i].tolist()
                    ],
                    oncoming_flags=[bool(x) for x in topology_geom["oncoming_flags"][i].tolist()],
                    non_drivable_flags=[
                        bool(x) for x in topology_geom["non_drivable_flags"][i].tolist()
                    ],
                    multiple_lanes_flags=[
                        bool(x) for x in topology_geom["multiple_lanes_flags"][i].tolist()
                    ],
                    boundary_sides=boundary_geom["sides_series"][i],
                    collision_per_step=per_step_collisions[i],
                    progress_per_waypoint=progress_per_wp[i],
                )
            if rl_config.safety_mode == "discrete":
                result_kwargs.update(
                    safety_gate=float(safety_gate_arr[i]), gated_progress=float(gated_progress[i])
                )
            result_kwargs.update(
                pdm_score=float(pdm_scores[i]),
                pdms_metrics={
                    "no_at_fault_collisions": float(pdms_nc[i]),
                    "drivable_area_compliance": float(pdms_dac[i]),
                    "driving_direction_compliance": float(pdms_ddc[i]),
                    "traffic_light_compliance": float(pdms_tlc[i]),
                    "ego_progress": float(pdms_ep[i]),
                    "time_to_collision": float(pdms_ttc[i]),
                    "lane_keeping": float(pdms_lk[i]),
                    "history_comfort": float(pdms_hc[i]),
                },
            )
            results.append(RLScoringResult(**result_kwargs))
        return results
