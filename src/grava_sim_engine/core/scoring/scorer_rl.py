from __future__ import annotations

from typing import List

import numpy as np

from ..enums import MultiMetricIndex, SemanticMapLayer
from ..geometry import state_to_coords, coords_to_polygons, calculate_progress
from ..types import SceneContext, RLScoringResult
from .base import ScorerBase, RLScorerConfig


class RLScorer(ScorerBase):
    """RL reward scorer with continuous and discrete modes.

    Continuous mode: independent continuous sub-rewards with soft safety gating.
    Discrete mode: all sub-metrics aligned with PDMScorerV1 (binary NC/TTC/HC,
    v1 progress normalization) for exact PDMS compatibility.

    Aggregation: safety_gate ^ alpha × weighted_avg(EP, TTC, LK, HC)
    where safety_gate = NC × DAC × DDC × TLC
    """

    # V1 progress distance threshold (must match PDMScorerV1)
    _PROGRESS_DISTANCE_THRESHOLD = 5.0

    def _normalize_progress_v1_inline(
        self,
        progress_raw: np.ndarray,
        nc: np.ndarray,
        dac: np.ndarray,
        pdm_masked_progress: float | None,
        *,
        independent_fallback: bool = False,
    ) -> np.ndarray:
        """V1-aligned progress normalization for discrete RL mode.

        Replicates PDMScorerV1._normalize_progress_v1 exactly:
        masked_progress = raw * NC * DAC, then normalize by
        max(masked_pred, masked_pdm) when PDM reference exists.
        """
        multi_prod = nc * dac
        masked_progress = progress_raw * multi_prod

        if pdm_masked_progress is not None and pdm_masked_progress > self._PROGRESS_DISTANCE_THRESHOLD:
            denominator = np.maximum(masked_progress, pdm_masked_progress)
            return np.divide(
                masked_progress,
                denominator,
                out=np.zeros_like(masked_progress, dtype=np.float64),
                where=denominator > 0.0,
            )

        if independent_fallback:
            normalized = np.ones_like(progress_raw, dtype=np.float64)
            normalized[multi_prod == 0.0] = 0.0
            return normalized

        max_masked = float(masked_progress.max()) if len(masked_progress) > 0 else 0.0
        if max_masked > self._PROGRESS_DISTANCE_THRESHOLD:
            return np.clip(masked_progress / max_masked, 0.0, 1.0)

        normalized = np.ones_like(progress_raw, dtype=np.float64)
        normalized[multi_prod == 0.0] = 0.0
        return normalized

    def score(
        self,
        waypoints_xy: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig | None = None,
    ) -> RLScoringResult:
        return self.score_batch(waypoints_xy[None, ...], scene, rl_config)[0]

    def _pdm_per_step_progress(self, scene: SceneContext) -> np.ndarray | None:
        """PDM per-step centerline arc-lengths, shape (1, T)."""
        if scene.pdm_trajectory is None:
            return None
        reference_waypoints = scene.pdm_trajectory[None, ...]
        try:
            proposals = self._build_proposals(reference_waypoints, scene)
            simulated = self._simulator.simulate_proposals(
                ego_state=scene.ego_state, proposals=proposals, observation=scene.observation,
            )
            coords = state_to_coords(simulated, self._vehicle)
            return calculate_progress(coords, scene.centerline)  # (1, T)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to compute PDM per-step progress for scene {scene.log_name}/{scene.scene_token}"
            ) from exc

    def _progress_per_waypoint(
        self, ego_coords: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> List[List[float]]:
        """8 values per proposal: pred_cumulative[i] / pdm_cumulative[i] at each waypoint."""
        pred_projected = calculate_progress(ego_coords, scene.centerline)  # (B, T)
        pdm_projected = self._pdm_per_step_progress(scene)  # (1, T) or None

        # Waypoint sample indices: step 5,10,...,40 (clamped to available timesteps)
        num_steps = pred_projected.shape[1]
        all_wp = [5, 10, 15, 20, 25, 30, 35, 40]
        wp_indices = [i for i in all_wp if i < num_steps]

        results = []
        for b in range(len(ego_coords)):
            start = float(pred_projected[b, 0])
            pred_cum = [max(0.0, float(pred_projected[b, idx]) - start) for idx in wp_indices]

            if pdm_projected is not None:
                pdm_start = float(pdm_projected[0, 0])
                pdm_cum = [max(0.0, float(pdm_projected[0, min(idx, pdm_projected.shape[1] - 1)]) - pdm_start) for idx in wp_indices]
                normalized = []
                for i in range(len(wp_indices)):
                    denom = pdm_cum[i]
                    if denom < 0.1:  # PDM cumulative < 10cm = truly stopped
                        normalized.append(1.0 if abs(pred_cum[i]) < 0.1 else 0.0)
                    else:
                        normalized.append(float(np.clip(pred_cum[i] / denom, 0.0, 1.0)))
            else:
                threshold = rl_config.progress_distance_threshold
                normalized = [float(np.clip(p / threshold, 0.0, 1.0)) for p in pred_cum]

            # Pad to 8 values if trajectory is shorter than expected
            while len(normalized) < 8:
                normalized.append(normalized[-1] if normalized else 0.0)
            results.append(normalized)
        return results

    def _resolve_pdm_masked_progress(self, scene: SceneContext) -> float | None:
        """Resolve RL EP denominator from official PDM reference context.

        Preference order:
            1. scene.pdm_masked_progress (precomputed official reference)
            2. Online PDM simulation from scene.pdm_trajectory
            3. None (caller falls back to threshold)

        GT fields may still be attached for open-loop analysis, diagnostics, or
        optional debug, but they no longer define RL's main reference semantics.
        """
        if scene.pdm_masked_progress is not None:
            return scene.pdm_masked_progress

        pdm_result = self._simulate_and_score_pdm(
            scene,
            multi_indices=[MultiMetricIndex.NO_COLLISION, MultiMetricIndex.DRIVABLE_AREA],
            weighted_indices=[],
        )
        if pdm_result is None:
            return None

        pdm_nc = float(pdm_result.multi_metrics[MultiMetricIndex.NO_COLLISION])
        pdm_dac = float(pdm_result.multi_metrics[MultiMetricIndex.DRIVABLE_AREA])
        return float(pdm_result.progress * pdm_nc * pdm_dac)

    def score_batch(
        self,
        trajectories_xy: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig | None = None,
        *,
        include_ego: bool = False,
        independent_progress_fallback: bool = False,
    ) -> List[RLScoringResult]:
        rl_config = rl_config or RLScorerConfig.v1()

        batch_waypoints = self._coerce_trajectories(trajectories_xy)
        proposals = self._build_proposals(batch_waypoints, scene, include_ego=include_ego)
        simulated_states = self._simulator.simulate_proposals(
            ego_state=scene.ego_state, proposals=proposals, observation=scene.observation,
        )
        ego_coords = state_to_coords(simulated_states, self._vehicle)
        ego_polygons = coords_to_polygons(ego_coords)
        ego_areas = self._calculate_ego_areas(ego_coords, scene)

        # Safety layer: continuous or discrete
        if rl_config.safety_mode == "continuous":
            collision_result = self._collision_metrics(simulated_states, ego_polygons, ego_areas, scene, rl_config)
            nc = collision_result["nc"]
            max_collision_overlap = collision_result["max_collision_overlap"]
            max_collision_penetration_distance = collision_result["max_collision_penetration_distance"]
            per_step_collisions = collision_result["per_step_collisions"]
            dac = self._dac_continuous(ego_polygons, scene, rl_config)
            ddc = (
                self._ddc_continuous(ego_coords, ego_areas, scene, rl_config)
                if rl_config.ddc_weight > 0.0
                else np.ones(len(batch_waypoints), dtype=np.float64)
            )
            tlc = (
                self._tlc_continuous(ego_polygons, scene, rl_config)
                if rl_config.tlc_weight > 0.0
                else np.ones(len(batch_waypoints), dtype=np.float64)
            )
            # Raw physics: obstacle and boundary distances + normalization denominators
            obstacle_dist_series = self._obstacle_distance_series(ego_polygons, scene, rl_config)
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
            boundary_dist_series = self._boundary_distance_series_raw(ego_coords, scene)
            min_boundary_dist = boundary_dist_series.min(axis=1)
            half_lane_w = self._half_lane_width(scene)
        else:
            # Discrete mode: V1-aligned binary metrics
            nc = self._no_at_fault_collision(
                simulated_states, ego_polygons, ego_areas, scene, use_observation_types=True,
            )
            dac = self._drivable_area_compliance(ego_areas)
            max_collision_overlap = np.where(nc < 1.0, 1.0 - nc, 0.0)
            max_collision_penetration_distance = np.zeros(len(batch_waypoints), dtype=np.float64)
            per_step_collisions = [[None] * ego_polygons.shape[1] for _ in range(len(batch_waypoints))]
            # V1 computes DDC unconditionally (even though weight=0 in aggregation)
            ddc = self._driving_direction_compliance(ego_coords, ego_areas, scene)
            # V1 computes TLC unconditionally
            tlc = self._traffic_light_compliance(ego_polygons, scene)
            min_obstacle_dist = np.zeros(len(batch_waypoints), dtype=np.float64)
            min_boundary_dist = np.zeros(len(batch_waypoints), dtype=np.float64)
            mean_obstacle_dist_5m = np.full(len(batch_waypoints), 5.0, dtype=np.float64)
            boundary_dist_series = None
            half_lane_w = 2.0

        # Performance layer
        pdm_masked = self._resolve_pdm_masked_progress(scene)
        if rl_config.safety_mode == "discrete":
            # V1-aligned: v1 progress normalization, binary TTC (1s), binary HC
            progress_raw = self._progress(ego_coords, scene)
            ep = self._normalize_progress_v1_inline(
                progress_raw,
                nc,
                dac,
                pdm_masked,
                independent_fallback=independent_progress_fallback,
            )
            ttc = self._time_to_collision(simulated_states, ego_coords, ego_areas, scene)
            hc = self._history_comfort(simulated_states, scene, use_past_states=False)
        else:
            # Continuous: independent continuous sub-rewards
            ep = self._ep_continuous(
                ego_coords,
                scene,
                rl_config,
                reference_masked_progress=pdm_masked,
            )
            ttc = self._ttc_continuous(simulated_states, ego_coords, ego_areas, scene, rl_config)
            hc = self._hc_continuous(simulated_states, scene)
        progress_per_wp = self._progress_per_waypoint(ego_coords, scene, rl_config)
        lk = (
            self._lk_continuous(ego_coords, ego_areas, scene, rl_config)
            if rl_config.lk_weight > 0.0
            else np.ones(len(batch_waypoints), dtype=np.float64)
        )

        # Standard v1 PDMS monitoring computed from the same simulated states.
        # This keeps /v1/score/rl useful for one-pass margin filtering: the
        # top-level RL fields may be continuous rewards, while pdms_* fields are
        # the discrete NavSim-compatible score and components.
        if rl_config.safety_mode == "discrete":
            pdms_nc = nc
            pdms_dac = dac
            pdms_ddc = ddc
            pdms_tlc = tlc
            pdms_ep = ep
            pdms_ttc = ttc
            pdms_hc = hc
        else:
            pdms_nc = self._no_at_fault_collision(
                simulated_states, ego_polygons, ego_areas, scene, use_observation_types=True,
            )
            pdms_dac = self._drivable_area_compliance(ego_areas)
            pdms_ddc = self._driving_direction_compliance(ego_coords, ego_areas, scene)
            pdms_tlc = self._traffic_light_compliance(ego_polygons, scene)
            pdms_progress_raw = self._progress(ego_coords, scene)
            pdms_ep = self._normalize_progress_v1_inline(
                pdms_progress_raw,
                pdms_nc,
                pdms_dac,
                pdm_masked,
                independent_fallback=independent_progress_fallback,
            )
            pdms_ttc = self._time_to_collision(simulated_states, ego_coords, ego_areas, scene)
            pdms_hc = self._history_comfort(simulated_states, scene, use_past_states=False)
        pdms_lk = self._lane_keeping(ego_coords, ego_areas, scene)
        pdms_safety_gate = pdms_nc * pdms_dac
        pdms_perf = (5.0 * pdms_ep + 5.0 * pdms_ttc + 2.0 * pdms_hc) / 12.0
        pdms_score_arr = pdms_safety_gate * pdms_perf

        lat_offset_signed = self._lateral_offset_signed(ego_coords, ego_areas, scene)
        lat_offset_change = self._lateral_offset_change(ego_coords, ego_areas, scene)
        centerline_geom = self._centerline_geometry(ego_coords, ego_areas, scene)
        boundary_geom = self._boundary_geometry(ego_coords, scene, rl_config, dists=boundary_dist_series)
        topology_geom = self._topology_occupancy(ego_areas)

        # Current-frame intersection check using ego_state global position
        ego_pos_global = scene.ego_state[:2].reshape(1, 1, 2)  # (1, 1, 2)
        ego_now_membership = scene.drivable_area_map.points_in_polygons(ego_pos_global)
        in_intersection_now = bool(ego_now_membership[0, 0, SemanticMapLayer.INTERSECTION])

        # Discrete mode: raw progress (meters) + safety gate + V1-aligned PDMS
        if rl_config.safety_mode == "discrete":
            raw_progress_meters = self._progress(ego_coords, scene)
            # V1 safety gate: NC × DAC only
            safety_gate_arr = nc * dac
            # Quality-modulated progress: raw_meters × TTC × HC, gated by safety
            raw_progress_gated = raw_progress_meters * safety_gate_arr * ttc * hc

        # Aggregation: soft safety gate × weighted performance average
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
                min_boundary_distance=float(min_boundary_dist[i]),
                mean_obstacle_distance_5m=float(mean_obstacle_dist_5m[i]),
                half_lane_width=float(half_lane_w),
                lateral_offset_signed=float(lat_offset_signed[i]),
                lateral_offset_change=float(lat_offset_change[i]),
                centerline_lateral_offset_start_signed=float(centerline_geom["start_signed"][i]),
                centerline_lateral_offset_end_signed=float(centerline_geom["end_signed"][i]),
                centerline_distance_mean=float(centerline_geom["mean_distance"][i]),
                centerline_distance_max=float(centerline_geom["max_distance"][i]),
                local_centerline_points=centerline_geom["local_centerline_points"],
                boundary_distance_start=float(boundary_geom["start"][i]),
                boundary_distance_end=float(boundary_geom["end"][i]),
                boundary_distance_min=float(boundary_geom["min"][i]),
                boundary_distance_mean=float(boundary_geom["mean"][i]),
                boundary_distances=[float(x) for x in boundary_geom["distances"][i].tolist()],
                boundary_side=boundary_geom["side"][i],
                nearest_boundary_side=boundary_geom["nearest_side"][i],
                nearest_boundary_distance=float(boundary_geom["nearest_distance"][i]),
                in_intersection_fraction=float(topology_geom["in_intersection_fraction"][i]),
                in_intersection_now=in_intersection_now,
                oncoming_fraction=float(topology_geom["oncoming_fraction"][i]),
                non_drivable_fraction=float(topology_geom["non_drivable_fraction"][i]),
                multiple_lanes_fraction=float(topology_geom["multiple_lanes_fraction"][i]),
                in_intersection_flags=[bool(x) for x in topology_geom["in_intersection_flags"][i].tolist()],
                oncoming_flags=[bool(x) for x in topology_geom["oncoming_flags"][i].tolist()],
                non_drivable_flags=[bool(x) for x in topology_geom["non_drivable_flags"][i].tolist()],
                multiple_lanes_flags=[bool(x) for x in topology_geom["multiple_lanes_flags"][i].tolist()],
                boundary_sides=boundary_geom["sides_series"][i],
                collision_per_step=per_step_collisions[i],
                progress_per_waypoint=progress_per_wp[i],
            )
            if rl_config.safety_mode == "discrete":
                result_kwargs.update(
                    safety_gate=float(safety_gate_arr[i]),
                    raw_progress=float(raw_progress_gated[i]),
                )
            result_kwargs.update(
                pdms_score=float(pdms_score_arr[i]),
                pdms_no_at_fault_collisions=float(pdms_nc[i]),
                pdms_drivable_area_compliance=float(pdms_dac[i]),
                pdms_driving_direction_compliance=float(pdms_ddc[i]),
                pdms_traffic_light_compliance=float(pdms_tlc[i]),
                pdms_ego_progress=float(pdms_ep[i]),
                pdms_time_to_collision=float(pdms_ttc[i]),
                pdms_lane_keeping=float(pdms_lk[i]),
                pdms_history_comfort=float(pdms_hc[i]),
            )
            results.append(RLScoringResult(**result_kwargs))
        return results
