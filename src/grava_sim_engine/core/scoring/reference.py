"""Reference simulation and safety-gated progress shared by the scorers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import coords_to_polygons, state_to_coords
from ..simulator import PDMSimulator
from ..types import SceneContext, VehicleParams
from . import collision_metrics, road_metrics, scene_geometry
from . import trajectory as trajectory_ops


@dataclass
class ReferenceProgress:
    progress: float
    masked_progress: float


def simulate_reference(
    trajectory: np.ndarray,
    scene: SceneContext,
    *,
    simulator: PDMSimulator,
    vehicle: VehicleParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate sparse future waypoints or a dense reference that already includes t=0."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[-1] not in (2, 3) or len(trajectory) == 0:
        raise ValueError(f"Invalid reference trajectory shape: {trajectory.shape}")
    waypoints = trajectory[None, ...]
    proposals = (
        trajectory_ops.build_proposals(waypoints, scene)
        if len(trajectory) == 8
        else trajectory_ops.ego_to_global(waypoints, scene.ego_state)
    )
    states = simulator.simulate_proposals(
        ego_state=scene.ego_state,
        proposals=proposals,
        observation=scene.observation,
    )
    return state_to_coords(states, vehicle), states


def reference_progress(
    trajectory: np.ndarray,
    scene: SceneContext,
    *,
    simulator: PDMSimulator,
    vehicle: VehicleParams,
) -> ReferenceProgress:
    coords, states = simulate_reference(trajectory, scene, simulator=simulator, vehicle=vehicle)
    polygons = coords_to_polygons(coords)
    areas = scene_geometry.calculate_ego_areas(coords, scene)
    progress = float(scene_geometry.progress(coords, scene)[0])
    nc = collision_metrics.no_at_fault_collision(states, polygons, areas, scene)[0]
    dac = road_metrics.drivable_area_compliance(areas)[0]
    return ReferenceProgress(progress, float(progress * (nc * dac)))


def pdm_masked_progress(
    scene: SceneContext,
    *,
    simulator: PDMSimulator,
    vehicle: VehicleParams,
) -> float | None:
    if scene.pdm_masked_progress is not None:
        return scene.pdm_masked_progress
    if scene.pdm_trajectory is None:
        return None
    return reference_progress(
        scene.pdm_trajectory,
        scene,
        simulator=simulator,
        vehicle=vehicle,
    ).masked_progress


def normalize_progress(
    progress_raw: np.ndarray,
    nc: np.ndarray,
    dac: np.ndarray,
    pdm_masked_progress: float | None,
) -> np.ndarray:
    """V1 progress normalized independently for each candidate against PDM."""
    multi_product = nc * dac
    masked = progress_raw * multi_product
    if pdm_masked_progress is not None and pdm_masked_progress > 5.0:
        denominator = np.maximum(masked, pdm_masked_progress)
        return np.divide(masked, denominator, out=np.zeros_like(masked), where=denominator > 0.0)
    normalized = np.ones_like(progress_raw, dtype=np.float64)
    normalized[multi_product == 0.0] = 0.0
    return normalized
