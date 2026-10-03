"""Atomic SceneContext and progress files shared by independent workers."""

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from grava_sim_engine.core.types import SceneContext


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_scene(path: Path, scene: SceneContext) -> None:
    atomic_write(path, pickle.dumps(scene, protocol=pickle.HIGHEST_PROTOCOL))


def read_scene(path: Path) -> SceneContext:
    with path.open("rb") as stream:
        scene = pickle.load(stream)
    if not isinstance(scene, SceneContext):
        raise TypeError(f"Expected SceneContext in {path}, got {type(scene).__name__}")
    for annotation in scene.key_action_obstacles:
        coords = np.asarray(annotation.polygon_coords)
        if (
            coords.ndim != 2
            or coords.shape[0] < 3
            or coords.shape[1] != 2
            or not np.isfinite(coords).all()
        ):
            raise ValueError(
                f"Invalid key-action polygon {annotation.token} in {path}: expected finite Nx2 coordinates, N >= 3"
            )
        polygon = Polygon(coords)
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError(
                f"Invalid key-action polygon {annotation.token} in {path}: {explain_validity(polygon)}"
            )
    return scene


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())
