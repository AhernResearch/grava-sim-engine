"""Static dataset specifications and per-request scene loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from grava_sim_engine.adapters.navsim.cache_loader import (
    metric_cache_to_scene_context,
    read_metric_cache,
)
from grava_sim_engine.core.scoring.reference import reference_progress
from grava_sim_engine.core.simulator import PDMSimulator
from grava_sim_engine.core.types import SceneContext, VehicleParams
from grava_sim_engine.utils.cache import read_scene, write_scene

MODES = {"pdms", "cds", "continuous", "discrete"}


@dataclass(frozen=True)
class SceneRef:
    log_name: str
    scene_token: str
    path: Path


@dataclass(frozen=True)
class Dataset:
    name: str
    path: Path
    format: str
    scoring_modes: tuple[str, ...]

    @classmethod
    def open(cls, name: str, path: str | Path) -> Dataset:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid dataset name: {name!r}")
        root = Path(path).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        metadata_path = root / "dataset_meta.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        source_format = metadata.get("format", "navsim_metric_cache")
        if source_format not in ("scene_context_v1", "navsim_metric_cache"):
            raise ValueError(f"Unknown scene format: {source_format}")
        defaults = (
            ["cds"] if source_format == "scene_context_v1" else ["pdms", "continuous", "discrete"]
        )
        modes = tuple(metadata.get("scoring_modes", defaults))
        if not modes or not set(modes) <= MODES:
            raise ValueError(f"Invalid scoring_modes in {metadata_path}: {modes}")
        return cls(name, root, source_format, modes)

    def scenes(self) -> Iterator[SceneRef]:
        self.path.stat()
        pattern = "*/*.pkl" if self.format == "scene_context_v1" else "*/*/*/metric_cache.pkl"
        for path in sorted(self.path.glob(pattern)):
            if self.format == "scene_context_v1":
                yield SceneRef(path.parent.name, path.stem, path)
            else:
                yield SceneRef(path.parents[2].name, path.parent.name, path)

    def source_path(self, log_name: str, scene_token: str) -> Path:
        if self.format == "scene_context_v1":
            path = self.path / log_name / f"{scene_token}.pkl"
            if path.is_file():
                return path
        else:
            matches = list((self.path / log_name).glob(f"*/{scene_token}/metric_cache.pkl"))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous scene: {log_name}/{scene_token}")
        raise SceneNotFound(f"Scene not found: {self.name}/{log_name}/{scene_token}")


class SceneNotFound(FileNotFoundError):
    """The requested scene is absent from the configured source."""


def precompute_reference(scene: SceneContext) -> None:
    vehicle = VehicleParams()
    simulator = PDMSimulator(vehicle=vehicle)
    for prefix in ("gt", "pdm"):
        trajectory = getattr(scene, f"{prefix}_trajectory")
        progress = getattr(scene, f"{prefix}_progress")
        masked = getattr(scene, f"{prefix}_masked_progress")
        if trajectory is None or (progress is not None and masked is not None):
            continue
        result = reference_progress(
            trajectory,
            scene,
            simulator=simulator,
            vehicle=vehicle,
        )
        if progress is None:
            setattr(scene, f"{prefix}_progress", result.progress)
        if masked is None:
            setattr(
                scene,
                f"{prefix}_masked_progress",
                result.masked_progress,
            )


class SceneStore:
    """Load one scene per request; persist converted scenes without retaining them."""

    def __init__(self, dataset: Dataset, cache_dir: Path):
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)

    def cache_path(self, log_name: str, scene_token: str) -> Path:
        return self.cache_dir / self.dataset.name / "scenes" / log_name / f"{scene_token}.pkl"

    def load(self, log_name: str, scene_token: str) -> SceneContext:
        path = self.cache_path(log_name, scene_token)
        if path.exists():
            scene = read_scene(path)
        else:
            source = self.dataset.source_path(log_name, scene_token)
            if self.dataset.format == "scene_context_v1":
                scene = read_scene(source)
            else:
                scene = metric_cache_to_scene_context(read_metric_cache(source), scene_token)
            precompute_reference(scene)
            self._check_identity(scene, log_name, scene_token)
            write_scene(path, scene)
        self._check_identity(scene, log_name, scene_token)
        return scene

    @staticmethod
    def _check_identity(scene: SceneContext, log_name: str, scene_token: str) -> None:
        if (scene.log_name, scene.scene_token) != (log_name, scene_token):
            raise ValueError(f"Scene identity does not match {log_name}/{scene_token}")
