from __future__ import annotations

import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Iterable, NamedTuple

from grava_sim_engine.adapters.navsim import cache_loader as navsim_cache
from grava_sim_engine.core.types import SceneContext

logger = logging.getLogger(__name__)

SCENE_CONTEXT_FORMAT = "scene_context_v1"
DATASET_META_FILENAME = "dataset_meta.json"


class SceneRef(NamedTuple):
    log_name: str
    token: str
    source_path: Path


class DatasetLoader(ABC):
    """Base loader for datasets that can produce SceneContext objects."""

    name: str = "base"

    @classmethod
    @abstractmethod
    def can_load(cls, cache_dir: str | Path) -> bool:
        """Return whether this loader should handle ``cache_dir``."""

    def load_scene_context(
        self,
        cache_dir: str | Path,
        log_name: str,
        token: str,
        boost_cache_dir: str | None,
    ) -> SceneContext:
        if boost_cache_dir is not None:
            boost_path = navsim_cache._boost_cache_path(boost_cache_dir, log_name, token)
            if boost_path.exists():
                ctx = self._load_scene_context_pickle(boost_path)
                if navsim_cache._attach_rl_precompute(ctx):
                    try:
                        navsim_cache._save_to_boost(ctx, boost_path)
                    except OSError:
                        logger.warning(
                            "Failed to update boost cache with backfilled metrics: %s",
                            boost_path,
                            exc_info=True,
                        )
                return ctx

        ctx = self.load_from_source(cache_dir, log_name, token)
        if boost_cache_dir is not None:
            try:
                navsim_cache._save_to_boost(
                    ctx,
                    navsim_cache._boost_cache_path(boost_cache_dir, log_name, token),
                )
            except OSError:
                logger.warning(
                    "Failed to write boost cache for %s/%s", log_name, token, exc_info=True
                )
        return ctx

    @abstractmethod
    def load_from_source(self, cache_dir: str | Path, log_name: str, token: str) -> SceneContext:
        """Load SceneContext from the dataset's canonical source format."""

    def iter_source_scenes(self, cache_dir: str | Path) -> Iterable[SceneRef]:
        return ()

    def warmup_boost_cache(
        self,
        source_dir: str,
        boost_dir: str,
        num_workers: int = 32,
    ) -> dict[str, int]:
        source_root = Path(source_dir)
        boost_root = Path(boost_dir)
        boost_root.mkdir(parents=True, exist_ok=True)
        progress_path = navsim_cache._progress_path(boost_root, source_root)

        if num_workers <= 0:
            navsim_cache._write_progress(
                progress_path,
                source_dir=str(source_root),
                status="disabled",
                total=0,
                converted=0,
                skipped=0,
                failed=0,
            )
            logger.info("Boost warmup disabled for %s (workers=%d)", source_dir, num_workers)
            return navsim_cache.get_warmup_stats()

        lock_path = navsim_cache._lock_path(boost_root, source_root)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            logger.info("Boost warmup already running for %s (lock=%s)", source_dir, lock_path)
            return navsim_cache.get_warmup_stats()

        with os.fdopen(lock_fd, "w") as lock_file:
            lock_file.write(f"pid={os.getpid()} source={source_root}\n")

        converted = skipped = failed = 0
        try:
            refs = list(self.iter_source_scenes(source_root))
            total = len(refs)
            navsim_cache._write_progress(
                progress_path,
                source_dir=str(source_root),
                status="running",
                total=total,
                converted=0,
                skipped=0,
                failed=0,
            )
            if total == 0:
                navsim_cache._write_progress(
                    progress_path,
                    source_dir=str(source_root),
                    status="done",
                    total=0,
                    converted=0,
                    skipped=0,
                    failed=0,
                )
                return navsim_cache.get_warmup_stats()

            workers = max(1, min(num_workers, total))
            chunksize = max(1, min(64, total // max(1, workers * 8) or 1))
            tasks = (
                (str(ref.source_path), ref.log_name, ref.token, str(boost_root))
                for ref in refs
            )
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for idx, status in enumerate(
                    pool.map(_warmup_scene_context_pickle, tasks, chunksize=chunksize),
                    1,
                ):
                    if status == "converted":
                        converted += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                    if idx % 2000 == 0 or idx == total:
                        navsim_cache._write_progress(
                            progress_path,
                            source_dir=str(source_root),
                            status="running",
                            total=total,
                            converted=converted,
                            skipped=skipped,
                            failed=failed,
                        )

            navsim_cache._write_progress(
                progress_path,
                source_dir=str(source_root),
                status="done",
                total=total,
                converted=converted,
                skipped=skipped,
                failed=failed,
            )
            return navsim_cache.get_warmup_stats()
        finally:
            try:
                lock_path.unlink()
            except OSError:
                logger.debug("Failed to remove warmup lock %s", lock_path, exc_info=True)

    @staticmethod
    def _load_scene_context_pickle(path: str | Path) -> SceneContext:
        with open(path, "rb") as file_obj:
            ctx = pickle.load(file_obj)
        if not isinstance(ctx, SceneContext):
            raise TypeError(f"Expected SceneContext in {path}, got {type(ctx).__name__}")
        return ctx


class InternalDatasetLoader(DatasetLoader):
    """Loader for offline-converted SceneContext caches."""

    name = "internal"

    @classmethod
    def can_load(cls, cache_dir: str | Path) -> bool:
        metadata = _read_dataset_metadata(cache_dir)
        return metadata.get("format") == SCENE_CONTEXT_FORMAT

    def load_from_source(self, cache_dir: str | Path, log_name: str, token: str) -> SceneContext:
        source_path = _scene_context_source_path(cache_dir, log_name, token)
        if not source_path.exists():
            raise FileNotFoundError(
                f"SceneContext not found for token={token}, log_name={log_name} under {cache_dir}"
            )
        ctx = self._load_scene_context_pickle(source_path)
        navsim_cache._attach_rl_precompute(ctx)
        return ctx

    def iter_source_scenes(self, cache_dir: str | Path) -> Iterable[SceneRef]:
        root = Path(cache_dir)
        for path in sorted(root.glob("*/*.pkl")):
            if ".warmup" in path.parts:
                continue
            rel = path.relative_to(root)
            if len(rel.parts) != 2:
                continue
            yield SceneRef(log_name=rel.parts[0], token=path.stem, source_path=path)


class NavSimDatasetLoader(DatasetLoader):
    """Loader for NavSim MetricCache directories."""

    name = "navsim"

    @classmethod
    def can_load(cls, cache_dir: str | Path) -> bool:
        return not InternalDatasetLoader.can_load(cache_dir)

    def load_from_source(self, cache_dir: str | Path, log_name: str, token: str) -> SceneContext:
        metric_cache = navsim_cache.load_metric_cache(cache_dir=cache_dir, log_name=log_name, token=token)
        return navsim_cache.metric_cache_to_scene_context(
            metric_cache=metric_cache,
            scene_token=token,
        )

    def warmup_boost_cache(
        self,
        source_dir: str,
        boost_dir: str,
        num_workers: int = 32,
    ) -> dict[str, int]:
        return navsim_cache.warmup_boost_cache(source_dir, boost_dir, num_workers)


def _read_dataset_metadata(cache_dir: str | Path) -> dict:
    path = Path(cache_dir) / DATASET_META_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _scene_context_source_path(cache_dir: str | Path, log_name: str, token: str) -> Path:
    return Path(cache_dir) / str(log_name) / f"{token}.pkl"


def _warmup_scene_context_pickle(args: tuple[str, str, str, str]) -> str:
    source_path_raw, log_name, token, boost_dir = args
    source_path = Path(source_path_raw)
    try:
        boost_path = navsim_cache._boost_cache_path(boost_dir, log_name, token)
        if boost_path.exists():
            return "skipped"
        ctx = DatasetLoader._load_scene_context_pickle(source_path)
        navsim_cache._attach_rl_precompute(ctx)
        navsim_cache._save_to_boost(ctx, boost_path)
        return "converted"
    except Exception:
        logger.warning("Failed to warm boost cache from %s", source_path, exc_info=True)
        return "failed"


def write_scene_context_metadata(cache_dir: str | Path, extra: dict | None = None) -> Path:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": SCENE_CONTEXT_FORMAT,
        "dataset_type": "internal",
        "coordinate_frame": "nuplan_ego",
    }
    if extra:
        payload.update(extra)
    path = root / DATASET_META_FILENAME
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return path


def save_source_scene_context(ctx: SceneContext, cache_dir: str | Path) -> Path:
    path = _scene_context_source_path(cache_dir, ctx.log_name, ctx.scene_token)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with tmp_path.open("wb") as file_obj:
            pickle.dump(ctx, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.debug("Failed to remove temporary source cache file %s", tmp_path)
    return path


def resolve_dataset_loader(cache_dir: str | Path) -> DatasetLoader:
    if InternalDatasetLoader.can_load(cache_dir):
        return InternalDatasetLoader()
    return NavSimDatasetLoader()


@lru_cache(maxsize=256)
def _load_scene_context_cached(
    cache_dir: str,
    log_name: str,
    token: str,
    boost_cache_dir: str | None,
) -> SceneContext:
    loader = resolve_dataset_loader(cache_dir)
    return loader.load_scene_context(cache_dir, log_name, token, boost_cache_dir)


def load_scene_context(cache_dir: str | Path, log_name: str, token: str) -> SceneContext:
    return _load_scene_context_cached(
        str(Path(cache_dir)),
        log_name,
        token,
        navsim_cache.get_boost_cache_dir(),
    )


load_scene_context.cache_clear = _load_scene_context_cached.cache_clear
load_scene_context.cache_info = _load_scene_context_cached.cache_info
load_scene_context.cache_parameters = _load_scene_context_cached.cache_parameters


def set_boost_cache_dir(path: str | None) -> None:
    navsim_cache.set_boost_cache_dir(path)


def get_boost_cache_dir() -> str | None:
    return navsim_cache.get_boost_cache_dir()


def get_warmup_stats() -> dict[str, int]:
    return navsim_cache.get_warmup_stats()


def warmup_boost_cache(source_dir: str, boost_dir: str, num_workers: int = 32) -> dict[str, int]:
    loader = resolve_dataset_loader(source_dir)
    return loader.warmup_boost_cache(source_dir, boost_dir, num_workers)


__all__ = [
    "DATASET_META_FILENAME",
    "SCENE_CONTEXT_FORMAT",
    "DatasetLoader",
    "InternalDatasetLoader",
    "NavSimDatasetLoader",
    "get_boost_cache_dir",
    "get_warmup_stats",
    "load_scene_context",
    "resolve_dataset_loader",
    "save_source_scene_context",
    "set_boost_cache_dir",
    "warmup_boost_cache",
    "write_scene_context_metadata",
]
