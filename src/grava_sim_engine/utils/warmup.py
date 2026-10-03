"""One launch-level coordinator with a process pool and disk progress records."""

from __future__ import annotations

import json
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.synchronize import Event
from pathlib import Path

from grava_sim_engine.adapters.dataset_loader import Dataset, SceneRef, SceneStore
from grava_sim_engine.utils.cache import write_json

logger = logging.getLogger(__name__)


def progress_path(cache_dir: Path, dataset_name: str) -> Path:
    return cache_dir / dataset_name / "warmup.json"


def initialize_warmup(datasets: list[Dataset], cache_dir: Path, workers: int) -> None:
    for dataset in datasets:
        write_json(
            progress_path(cache_dir, dataset.name),
            {
                "status": "pending" if workers else "disabled",
                "total": None,
                "converted": 0,
                "reused": 0,
                "failed": 0,
                "failures": [],
                "updated_at": time.time(),
            },
        )


def read_warmup(datasets: list[Dataset], cache_dir: Path) -> dict:
    return {
        dataset.name: json.loads(progress_path(cache_dir, dataset.name).read_text())
        for dataset in datasets
    }


def warm_scene(task: tuple[Dataset, Path, SceneRef]) -> dict:
    dataset, cache_dir, scene = task
    store = SceneStore(dataset, cache_dir)
    try:
        reused = store.cache_path(scene.log_name, scene.scene_token).exists()
        store.load(scene.log_name, scene.scene_token)
        return {"status": "reused" if reused else "converted"}
    except Exception:
        # A failed scene must not discard the rest of a dataset's warmup work.
        logger.exception("Warmup failed: %s/%s/%s", dataset.name, scene.log_name, scene.scene_token)
        return {
            "status": "failed",
            "log_name": scene.log_name,
            "scene_token": scene.scene_token,
            "exception": traceback.format_exc(),
        }


def warm_datasets(
    datasets: list[Dataset], cache_dir: Path, workers: int, stop_event: Event | None = None
) -> None:
    for dataset in datasets:
        state = {
            "status": "running",
            "total": None,
            "converted": 0,
            "reused": 0,
            "failed": 0,
            "failures": [],
            "updated_at": time.time(),
        }
        path = progress_path(cache_dir, dataset.name)
        if stop_event is not None and stop_event.is_set():
            state.update(status="task_failed", exception="Service stopped before warmup completed")
            write_json(path, state)
            continue
        try:
            scenes = list(dataset.scenes())
            state["total"] = len(scenes)
            write_json(path, state)
            tasks = ((dataset, cache_dir, scene) for scene in scenes)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(warm_scene, task) for task in tasks]
                stopping = False
                for future in futures:
                    if stop_event is not None and stop_event.is_set() and not stopping:
                        pool.shutdown(wait=True, cancel_futures=True)
                        stopping = True
                        state.update(
                            status="task_failed", exception="Service stopped during warmup"
                        )
                    if future.cancelled():
                        continue
                    result = future.result()
                    state[result["status"]] += 1
                    if result["status"] == "failed":
                        state["failures"].append(result)
                    state["updated_at"] = time.time()
                    write_json(path, state)
            if state["status"] != "task_failed":
                state["status"] = "partial_failed" if state["failed"] else "completed"
                if state["failed"] and state["converted"] + state["reused"] == 0:
                    state["status"] = "task_failed"
        except Exception:
            # Pool/source failures are distinct from individual scene failures.
            logger.exception("Warmup task failed: %s", dataset.name)
            state["status"] = "task_failed"
            state["exception"] = traceback.format_exc()
        state["updated_at"] = time.time()
        write_json(path, state)
