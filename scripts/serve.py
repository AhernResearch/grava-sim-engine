"""Start HTTP workers and one background scene warmup coordinator."""

import argparse
import json
import logging
import multiprocessing
import os
import signal
from multiprocessing.synchronize import Event
from pathlib import Path

import uvicorn

from grava_sim_engine.adapters.dataset_loader import Dataset
from grava_sim_engine.utils.warmup import initialize_warmup, warm_datasets


def run_warmup(datasets: list[Dataset], cache_dir: Path, workers: int, stop_event: Event) -> None:
    # The launch process coordinates shutdown; converters finish their atomic writes.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    warm_datasets(datasets, cache_dir, workers, stop_event)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--warmup-workers", type=int, default=32)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    if args.warmup_workers < 0 or args.workers < 1:
        parser.error("--warmup-workers must be >= 0 and --workers must be >= 1")
    specifications = [item.split("=", 1) for item in args.dataset]
    if any(len(item) != 2 for item in specifications):
        parser.error("--dataset must be NAME=PATH")
    if len({name for name, _ in specifications}) != len(specifications):
        parser.error("Dataset names must be unique")
    datasets = [Dataset.open(name, path) for name, path in specifications]
    cache_dir = args.cache_dir.resolve()
    os.environ["GRAVA_SIM_CONFIG"] = json.dumps(
        {
            "datasets": {dataset.name: str(dataset.path) for dataset in datasets},
            "cache_dir": str(cache_dir),
        }
    )
    logging.basicConfig(level=logging.INFO)
    initialize_warmup(datasets, cache_dir, args.warmup_workers)
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    coordinator = None
    if args.warmup_workers:
        coordinator = context.Process(
            target=run_warmup,
            args=(datasets, cache_dir, args.warmup_workers, stop_event),
            name="scene_warmup",
        )
        coordinator.start()
    try:
        uvicorn.run(
            "grava_sim_engine.server.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            workers=args.workers,
        )
    finally:
        if coordinator is not None:
            stop_event.set()
            coordinator.join()


if __name__ == "__main__":
    main()
