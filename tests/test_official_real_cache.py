"""Opt-in real-cache HTTP test. No synthetic scenes or mocked scoring service."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest


CACHE = os.environ.get("GRAVA_TEST_CACHE")
pytestmark = pytest.mark.skipif(not CACHE, reason="Set GRAVA_TEST_CACHE in the pinned NAVSIM environment")


@pytest.fixture(scope="module")
def service():
    from grava_sim_engine.official import verify_runtime
    verify_runtime()
    root = Path(CACHE).resolve()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "grava_sim_engine", "--host", "127.0.0.1", "--port", str(port),
         "--workers", "1", "--dataset", f"test={root}", "--warmup-workers", "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    try:
        for _ in range(100):
            if proc.poll() is not None:
                raise RuntimeError(proc.stderr.read())
            try:
                with urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=1) as response:
                    assert json.load(response)["backend"] == "official_navsim_v1"
                break
            except OSError:
                time.sleep(.1)
        else:
            raise TimeoutError("Official service did not start")
        yield f"http://127.0.0.1:{port}", root
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_three_real_scenes_through_http(service):
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.geometry.convert import absolute_to_relative_poses
    from grava_common.coordinates import CoordinateConverter
    from grava_sim_engine.adapters.navsim.cache_loader import load_metric_cache
    from grava_sim_engine.client import SimEngineClient
    from grava_sim_engine.official import score_metric_cache

    url, root = service
    client = SimEngineClient(url)
    paths = sorted(root.glob("*/*/*/metric_cache.pkl"))[:3]
    assert len(paths) == 3, "Provide at least three genuine metric caches"
    for path in paths:
        log, _, token = path.relative_to(root).parts[:3]
        cache = load_metric_cache(root, log, token)
        poses = [cache.trajectory.get_state_at_time(
            TimePoint(cache.ego_state.time_point.time_us + i * 500000)).rear_axle
            for i in range(1, 9)]
        relative = absolute_to_relative_poses([cache.ego_state.rear_axle, *poses])[1:]
        xy = np.array([[p.x, p.y] for p in relative])
        reference = score_metric_cache(cache, xy)
        model_xy = CoordinateConverter.nuplan_to_nous_batch(xy)
        score, response = client.score(model_xy.tolist(), token, log, "test")
        assert not response.get("error")
        assert score == pytest.approx(reference["pdm_score"], abs=1e-10)
        for key in ("ego_progress", "time_to_collision", "no_at_fault_collisions",
                    "drivable_area_compliance", "history_comfort"):
            assert response[key] == pytest.approx(reference[key], abs=1e-10)
        _, failed = client.score(model_xy.tolist(), token, log, "unregistered")
        assert failed.get("error"), "Infrastructure failure must not masquerade as model failure"
