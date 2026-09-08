# GRAVA Simulation Engine

HTTP trajectory scoring for reproducible GRAVA training and evaluation.
The default backend calls pinned **official NAVSIM v1** directly, using eight
future xy points at 0.5s and a 4s simulation horizon. Run it in the scoring
environment specified by grava-train/requirements/scoring.lock.

```bash
grava-sim-engine --host 127.0.0.1 --port 8100 --workers 1 \
  --dataset train=/data/metric_cache_train \
  --dataset test=/data/metric_cache_test --warmup-workers 0
```

Routes: GET `/v1/health`, POST `/v1/score`, POST `/v1/score/batch`.
Requests name the registered dataset, official log_name/scene_token and
trajectory. HTTP trajectories use NAVSIM's forward-positive x / left-positive y
frame. The Python client can convert the legacy `nous` model-frame convention.
Missing scenes, wrong scoring versions and non-finite inputs fail explicitly.

Install `grava-common` and this package from their matching 0.1.0 revisions.
The `server` extra installs HTTP dependencies; `navsim` pins the official
reference source. See [full installation](https://github.com/AhernResearch/grava-train/blob/main/docs/installation.md).

`--backend standalone` retains the prior standalone implementation for explicit
comparison and optional dataset adapters. It is not the default paper backend.
A real-cache comparison found differences above 1e-4 in a component metric;
therefore it must not be silently substituted for the official backend.

`scripts/navsim_scorer.py` compares the standalone scorer against pinned official
v1 and fails on unavailable reference runtime or differences above tolerance.
No model training or GPU allocation is needed for this comparison.
