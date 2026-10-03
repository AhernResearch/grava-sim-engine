<div align="center">
  <img src="assets/grava_logo.png" alt="GRAVA logo" width="144">
  <h1>GRAVA Sim Engine</h1>
  <p><strong>An efficient simulation and reinforcement learning reward engine for autonomous driving.</strong></p>
  <p>
    <a href="https://github.com/AhernResearch/grava"><img alt="Project" src="https://img.shields.io/badge/Project-GRAVA-2563eb"></a>
    <a href="https://arxiv.org/abs/2609.15169"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-6366f1"></a>
    <a href="https://github.com/AhernResearch/grava-train"><img alt="Training" src="https://img.shields.io/badge/Code-GRAVA_Train-8b5cf6"></a>
    <a href="docs/api.md"><img alt="API" src="https://img.shields.io/badge/Read-API-475569"></a>
  </p>
  <p><strong>English</strong> | <a href="README_zh.md">简体中文</a></p>
</div>

## Overview

GRAVA Sim Engine provides efficient simulation and reward computation for
reinforcement learning in autonomous driving. It simulates candidate trajectories
and computes driving rewards for model training, rollout selection and evaluation.
The engine can run as an independent CPU service accessed over HTTP. Use it with
[GRAVA Train](https://github.com/AhernResearch/grava-train) or connect your own
training code through the same API.

Reinforcement learning repeatedly compares several candidate trajectories for the
same scene. We load that scene once and simulate the candidates in a batch.
Separating this CPU work from model generation and optimization lets simulation
run on its own machines. Simulation resources can then grow with the scoring
workload while the model workers keep their own environments.

Background workers prepare scene caches and reference metrics while the service
handles requests. Completed caches are reusable across restarts. The engine
implements PDMS v1, checked against a pinned NAVSIM reference, alongside CDS and
the continuous/discrete rewards used in our driving experiments.

## Quick start

### 1. Install

Use Python 3.10 for the NAVSIM workflow below. The base package supports Python
3.10 and later.

```bash
git clone https://github.com/AhernResearch/grava-sim-engine.git
cd grava-sim-engine
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For NAVSIM MetricCache input, install the pinned NAVSIM and nuPlan dependencies:

```bash
python -m pip install -e '.[navsim]'
```

Prepared SceneContext files use the base installation. Keep this scoring
environment separate from the model training environment; see
[installation](docs/installation.md) for dependencies and setup details.

### 2. Choose the scene source

| Source | Directory layout | Scoring modes |
| --- | --- | --- |
| NAVSIM MetricCache | `<log_name>/<scenario_type>/<scene_token>/metric_cache.pkl` | PDMS, continuous, discrete |
| Prepared SceneContext | `dataset_meta.json` and `<log_name>/<scene_token>.pkl` | CDS by default; other modes declared in metadata |

The scene source must match the `log_name` and `scene_token` in your training or
evaluation records. Simulation needs these scene assets in addition to the JSONL
and camera images used by the model. See [scene data](docs/data.md) for the source formats.

For NAVSIM, set the source and a separate directory for the converted cache:

```bash
export SCENE_ROOT=/path/to/metric_cache_navtest
export CACHE_DIR=/path/to/grava_scene_cache
```

### 3. Start scoring

```bash
python scripts/serve.py \
  --dataset "navtest=$SCENE_ROOT" \
  --cache-dir "$CACHE_DIR"
```

The service listens on `0.0.0.0:8100`, with up to four HTTP workers and 32 warmup
workers by default. Set `--workers` and `--warmup-workers` for the available CPUs;
`--warmup-workers 0` disables background conversion. Add datasets by repeating
`--dataset NAME=PATH`. The [warmup guide](docs/warmup.md) explains cache reuse and
progress reporting.

In another terminal, check the service and send the recorded example request:

```bash
curl --fail-with-body http://localhost:8100/v1/health
curl --fail-with-body http://localhost:8100/v1/score \
  -H 'Content-Type: application/json' --data @examples/score_request.json
```

Run this from the repository directory with the example's NAVSIM scene available
in `SCENE_ROOT`. To score another scene, change `log_name`, `scene_token` and
`trajectory` in [the request](examples/score_request.json).

## Scoring API

| `scoring_mode` | Main result | Input |
| --- | --- | --- |
| `pdms` | `pdm_score` | Trajectory or controls |
| `cds` | `cds` | Trajectory with key-action scene annotations |
| `continuous` | `rl_score` | Trajectory |
| `discrete` | `rl_score` | Trajectory |

All trajectory modes use `POST /v1/score` and `POST /v1/score/batch`. A trajectory
contains eight future XY or XYH points, spaced 0.5 seconds apart, covering four
seconds. HTTP coordinates follow nuPlan: x forward, y left, heading in radians.
The engine supplies the current ego pose.

The Python client can read the same example request:

```python
import json
from pathlib import Path

from grava_sim_engine.sim_client import SimEngineClient

request = json.loads(Path("examples/score_request.json").read_text())
client = SimEngineClient("http://localhost:8100")
score, metrics = client.score(**request)
print(score)
```

Use `client.score_batch()` for candidates from one scene; results retain their
input order. Scalar scores keep full precision. Set `include_details=True` to
inspect per-step progress, boundary distances and collision diagnostics. CDS
also returns `sample_valid` and `invalid_reason` to distinguish an inapplicable
sample from a valid zero score. Computation failures return HTTP errors.

Control endpoints accept eight `[acceleration, heading_rate]` pairs in m/s² and
rad/s at the same 0.5-second interval. See the [control example](examples/control_request.json)
and [API reference](docs/api.md) for all seven endpoints and reward configuration.
While the service is running, [interactive API documentation](http://localhost:8100/docs)
is available, along with dataset capabilities at `/v1/datasets` and warmup progress
at `/v1/warmup`.

## Use with GRAVA Train

[GRAVA Train](https://github.com/AhernResearch/grava-train) provides model training,
vLLM rollout generation, Active RL and rejection sampling. GRAVA Sim Engine
provides the simulation rewards. The training code calls the scoring API over HTTP, so the simulator and model
workers can run on separate machines.

For GRPO rewards, set these variables in the training environment. Use an address
reachable from the training workers and a dataset name configured on the service:

```bash
export SIM_ENGINE_URL=http://localhost:8100
export SIM_ENGINE_DATASET=navtest
```

Evaluation and rollout scripts both use `--sim-url` for the service address.
Set the scene dataset with `--dataset` in evaluation and `--sim-dataset` in rollout.
Continue with
[GRPO training](https://github.com/AhernResearch/grava-train/blob/main/docs/training.md#grpo),
[Active RL](https://github.com/AhernResearch/grava-train/blob/main/docs/active_rl.md)
or [self-distillation](https://github.com/AhernResearch/grava-train/blob/main/docs/self_distillation.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/installation.md) | Environments, dependencies and development checks |
| [Scene data](docs/data.md) | MetricCache and SceneContext layouts |
| [HTTP API](docs/api.md) | Requests, score fields, controls and configuration |
| [Warmup](docs/warmup.md) | Worker settings, disk cache and progress |
| [NAVSIM alignment](docs/alignment.md) | Pinned reference, comparison script and numerical regression |

## Citation

If you use this code in your research, please cite [GRAVA](https://arxiv.org/abs/2609.15169):

```bibtex
@misc{liu2026grava,
  title         = {GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving},
  author        = {Xiao Liu and Haoyu Li and Jianghao Leng and Lin Wang and Chao Sun},
  year          = {2026},
  eprint        = {2609.15169},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.15169}
}
```

## Acknowledgements

We thank the [NAVSIM](https://github.com/autonomousvision/navsim) and
[nuPlan](https://github.com/motional/nuplan-devkit) teams for their simulation tools,
data formats and evaluation protocols.
