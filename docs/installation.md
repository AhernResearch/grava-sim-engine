# Installation

Use a separate CPU environment for the scoring service. GRAVA Train connects over
HTTP and does not need the simulator's dependencies in its training environment.

## Prepared SceneContext

Python 3.10 or later is required.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The base install includes NumPy, SciPy, Shapely, FastAPI, Pydantic 2 and Uvicorn.
Prepared SceneContext files use the engine's own classes and need no NAVSIM import.
The service does not require a GPU.

## NAVSIM MetricCache

MetricCache files contain NAVSIM and nuPlan objects. Install their real types in
the scoring environment:

```bash
python -m pip install -e '.[navsim]'
```

This extra pins NAVSIM to `0811876c274e8b058ab2be9b3dcd4d37bd23f177`, which also
specifies its nuPlan dependency. Use Python 3.10: the upstream requirements pin
NumPy 1.23.4 and include packages for the wider NAVSIM stack. The service reads
MetricCache using the original pickle types and then converts it to SceneContext.

For a checkout-based [alignment check](alignment.md), install the same extra,
then select the pinned reference checkout with `PYTHONPATH` as shown there.

## Start the service

```bash
python scripts/serve.py \
  --dataset navtest=/path/to/metric_cache_navtest \
  --dataset action=/path/to/scene_contexts \
  --cache-dir /path/to/grava_scene_cache \
  --host 0.0.0.0 --port 8100 --workers 4 --warmup-workers 32
```

`--dataset` and `--cache-dir` are required. HTTP workers default to
`min(4, os.cpu_count())`; the warmup pool defaults to 32 workers. Choose these
counts for the CPU and storage capacity available to the service.

Dataset names are fixed for the life of a service instance. Change the launch
arguments and restart to change the configured datasets. See [data](data.md)
and [warmup](warmup.md).

## Development checks

```bash
python -m pip install -e '.[dev]'
ruff check src scripts
```

For numerical validation, run the [NAVSIM comparison](alignment.md) against your
scene assets.
