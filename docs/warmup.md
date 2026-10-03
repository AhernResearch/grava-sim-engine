# Background warmup and disk cache

```text
MetricCache / prepared SceneContext
        ↓ background conversion and reference precomputation
Disk SceneContext
        ↓ load once per request
Batch simulation → scores
```

`scripts/serve.py` starts one warmup coordinator for the service instance. It
owns a process pool controlled by `--warmup-workers` (default 32). HTTP workers
only read scenes and progress files; starting more HTTP workers does not start
another warmup job. Set `--warmup-workers 0` to disable the background job.

## Disk layout

```text
cache_dir/
└── <dataset_name>/
    ├── warmup.json
    └── scenes/
        └── <log_name>/
            └── <scene_token>.pkl
```

Each converted scene includes precomputed reference progress where the scene
provides reference trajectories. Files are written to a temporary file in the
same directory, then atomically replaced. Dataset names isolate caches even when
two datasets contain the same log and token.

A cache miss loads the canonical source, precomputes references and writes the
scene to disk. A later request reads the disk artifact. Scenes are not retained
across requests. Within a batch, all candidates share the loaded scene and
reference calculation.

On restart, warmup reads and validates existing artifacts, then reuses them.
A corrupt cache is reported as a failure and is not replaced silently from the
source. Repair or remove the corrupt artifact before retrying. A source change
requires a new cache directory; source versions are managed by the caller.

## Progress and failures

`GET /v1/warmup` returns `datasets`, keyed by dataset name. Each value contains
`status`, `total`, `converted`, `reused`, `failed`, `failures` and `updated_at`.
`total` is null until source enumeration completes, then reports the actual scene
count. It remains null when background warmup is disabled.

| Status | Meaning |
|---|---|
| `pending` | Coordinator has not started this dataset |
| `running` | Source enumeration or conversion is in progress |
| `completed` | All scenes converted or validated and reused |
| `partial_failed` | Per-scene failures; other scenes finished |
| `task_failed` | Dataset/pool failure, all scenes failed, or warmup was stopped |
| `disabled` | Started with zero warmup workers |

A failed scene records its log, token and exception traceback in `failures` and
in service logs. Processing continues for other scenes. A dataset-level failure
also records `exception`. Progress files survive a process exit; each launch
initializes a new progress run while keeping completed scene artifacts.

On service shutdown, the coordinator cancels queued conversions and waits for
running conversions to finish. Unfinished warmup runs are marked `task_failed`;
completed scene files remain available for the next launch.
