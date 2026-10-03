# Scene data

The service reads two formats. Scene sources and the generated disk cache are
separate directories. Each configured dataset name identifies one fixed version
of a scene source.

## NAVSIM MetricCache

```text
metric_cache_navtest/
└── <log_name>/
    └── <scenario_type>/
        └── <scene_token>/
            └── metric_cache.pkl
```

These are the standard LZMA-compressed NAVSIM v1 MetricCache pickles. Their
fields include ego state, predicted objects, drivable map, centerline, route lane
IDs and the PDM reference trajectory. Use the same scene split as the training
or evaluation JSONL. A scene is identified by both `log_name` and `scene_token`.

MetricCache datasets support `pdms`, `continuous` and `discrete`, plus PDMS
control scoring. The loader keeps reference headings and converts object/map
geometry into the engine's types. NAVSIM and nuPlan must be installed for a cold
source read or background conversion.

## Prepared SceneContext

```text
scene_contexts/
├── dataset_meta.json
└── <log_name>/
    └── <scene_token>.pkl
```

`dataset_meta.json`:

```json
{
  "format": "scene_context_v1",
  "scoring_modes": ["cds"]
}
```

Each uncompressed pickle contains a
[`grava_sim_engine.core.types.SceneContext`](../src/grava_sim_engine/core/types.py).
This serialization path, and the geometry/observation/map type paths it uses,
remain stable. The service validates the stored scene identity against the
requested log and token. Key-action polygon coordinates must be finite Nx2 arrays
with at least three points and valid, nonempty polygons. Malformed annotations
raise a data error on both source and cache reads; they are neither skipped nor
automatically repaired.

| Fields | Meaning |
|---|---|
| `ego_state`, `ego_past_states` | Current and historical simulator states |
| `observation`, `drivable_area_map` | Dynamic/static occupancy and road polygons |
| `centerline`, `route_lane_ids` | Route geometry |
| `pdm_trajectory`, `pdm_progress`, `pdm_masked_progress` | PDM reference for PDMS and RL |
| `gt_trajectory`, `gt_progress`, `gt_masked_progress` | Recorded human trajectory and reference metrics |
| `key_action_obstacles` | Labeled obstacles used by CDS |
| `candidate_trajectory`, `candidate_progress` | Candidate reference used by CDS |

Simulator states are `[x, y, heading, vx, vy, ax, ay, steering_angle,
steering_rate, angular_velocity, angular_acceleration]`. Scene geometry and ego
states share a global frame; trajectory fields are relative to the ego in nuPlan
coordinates. CDS prepared scenes use an ego-centered map and centerline, with the
current ego pose at the origin. Serialized inputs must be prepared consistently.

A prepared dataset defaults to CDS. Declare other modes in `scoring_modes` only
when its scenes carry the corresponding road, observation and reference context.
The accepted names are `pdms`, `cds`, `continuous` and `discrete`. Control scoring
is enabled when `pdms` is supported. Source metadata contains capabilities, not
per-request scoring options.

Raw business info/frame converters are outside this repository. The service
consumes the prepared scene representation directly. Load pickle assets only
from a trusted source.

## Relation to training data

GRAVA Train's JSONL contains the scene identity and model input. Camera images
provide the visual input. Simulation additionally needs the scene assets above;
JSONL and images alone do not contain maps or future object occupancy.

Scene assets are supplied separately. The examples in this repository contain
trajectory requests; they do not include the corresponding scene caches.

For a new source version, use a new `--cache-dir`. The engine does not compare
source content against existing cache files.
