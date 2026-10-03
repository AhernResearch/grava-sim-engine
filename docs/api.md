# HTTP API

The default address is `http://localhost:8100`. FastAPI also exposes the request
schema at `/docs` and `/openapi.json`.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/v1/score` | One trajectory, any supported mode |
| POST | `/v1/score/batch` | Trajectories from one scene |
| POST | `/v1/score/control` | One control sequence, PDMS |
| POST | `/v1/score/control/batch` | Control sequences from one scene, PDMS |
| GET | `/v1/health` | Status and package version |
| GET | `/v1/datasets` | Startup datasets, formats and scoring capabilities |
| GET | `/v1/warmup` | Per-dataset conversion progress and failures |

## Requests

All scoring requests require `dataset`, `log_name` and `scene_token`.

| Field | Default | Meaning |
|---|---|---|
| `trajectory` / `trajectories` | Required | Eight future XY or XYH points per candidate |
| `scoring_mode` | `pdms` | `pdms`, `cds`, `continuous` or `discrete` |
| `config_overrides` | `{}` | Mode-specific CDS/RL numerical options |
| `include_details` | `false` | Return per-step diagnostic arrays |

Each trajectory spans 4 seconds with samples at 0.5, 1.0, …, 4.0 seconds.
Coordinates are ego-relative nuPlan coordinates: x forward, y left, heading in
radians. Provided headings are preserved. XY input uses the existing spline
heading calculation. The service supplies the current ego pose.

A batch shares the scene, mode and configuration. It may contain both XY and XYH
candidates. `results` preserves input count and order. Use `trajectory` for the
single endpoint and `trajectories` for the batch endpoint.

Control endpoints accept `control_signals` or `control_signals_batch`: eight
`[acceleration, heading_rate]` pairs at 0.5-second intervals, in m/s² and rad/s.
They score PDMS and do not accept `scoring_mode` or `config_overrides`.

[Trajectory](../examples/score_request.json) and
[control](../examples/control_request.json) examples use recorded inputs.

## Configuration

For `continuous` and `discrete`, supported keys come from
[`RLScorerConfig`](../src/grava_sim_engine/core/scoring/config.py).
For `cds`, they come from
[`CDSScorerConfig`](../src/grava_sim_engine/core/scoring/cds_scorer.py).
`scoring_mode` selects the safety mode; it cannot be overridden inside config.
PDMS uses its fixed formula and accepts no configuration overrides.

Weights and margins must be finite and non-negative; divisor scales must be
positive. Unknown keys are rejected. For example, a continuous request can set:

```json
{"scoring_mode": "continuous", "config_overrides": {"ep_weight": 5.0, "ttc_weight": 5.0, "hc_weight": 2.0}}
```

## Scores

| Mode | Main field | Other fields |
|---|---|---|
| PDMS | `pdm_score` | Collision, drivable-area, progress, TTC, comfort and diagnostic road metrics |
| CDS | `cds` | `safety_score`, `comfort_score`, `key_action_score`, `progress_score`, validity |
| Continuous/discrete | `rl_score` | Current-mode reward components and geometry scalars |

PDMS uses `(NC × DAC) × (5 EP + 5 TTC + 2 HC) / 12`.
The RL composite retains its safety gate and configured performance weights.
RL responses also contain `pdm_score` and a separate `pdms_metrics` object holding
the discrete PDMS components, including `no_at_fault_collisions`,
`drivable_area_compliance`, `ego_progress`, `time_to_collision` and `history_comfort`.
The same-named top-level components describe the selected RL mode.

`min_boundary_distance` is the minimum signed distance from ego corners to the
road boundary over the simulated horizon, in meters: positive inside and negative
outside. Both RL modes return the computed value.

Discrete RL additionally returns `gated_progress` in meters:
`actual_progress × NC × DAC × TTC × HC`. Continuous RL omits this field.
CDS retains its own `raw_progress` field, with its existing route-progress meaning.

Scores and training scalars retain full precision. They are not rounded for HTTP.

### Details

With `include_details: true`, RL responses include a `details` object containing:

- `local_centerline_points`, `boundary_distances`, `boundary_sides`;
- `in_intersection_flags`, `oncoming_flags`, `non_drivable_flags`, `multiple_lanes_flags`;
- `collision_per_step`, `progress_per_waypoint`.

These arrays are omitted by default. Reference simulation for per-waypoint progress,
local centerline sampling and per-step collision diagnostics run only when details
are requested. Calculations shared with scalar metrics still run in both cases.
Scalar summaries remain at the top level.
The standalone Python client leaves geometry in nuPlan coordinates; GRAVA Train's
client converts `details.local_centerline_points` to GRAVA coordinates.

### CDS validity

`sample_valid: false` with `invalid_reason` identifies a scene to which CDS does
not apply, such as a scene without relevant labeled key actions. This is separate
from an applicable trajectory receiving `cds: 0`. Keep the validity fields when
aggregating evaluation results or selecting rollout targets.

## Errors

Unknown datasets and absent scenes return 404. Invalid shapes, unsupported modes,
unknown fields and invalid configurations return 422. Corrupt scene files and
simulation failures return 5xx, with tracebacks in service logs. A batch with a
calculation failure fails as a whole. Successful responses have no error placeholder.
Clients propagate HTTP, decoding and missing-field failures.
