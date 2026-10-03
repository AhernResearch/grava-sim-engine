# Numerical alignment

The online service uses GRAVA's own simulation and scoring components. Official
NAVSIM v1 is a separate reference for regression validation.

Reference revision: `0811876c274e8b058ab2be9b3dcd4d37bd23f177`.
The scalar tolerance is `1e-4`.

## Run the official comparison

Install the [NAVSIM dependencies](installation.md), then check out the reference
outside the engine repository:

```bash
git clone https://github.com/autonomousvision/navsim.git /path/to/navsim_v1
git -C /path/to/navsim_v1 checkout 0811876c274e8b058ab2be9b3dcd4d37bd23f177
export NAVSIM_ROOT=/path/to/navsim_v1
export METRIC_CACHE=/path/to/metric_cache_navtest
PYTHONPATH="$NAVSIM_ROOT:$PWD/src" python scripts/check_navsim.py \
  --metric-cache "$METRIC_CACHE" --cases examples/alignment_cases.json \
  --output /tmp/grava_navsim_alignment.json
```

The script verifies the imported checkout's commit and compares PDMS, NC, DAC,
EP, TTC and comfort. Missing data or dependencies fail the check. The example
contains six recorded trajectories from three NAVSIM scenes: one PDM output
with heading and one human GT trajectory with XY per scene. For XY, both paths
use the engine's existing tangent-heading convention.

The numerical check does not start a service or a training job.

## Validation results

Before restructuring the engine, full-precision results were captured for nine
real inputs from six scenes. Local regression checks cover PDMS, CDS, both RL
modes and three recorded control sequences using standardized SceneContexts.
The test suite and recorded scene caches are maintained locally; the public
comparison script above runs with your own NAVSIM scene assets.

The response migration changed field organization and names. It also made the
discrete minimum boundary distance report the measured distance, and removed
HTTP rounding. The motion model, interpolation, scoring formulas and default
reward weights were preserved. Batch normalization uses the same per-candidate
reference semantics as single scoring, including scenes without a usable PDM
reference.

On 2026-10-02, the six official comparisons above passed with maximum scalar
difference `9.541151158032335e-8`. This is a regression result on the listed
inputs. The earlier full NAVSIM v1 alignment run covered 12,146 human-GT scenes
with maximum PDMS difference `2.4e-7`; this refactor did not repeat that full run.

| Check | Environment used |
|---|---|
| Engine, HTTP and warmup regression | Python 3.10; NumPy 2.2.6, SciPy 1.15.3, Shapely 2.1.2 |
| Pinned official comparison | Existing Python 3.9 reference environment; NumPy 1.23.4, SciPy 1.13.1, Shapely 2.0.7 |

The package requires Python 3.10+. The older reference environment above records
where the official comparison was executed; it is not the service installation
target. Fresh dependency installation was not verified in the offline workspace.

Three real MetricCaches were also read cold, warmed with a process pool, and
reused after restart; cold and cached scores matched. The base wheel was built
and its service/client imports checked in an isolated install directory.

A subsequent review corrected `progress_per_waypoint`: dense 0.1-second PDM
references already include the current pose and are no longer interpolated as
0.5-second future points. This changes that diagnostic array, not the primary
scores. Local regression tests check the corrected time dimensions and
displacement ratios on real scenes.
