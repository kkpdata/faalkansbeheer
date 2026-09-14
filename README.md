# failure_paths

Python tooling for working with failure-path event graphs from Excel input and for deriving path probabilities and reliability curves.

The tool models flood-defense failure mechanisms (e.g. dike/levee sections) as event graphs, so that individual mechanism probabilities can be combined into failure-path and section-level reliability curves.

## What this repo is for

Use this repository to:

- load failure-path definitions from Excel workbooks;
- validate and inspect event graphs;
- persist scenarios to SQLite for reuse;
- calculate failure-path probabilities over water levels;
- plot and compare path-based `Pf_h` and `Beta_h` curves.

The code is aimed at engineers who need a reproducible workflow from source spreadsheets to analysable graphs and curves.

## Main components

- `failure_paths.eventgraph`: load, build, plot, and persist event graphs.
- `failure_paths.reliability`: reliability integration utilities and curve models.
- `scripts/`: runnable examples for ingestion, plotting, and end-to-end workflows.
- `tests/`: regression tests for Excel parsing, graph logic, SQLite storage, and reliability calculations.

## Setup

[`Pixi`](https://pixi.sh) is the supported way to create the development environment. It installs the Python dependencies plus the Graphviz binary that `EventGraph.plot()` uses. The package metadata is also available in `src/failure_paths/pyproject.toml`, but a pip-only environment does not install the complete workspace toolchain.

```bash
pixi install
pixi run python -c "import failure_paths; print(failure_paths.__version__)"
```

Run project commands with `pixi run`, or enter the environment first with `pixi shell`.

## Typical workflow

1. Load a workbook with `ExcelEventGraph`.
2. Inspect or plot the graph.
3. Store scenarios in SQLite with `EventGraphStore`.
4. Reload them with `SqliteEventGraph`.
5. Derive path probabilities or merged section/node plots.

Minimal example:

```python
from failure_paths import ExcelEventGraph

graph = ExcelEventGraph.load("scripts/example_input/voorbeeld_minimaal.xlsx", "scenario1")
graph.plot(output_path="event_tree.png", water_level=3)
fc_comb, results = graph.get_failure_path_probabilities(water_levels=[3, 4])
```

`graph.plot()` renders the event tree to an image file. `get_failure_path_probabilities()` returns `fc_comb`, the combined scenario fragility curve (`Pf`/`Beta` per water level), plus `results`, a list with one entry per failure path giving its per-node and cumulative probabilities at each requested water level.

## Input assumptions

The Excel-based workflow expects workbook content structured around:

- `metadata`
- `faalpadschema`
- `faalpadkansen`
- one or more `FP_*` frequency sheets

If these tables or sheets are missing or inconsistent, loading will fail fast with validation errors.

## Standalone reliability integration

`failure_paths.reliability` can also be used on its own, outside the event-graph/Excel workflow, to integrate two independent stochasts — for example an empirical water-level exceedance curve (hazard) against a strength fragility curve:

```python
from failure_paths.reliability import IntegrationConfig, ReliabilityIntegrator
from failure_paths.reliability.curves import FragilityCurve, HazardCurve

hazard_curve = HazardCurve(
    hazard_levels=[0, 2, 4, 6, 8],
    exceedance_probs=[1, 0.1, 1e-2, 1e-3, 1e-4],
)
fragility_curve = FragilityCurve(
    hazard_levels=[0, 2, 4, 6, 8],
    betas=[5, 4, 3, 2, 1],
)

config = IntegrationConfig(hazard_curve=hazard_curve, fragility_curve=fragility_curve)
result = ReliabilityIntegrator(config=config).run()
print(result.summary())
```

This combines the two curves into a single failure probability `result.pf` and reliability index `result.beta_pf` (`Φ⁻¹(pf)`, a direct transform of the integrated probability). These are the authoritative outputs and carry their own error bounds: `result.converged`, `result.estimated_logpf_error`, `result.truncation_pf_error_bound`.

Because R and S are independent, `pf` reduces to a 1D integral (`pf = ∫ F_R(s) f_S(s) ds`) instead of a full joint integration — `ReliabilityIntegrator` evaluates this adaptively in standard-normal space, refining the grid only near the limit-state boundary where most of the failure probability mass sits, until a configurable error tolerance (`IntegrationConfig.adaptive_logpf_tol`) is met. This makes it fast and accurate for the very small probabilities typical of flood-defense reliability problems, without resorting to brute-force grid sampling or Monte Carlo. It also detects discrete/step components in either distribution (probability mass concentrated at specific points, e.g. a discrete support or a singularity) and injects mandatory grid edges there, so jumps aren't skipped over by the adaptive refinement; purely continuous distributions and curve-derived knot levels are merged into the same grid.

`result.beta_star` and `result.alpha` are separate FORM-style diagnostics describing *where* failure is most likely to occur, rather than how likely it is. The integrator locates the single most probable failure point and reports `beta_star` as its reliability index and `alpha` as the relative sensitivity of resistance vs. solicitation there, with `result.design_point_physical` giving the same point in physical units. `beta_star` normally sits close to `beta_pf`, but since it describes only that one point, it can diverge when the limit-state curve is irregular or has multiple failure regions — in that case, trust `pf`/`beta_pf` and treat `beta_star`/`alpha` as a rough sensitivity sketch rather than ground truth.

## Useful commands

Run tests from the Pixi environment:

```bash
pixi run pytest
```

Ingest Excel scenarios into SQLite. Edit the configuration constants at the top of the script (`DB_PATH`, `SECTION`, `SHEETS`, `EXCELS`, and `TAGS`) before running it:

```bash
pixi run python scripts/ingest_excels_to_sqlite.py
```

The ingest script also demonstrates loading a scenario back from SQLite and plotting its event tree. Plot merged curves from an existing SQLite database:

```bash
pixi run python scripts/plot_sqlite_vakken_knopen.py \
    --db-path path/to/failure_paths.db \
    --output-file output.png
```

`--sections` and `--nodes` are optional comma-separated filters; omit them to include all available sections or nodes. Use `--metric pf` to plot failure probabilities instead of reliability indices.
