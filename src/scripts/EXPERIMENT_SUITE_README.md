# Scheduler Experiment Suite

Files:
- `scheduler_experiment_suite.py` — runs preset experiment suites against the scheduler.
- `analyze_experiment_suite.py` — summarizes per-run CSVs into `summary_runs.csv` and `summary_aggregated.csv`.
- `plot_experiment_suite.py` — makes first-pass figures from `summary_aggregated.csv`.

## Suggested placement

Put these under your repo, for example:
- `scripts/scheduler_experiment_suite.py`
- `scripts/analyze_experiment_suite.py`
- `scripts/plot_experiment_suite.py`

Run from the repo root so `qdc_sched` imports resolve.

## Example usage

Run the system-value suite with 3 seeds:

```bash
python scripts/scheduler_experiment_suite.py --suite system_value --seeds 0,1,2 --outdir results/experiment_suite
```

Run the force-cut and method-comparison suites:

```bash
python scripts/scheduler_experiment_suite.py --suite forcecut_value --seeds 0,1,2 --outdir results/experiment_suite
python scripts/scheduler_experiment_suite.py --suite cut_method --seeds 0,1,2 --outdir results/experiment_suite
```

Run the smaller quality suite:

```bash
python scripts/scheduler_experiment_suite.py --suite quality_small --seeds 0,1 --outdir results/experiment_suite
```

Summarize:

```bash
python scripts/analyze_experiment_suite.py results/experiment_suite
```

Plot:

```bash
python scripts/plot_experiment_suite.py results/experiment_suite
```

## Current suite groups

- `system_value`
  - compares scheduler-aware FitCut vs no-cut scheduler vs cut-single-seq baseline
  - workloads: `light_fit`, `mixed_fit`

- `forcecut_value`
  - compares methods on a workload with jobs too large for any single backend
  - workloads: `mixed_forcecut`

- `cut_method`
  - compares FitCut vs addon vs naive chunking inside the same scheduler
  - workloads: `mixed_fit`, `mixed_forcecut`

- `quality_small`
  - smaller full-evaluation sweep for quality-focused runs

## Output structure

Each run directory contains:
- `records.csv`
- `events.csv`
- `job_manifest.csv`
- `run_meta.json`
- `stdout_summary.json`

The suite root contains:
- `suite_manifest.json`
- `summary_runs.csv`
- `summary_aggregated.csv`
- `figures/*.png`

## Important notes

- The runner defaults to analytic timing mode and reserves non-simulated tasks, matching your current scheduler experiments.
- It forces planner-side fast partitioning defaults to avoid the previous planner stall.
- `quality_small` sets `compute_expectation=True`, but it still depends on the current executor/cutting support in your repo.
- The naive baseline is intentionally simple and is meant as a lower-bound/reference baseline, not a strong production cutting method.
