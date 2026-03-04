# dqc_sched — Experiments Guide

Our scheduler chooses one of four plans per job:

- **A** — run the full circuit on one QPU (no cutting)
- **B** — circuit cutting, execute all subcircuits sequentially on a single QPU
- **C** — circuit cutting across multiple QPUs (distributed execution)
- **WAIT** — defer if no feasible option exists right now

The main experiment driver is:

python -m qdc_sched.demo.run_demo_workload

It generates a mixed workload, schedules it over simulated time, and exports results to CSV 

## Quickstart

From the repo root:


python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .


Then run:

python -m qdc_sched.demo.run_demo_workload

## Outputs

By default, outputs go to `results/demo_workload/` via `QDC_OUTDIR`.

- `records.csv` — per-job summary metrics  
- `events.csv` — time-ordered event/task timeline (for utilization + Gantt-like plots)

---

## What the workload contains

`make_workload()` produces three kinds of jobs:

1) **Counts jobs** (random CX + measurement): small circuits that always fit (`n is {3,5,7}` and `depth is {3,6,9}`)  
2) **Expectation jobs** (GHZ without measurement + global Z observable): widths sampled from `QDC_WIDE_WIDTHS` (default `10,12,14,16`)  
3) **Wide-slot expectation jobs**: always expectation, chosen to “exercise cutting paths”

Default workload sizing:
- `QDC_N_JOBS=160`
- `QDC_PCT_EXPECT=0.40`
- `QDC_PCT_WIDE=0.15`

## Key exports (CSV formats)

### `records.csv` (per-job summary)

Written to `${QDC_OUTDIR}/records.csv`.

Includes:
- `plan_kind`, `qpu_id`, `submit_time_s`
- timing breakdown: `t_schedule_s`, `t_partition_s`, `t_mapping_s`, `t_execution_s`, `t_reconstruction_s`, `end_to_end_s`
- quality proxies: `fidelity_proxy`, `fidelity_estimated`
- JSON dump of extra metadata: `details_json`

### `events.csv` (event/task timeline)

Written to `${QDC_OUTDIR}/events.csv`.

Each row is a task/event with:
- `kind`, `start_s`, `end_s`, `qpu_id`, `qubits`, dependencies, metadata

This is the file used to build:
- utilization curves
- concurrency over time
- queue delay CDFs
- timing breakdown plots
- fairness/per-QPU share metrics

## Core experiment knobs (environment variables)

### Output / reproducibility
- `QDC_OUTDIR` (default `results/demo_workload`)
- `QDC_N_JOBS` (default `160`)
- `QDC_PCT_EXPECT` (default `0.40`)
- `QDC_PCT_WIDE` (default `0.15`)
- `QDC_WIDE_WIDTHS` (default `10,12,14,16`)

### Backends and queue-delay baselines
- `QDC_EXCLUDE_QPU_C` (default `1`)  
  - If `0`, adds `qpu_C` (a 14-qubit line device) to the backend pool.
- `QDC_QPU_A_BASE_DELAY` (default `1.5`)
- `QDC_QPU_B_BASE_DELAY` (default `0.2`)
- `QDC_QPU_C_BASE_DELAY` (default `0.3`)

### Congestion injection
Congestion is injected by reserving all qubits on `qpu_B` starting at `t=15` and `t=40`, for duration:
- `QDC_CONGEST_BURST_S` (default `15.0`)

### Wide jobs and cutting controls
Wide-slot jobs (subset of jobs) set constraints like:
- `allow_cutting=True`
- `allow_multi_qpu=True`
- `max_cuts=QDC_WIDE_MAX_CUTS` (default `2`)
- `comm_overhead_s=QDC_WIDE_COMM_OVERHEAD_S` (default `0.08`)
- `force_cutting=QDC_FORCE_CUT_WIDE` (default `1`)

Non-wide expectation jobs use:
- `max_cuts=QDC_MAX_CUTS` (default `6`)
- `comm_overhead_s=QDC_COMM_OVERHEAD_S` (default `0.02`)
- `force_cutting` can be enabled with:
  - `QDC_FORCE_CUT_MED=1`or
  - `QDC_FORCE_CUT_IF_GT=<threshold>`

### Timing mode (analytic vs Aer timing)
- `QDC_TIMING_MODE=analytic` (default)
- `QDC_AER_TIMING_REPEATS` (default `1`)
- `QDC_AER_TIMING_USE_NOISE` (default `0`)
- `QDC_AER_TIMING_INCLUDE_TRANSPILE` (default `0`)

### Runtime limits / watchdog
- `QDC_MAX_STEPS` (default `50000`)
- `QDC_MAX_TO_SCHEDULE` (default `2`)
- `QDC_STEP_WALL_CAP_S` (default `15.0`)
- `QDC_WATCHDOG_ACTION`:
  - `adapt` (default): reduces `QDC_MAX_TO_SCHEDULE`
  - `break`: stops the run if a step is too slow

### Wide-job debug printing
- `QDC_WIDE_DEBUG_N` (default `10`) — number of wide jobs to print debug lines for

## Copy/paste experiment recipes

### Baseline A/B (exclude qpu_C, no forced cutting on wide)
```bash
QDC_EXCLUDE_QPU_C=1 QDC_FORCE_CUT_WIDE=0 QDC_OUTDIR=results/exp_baseline_AB python -m qdc_sched.demo.run_demo_workload
```

### Enable qpu_C (allows distributed plan C if it wins)
```bash
QDC_EXCLUDE_QPU_C=0 QDC_FORCE_CUT_WIDE=0 QDC_OUTDIR=results/exp_with_C_enabled python -m qdc_sched.demo.run_demo_workload
```

### Add more congestion (stress scheduling decisions)
```bash
QDC_EXCLUDE_QPU_C=0 QDC_FORCE_CUT_WIDE=0 QDC_CONGEST_BURST_S=30 QDC_PREMIUM_CONGEST_S=180 QDC_OUTDIR=results/exp_congestion_30s python -m qdc_sched.demo.run_demo_workload
```

### Make wide jobs “actually wide”
```bash
QDC_EXCLUDE_QPU_C=0 QDC_FORCE_CUT_WIDE=0 QDC_WIDE_WIDTHS=14,16 QDC_OUTDIR=results/exp_wider python -m qdc_sched.demo.run_demo_workload
```

## Interpreting the console output

### “reserve_calls” lines
Example:
```
[t=  33.0] reserve_calls={'qpu_B': 27, 'qpu_C': 93} | queue=0 pending=0
```

- `reserve_calls` tracks reservation attempts per QPU
- `queue` = jobs not yet admitted (if applicable)
- `pending` = jobs waiting for resources / scheduling to succeed

If `pending` stays near **0**, the system is not heavily contended.

### `[WIDE-DBG]` lines
Example:
```
[WIDE-DBG] job=J002 kind=B_CUT_SINGLE_SEQ qpu_id=qpu_B scores={... 'C_CUT_MULTI_QPU:None': 1.68}
```

This is the fastest way to tell:
- which plan was selected
- what alternatives were considered
- whether `C_CUT_MULTI_QPU` was feasible but lost on score


## Troubleshooting

### “C didn’t show up”
If you see `C_CUT_MULTI_QPU:None` in `[WIDE-DBG] scores={...}` but it never wins, then **C is feasible but loses on score**.

Probable cause in this workload: low contention (often `pending=0` for most of the run), which makes the “parallelism benefit” of C small compared to its overhead terms

Try (our next step UPDATE THIS ONCE WE GET IT WORKING):
- larger congestion windows: `QDC_CONGEST_BURST_S=30` (or bigger)
- wider circuits: `QDC_WIDE_WIDTHS=14,16`
- increase planning/cutting budgets to reduce timeouts and improve candidates

### Cutting timeouts
If you see `TimeoutError: cut timed out after ...`, increase the planner’s cutting budgetsor reduce workload difficulty.


## Where to add new experiments

- Workload generator: `src/qdc_sched/demo/run_demo_workload.py`
- Plan selection logic: `src/qdc_sched/core/planner.py`
- Cutting integrations: `src/qdc_sched/cutting/`
- Event/record logging: `src/qdc_sched/core/metrics.py`