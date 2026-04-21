# Resource-Aware Scheduling for Distributed Quantum Computing

This repository contains the implementation and experimental evaluation for the
paper **"Resource-Aware Scheduling for Distributed Quantum Computing"**. It
includes the `qdc-sched` Python package — a modular distributed quantum
computing scheduler with circuit-cutting hooks — together with all experiment
runners, a unified figure generator, and statistical analysis scripts needed to
reproduce the paper's results.

---

## Requirements

- Python >= 3.10
- `qiskit >= 2.0.0`
- `qiskit-aer >= 0.14.0`
- `qiskit-addon-cutting >= 0.10.0`
- `networkx >= 3.0`
- `numpy`, `matplotlib`, `pandas`, `scipy`
- *(Optional)* `qiskit-ibm-runtime >= 0.24.0` — required for the `qdc_sched.ibm`
  submodule (fake backend loading and IBM device profile builder)

No physical quantum hardware is required for now. All experiments run on classical
CPU hardware using simulated QPUs and Qiskit AerSimulator. A modern workstation
with 4+ CPU cores and 16 GB RAM is recommended.

---

## Installation

```bash
git clone https://github.com/cwhowe/DQC_Scheduler
cd DQC_Scheduler
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify the installation:

```bash
python -c "import qdc_sched; print(qdc_sched.__version__)"
```

For IBM backend support:

```bash
pip install -e ".[ibm]"
```

---

## Package Structure

The scheduler is implemented as the `qdc-sched` package under `src/qdc_sched/`:

```
src/qdc_sched/
├── core/
│   ├── scheduler.py     # Top-level Scheduler and SchedulerConfig; tick loop,
│   │                    #   pending queue, plan dispatch
│   ├── planner.py       # Planner and PlannerConfig; generates Plan A/B/C
│   │                    #   candidates and selects among them
│   ├── executor.py      # Executor and ExecConfig; simulates QPU execution,
│   │                    #   communication, and reconstruction on the timeline
│   ├── hardware.py      # HardwareProfile and QPUState; QPU topology,
│   │                    #   error rates, gate times, reservation management
│   ├── types.py         # Core data types: Job, Plan, Task, TaskGraph,
│   │                    #   CircuitProfile, JobConstraints, RunToggles
│   ├── profiler.py      # Circuit profiling: width, depth, gate counts,
│   │                    #   QPU fit analysis
│   ├── quality.py       # QualityModel: fidelity proxy estimation from
│   │                    #   hardware error rates and circuit structure
│   ├── runtime.py       # Analytic timing models for QPU execution,
│   │                    #   communication, and reconstruction duration
│   ├── metrics.py       # MetricsRecorder and JobRunRecord; per-job timing
│   │                    #   and quality bookkeeping
│   └── resources.py     # Resource tracking helpers
│
├── cutting/
│   ├── fitcut.py        # FitCutCutStrategy: primary cutting strategy,
│   │                    #   uses qiskit-addon-cutting under the hood
│   ├── qiskit_addon.py  # QiskitAddonCutStrategy: direct addon wrapper
│   ├── assignment.py    # MinMakespanGreedyAssignment: assigns subcircuits
│   │                    #   to QPUs to minimise makespan
│   └── base.py          # Abstract CutStrategy, CutConstraints,
│                        #   CutAnalysis, PartitionPlan interfaces
│
├── ibm/
│   ├── fake_loader.py       # Load Qiskit fake IBM backends for simulation
│   └── profile_builder.py  # Build HardwareProfile from IBM device data
│
└── demo/
    ├── run_demo_workload.py  # End-to-end demo: submit a small workload
    └── validate_timing.py    # Validate analytic timing against simulation
```

---

## Scheduling Plans

The scheduler selects among three execution plans per job:

| Plan | Name | Description |
|------|------|-------------|
| A | No-cut single-QPU | Run the circuit intact on the best available QPU |
| B | Cut single-QPU sequential | Cut the circuit and run subcircuits sequentially on one QPU |
| C | Cut multi-QPU distributed | Cut the circuit and run subcircuits in parallel across multiple QPUs |

A fourth outcome, WAIT (deferral), occurs when no plan meets the job's
constraints at the current time step.

---

## Repository Structure

```
DQC_Scheduler/
├── src/qdc_sched/              # Scheduler package (see above)
├── pyproject.toml              # Package build config (setuptools, Python >=3.10)
│
├── run_suite.py                # Main experiment runner — all paper suites
├── run_experiments_supplementary.py  # Supplementary experiments E17–E24
├── run_ci_seeds.py             # Repeated-seed runner for CI/Wilcoxon analysis
│
├── plot_figures.py             # Unified figure generator — all paper figures
│
├── analyze_ci.py               # Bootstrap CIs + Wilcoxon tests for key claims
├── analyze_experiment_suite.py # Aggregate suite run dirs into summary CSVs
├── analyze_seed_stats.py       # E1-focused CI analysis (supplementary)
├── analyze_e1_ci.py            # Minimal E1-only CI script (supplementary)
├── summarize_records.py        # Inspect and compare individual records.csv files
│
└── results/                    # Created at runtime; not committed
```

---

## Running the Experiments

### Quick smoke test

Add `--fast` to verify everything works before a full run. Completes in under
30 minutes:

```bash
python run_suite.py --suite paper_main --seeds 2026 \
    --outdir results/experiments --fast

python run_experiments_supplementary.py --seed 2026 \
    --outdir results/experiments --fast

python run_experiments_supplementary.py --diagnose  # API compatibility check
```

---

### Stage 1 — Main experiment suite

`run_suite.py` runs the core paper experiments across configurable workload
presets and scheduling methods. All outputs are flat CSVs written to `--outdir`.

```bash
python run_suite.py \
    --suite all \
    --seeds 2026 \
    --outdir results/experiments
```

**Available suites:**

| Suite | Description |
|-------|-------------|
| `system_value` | Plan A/B/C comparison on workloads that fit a single QPU |
| `forcecut_value` | Forced-cut workloads that stress Plan B and C paths |
| `cut_method` | FitCut vs QiskitAddon vs Naive cutting strategy comparison |
| `quality_small` | Full-evaluation quality experiment with fidelity estimation (slow) |
| `paper_main` | The two primary paper workloads |
| `all` | All suites above |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--suite` | `all` | Suite(s) to run |
| `--seeds` | `2026` | Comma-separated random seeds |
| `--outdir` | `results/experiments` | Output directory for CSVs |
| `--fast` | off | Reduced job counts for quick testing |

After the suite completes, optionally aggregate the per-run metadata:

```bash
python analyze_experiment_suite.py results/experiment_suite
```

---

### Stage 2 — Supplementary experiments (E17–E24)

`run_experiments_supplementary.py` runs all supplementary experiments. Any
subset can be selected with `--experiments`:

```bash
python run_experiments_supplementary.py \
    --experiments E17,E18,E19,E20,E21,E24 \
    --n-jobs 40 \
    --seed 2026 \
    --outdir results/experiments
```

**Experiments:**

| ID | Description | Output file |
|----|-------------|-------------|
| E17 | Congestion × arrival-rate joint sweep (4×3 = 12 conditions) | `e17_congestion_arrival.csv` |
| E18 | Utilisation–throughput Pareto frontier (pool-size × λ grid) | `e18_utilization_pareto.csv` |
| E19 | Job-stream composition sweep (heavy-circuit fraction 0→100%) | `e19_stream_composition.csv` |
| E20 | Batch vs stream: 7 submission strategies on a fixed mixed workload | `e20_batch_stream.csv` |
| E21 | Throughput scaling vs job load (heterogeneous 3-QPU pool, λ=2.0) | `e21_throughput_scaling.json` |
| E24 | QPU idle fraction vs arrival rate λ (homogeneous 3-QPU pool) | `e24_idle_fraction.json` |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--experiments` | all | Comma-separated subset, e.g. `E19,E20` |
| `--n-jobs` | `40` | Jobs per condition |
| `--seed` | `2026` | Random seed |
| `--outdir` | `results/experiments` | Output directory |
| `--fast` | off | Reduced sweep sizes |
| `--diagnose` | off | Smoke-test API calls and exit |

---

### Stage 3 — Repeated-seed runs for confidence intervals

`run_ci_seeds.py` reruns E11, E19, and E20 across multiple seeds to produce
the `seed_N/` directories consumed by `analyze_ci.py`:

```bash
python run_ci_seeds.py \
    --outdir results/paper_ci \
    --seeds 2026,2027,2028,2029,2030 \
    --n-jobs 40
```

The paper used seeds 2026–2034 (9 seeds). Use `--fast` for a quick check.
The `--experiments` flag accepts any subset of `E11,E19,E20`.

---

### Stage 4 — Generate figures

All paper figures are generated by a single script. Figures whose required
input CSV or JSON is not found are silently skipped:

```bash
python plot_figures.py \
    --indir  results/experiments \
    --indir2 results/experiments \
    --outdir results/paper_figures
```

To generate a specific subset:

```bash
python plot_figures.py \
    --indir  results/experiments \
    --indir2 results/experiments \
    --outdir results/paper_figures \
    --figures 1,2,16,49,60,61
```

To apply post-review style and annotation patches to selected figures:

```bash
python plot_figures.py \
    --indir  results/experiments \
    --indir2 results/experiments \
    --outdir results/paper_figures \
    --patch
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--indir` | `results/experiments` | Input directory for E1–E16 CSVs |
| `--indir2` | same as `--indir` | Input directory for E17–E24 CSVs/JSONs |
| `--outdir` | `results/paper_figures` | Output directory for figures |
| `--figures` | `all` | Comma-separated figure numbers, or `all` |
| `--style` | `paper` | Font size preset: `paper` or `slides` |
| `--patch` | off | Apply post-review patches to selected figures |

---

### Stage 5 — Statistical analysis

`analyze_ci.py` computes bootstrap confidence intervals and Wilcoxon
signed-rank tests for the paper's key quantitative claims. Configure the
`E1_ROOT`, `E11_ROOT`, `E19_ROOT`, and `E20_ROOT` paths at the top of the
script to point at your `results/paper_ci/` directory, then run:

```bash
python analyze_ci.py
```

It reports results for six claims:

1. Main latency result — no-cut vs cut-multi P90 end-to-end latency (E1)
2. Scheduler decision overhead as a fraction of end-to-end latency (E1)
3. Reconstruction share of total cutting overhead (E1)
4. Batch vs stream P90 latency comparison (E20)
5. SLO completion rate — unconstrained vs SLO-constrained scheduling (E11)
6. Stream composition sweep — low-heavy vs high-heavy workload P90 (E19)

---

## Expected Outputs

| Script | Output location | Contents |
|--------|----------------|----------|
| `run_suite.py` | `results/experiments/` | Flat CSVs per experiment (e1_plan_comparison.csv, etc.) and per-run subdirs under `results/experiment_suite/` |
| `run_experiments_supplementary.py` | `results/experiments/` | `e17_*.csv`, `e18_*.csv`, `e19_*.csv`, `e20_*.csv`, `e21_*.json`, `e24_*.json` |
| `run_ci_seeds.py` | `results/paper_ci/seed_N/` | Per-seed CSVs for E11, E19, E20 |
| `plot_figures.py` | `results/paper_figures/` | `fig01_*.pdf/.png` … `fig61_*.pdf/.png` |
| `analyze_ci.py` | stdout | Bootstrap CIs and Wilcoxon p-values for all key claims |

---

## Reproducing the Paper's Main Claims

**1. Reconstruction dominates cutting overhead.**
Run Stage 1, then `analyze_ci.py`. The "Reconstruction share of cutting
overhead" section reports the mean share and 95% bootstrap CI. Visualised
in fig16 and fig16b.

**2. Distributed execution reduces P90 tail latency.**
Same Stage 1 run. "Main latency result" in `analyze_ci.py` reports the mean
paired P90 reduction, 95% CI, and Wilcoxon p-value. Visualised in fig01,
fig04, fig30.

**3. Plan adoption shifts interpretably under varying conditions.**
fig05 (workload family mix), fig09/fig10 (QPU pool diversity), fig22
(coordination penalty sweep), fig49 (batch vs stream submission strategy).

**4. Scheduler overhead is negligible.**
"Scheduler overhead fraction" section in `analyze_ci.py`. Visualised in
fig02 and fig40.

---

## Utility Scripts

**`summarize_records.py`** — Inspect or compare `records.csv` files directly:

```bash
python summarize_records.py results/experiment_suite/some_run/records.csv
python summarize_records.py run1/records.csv run2/records.csv --compare
```

**`analyze_seed_stats.py`** and **`analyze_e1_ci.py`** — Simpler standalone
alternatives to `analyze_ci.py` for E1-specific spot-checks, useful without
running the full multi-seed pipeline.

---

## Citation

```
[BibTeX to be added after publication]
```