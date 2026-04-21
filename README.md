# DQC Scheduler — Artifact README

This repository contains the implementation and experiment suite for the paper
**"[Paper Title]"**. It includes the `qdc_sched` Python package (the scheduler
itself), experiment runner scripts, a unified figure generator, and statistical
analysis scripts needed to reproduce the paper's results.

---

## Requirements

- Python >= 3.10
- `qiskit >= 2.0.0`
- `qiskit-aer >= 0.14.0`
- `qiskit-addon-cutting >= 0.10.0`
- `networkx >= 3.0`
- `numpy`, `matplotlib`, `pandas`, `scipy`
- *(Optional)* `qiskit-ibm-runtime >= 0.24.0` for IBM-specific backend extensions

No physical quantum hardware is required. All experiments run on classical
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
python -c "import qdc_sched; print('OK')"
```

---

## Repository Structure

```
DQC_Scheduler/
├── qdc_sched/                    # Scheduler package (core library)
│   ├── core/                     # Scheduler, planner, executor, hardware, metrics
│   └── cutting/                  # Circuit-cutting strategies (FitCut, etc.)
│
├── run_suite.py                  # Main experiment suite — E1–E16 equivalent
├── run_experiments_supplementary.py            # Supplementary experiments — E17–E24
├── run_ci_seeds.py               # Repeated-seed runner for CI analysis
│
├── plot_figures.py               # Unified figure generator — all paper figures
│
├── analyze_ci.py                 # Bootstrap CIs + Wilcoxon tests (primary)
├── analyze_experiment_suite.py   # Aggregate suite run dirs into summary CSVs
├── analyze_seed_stats.py         # E1-focused CI analysis (supplementary)
├── analyze_e1_ci.py              # Minimal E1-only CI script (supplementary)
├── summarize_records.py          # Inspect individual records.csv files
│
└── results/                      # Created at runtime; not committed
```

---

## Running the Experiments

### Quick smoke test

Add `--fast` to any runner to verify everything is working before committing
to a full run. This reduces job counts and sweep sizes substantially and
completes in well under 30 minutes:

```bash
python run_suite.py --suite all --seeds 2026 --outdir results/experiment_suite --fast
python run_experiments_supplementary.py --seed 2026 --outdir results/experiments --fast
```

To check API compatibility for the supplementary experiments specifically:

```bash
python run_experiments_supplementary.py --diagnose
```

---

### Stage 1 — Main experiment suite (E1–E16 equivalent)

`run_suite.py` is the primary runner. It covers plan comparison, width sweep,
congestion, coordination penalty, SLO constraints, QPU pool scaling, backend
comparison, noise sensitivity, and related experiments across configurable
workload presets and methods.

```bash
python run_suite.py \
    --suite all \
    --seeds 2026 \
    --outdir results/experiment_suite
```

**Available suites:**

| Suite | Description |
|-------|-------------|
| `system_value` | Plan A/B/C comparison on fit workloads |
| `forcecut_value` | Forced-cut workloads stressing Plan B and C |
| `cut_method` | FitCut vs Addon vs Naive cutting strategies |
| `quality_small` | Full-evaluation quality experiment (slow) |
| `paper_main` | The two primary paper workloads |
| `all` | All suites above |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--suite` | `all` | Which suite(s) to run |
| `--seeds` | `2026` | Comma-separated random seeds |
| `--outdir` | `results/experiment_suite` | Output root directory |
| `--fast` | off | Reduced job counts for quick testing |

After the suite finishes, aggregate the per-run outputs:

```bash
python analyze_experiment_suite.py results/experiment_suite
```

This writes `summary_runs.csv` and `summary_aggregated.csv` into the suite
directory.

---

### Stage 2 — Supplementary experiments (E17–E24)

`run_experiments_supplementary.py` handles all supplementary experiments in one script.
Run any combination with `--experiments`:

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
| E17 | Congestion × arrival-rate joint sweep (4×3 grid) | `e17_congestion_arrival.csv` |
| E18 | Utilisation–throughput Pareto frontier (pool × λ grid) | `e18_utilization_pareto.csv` |
| E19 | Job-stream composition sweep (heavy fraction 0→100%) | `e19_stream_composition.csv` |
| E20 | Batch vs stream: 7 submission strategies compared | `e20_batch_stream.csv` |
| E21 | Throughput scaling vs job load (heterogeneous 3-QPU pool) | `e21_throughput_scaling.json` |
| E24 | QPU idle fraction vs arrival rate (homogeneous 3-QPU pool) | `e24_idle_fraction.json` |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--experiments` | all | Comma-separated subset, e.g. `E19,E20` |
| `--n-jobs` | `40` | Jobs per condition |
| `--seed` | `2026` | Random seed |
| `--outdir` | `results/experiments` | Output directory |
| `--fast` | off | Reduced sweep sizes |
| `--diagnose` | off | Check API compatibility and exit |

---

### Stage 3 — Repeated-seed runs for confidence intervals

`run_ci_seeds.py` reruns E11, E19, and E20 across multiple seeds and produces
the `seed_N/` directories consumed by `analyze_ci.py`:

```bash
python run_ci_seeds.py \
    --outdir results/paper_ci \
    --seeds 2026,2027,2028,2029,2030 \
    --n-jobs 40
```

The full paper used seeds 2026–2034 (9 seeds). Use `--fast` for a quick check
with reduced job counts. The `--experiments` flag accepts any subset of
`E11,E19,E20`.

---

### Stage 4 — Generate figures

All paper figures are generated by a single script. It silently skips any
figure whose required input data file is not found:

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

To apply post-review style and annotation patches:

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
| `--indir` | `results/experiments` | Primary input directory (E1–E16 CSVs) |
| `--indir2` | same as `--indir` | Secondary input directory (E17–E24 CSVs/JSONs) |
| `--outdir` | `results/paper_figures` | Output directory |
| `--figures` | `all` | Comma-separated figure numbers, or `all` |
| `--style` | `paper` | Font preset: `paper` or `slides` |
| `--patch` | off | Apply post-review patches to selected figures |

---

### Stage 5 — Statistical analysis

`analyze_ci.py` computes bootstrap confidence intervals and Wilcoxon
signed-rank tests for all key quantitative claims. It requires the
repeated-seed directories from Stage 3:

```bash
python analyze_ci.py
```

Configure the `E1_ROOT`, `E11_ROOT`, `E19_ROOT`, and `E20_ROOT` paths at the
top of the script to point at your output directories. It reports:

1. Main latency result — no-cut vs cut-multi P90 (E1)
2. Scheduler overhead fraction of end-to-end latency (E1)
3. Reconstruction share of cutting overhead (E1)
4. Batch vs stream P90 latency comparison (E20)
5. SLO completion rate — no-SLO vs SLO-constrained (E11)
6. Stream composition sweep — low-heavy vs high-heavy P90 (E19)

---

## Expected Outputs

| Script | Output location | Contents |
|--------|----------------|----------|
| `run_suite.py` | `results/experiments/` (flat CSVs) and `results/experiment_suite/*/` (per-run dirs) | `records.csv`, `events.csv`, `job_manifest.csv`, `run_meta.json` per run |
| `analyze_experiment_suite.py` | `results/experiment_suite/` | `summary_runs.csv`, `summary_aggregated.csv` |
| `run_experiments_supplementary.py` | `results/experiments/` | `e17_*.csv`, `e18_*.csv`, `e19_*.csv`, `e20_*.csv`, `e21_*.json`, `e24_*.json` |
| `run_ci_seeds.py` | `results/paper_ci/seed_N/` | `e11_*.csv`, `e19_*.csv`, `e20_*.csv` per seed |
| `plot_figures.py` | `results/paper_figures/` | `fig01_*.pdf/.png` … `fig61_*.pdf/.png` |
| `analyze_ci.py` | stdout | Bootstrap CIs and Wilcoxon p-values for all key claims |

---

## Reproducing the Paper's Main Claims

**1. Reconstruction dominates cutting overhead.**
Run Stage 1, then `analyze_ci.py`. See "Reconstruction share of cutting
overhead" in the output. Visualised in fig16 and fig16b.

**2. Distributed execution reduces P90 tail latency.**
Same Stage 1 run. "Main latency result" in `analyze_ci.py` output reports
the mean paired P90 reduction, 95% bootstrap CI, and Wilcoxon p-value.
Visualised in fig01, fig04, fig30.

**3. Plan adoption shifts interpretably under varying conditions.**
Inspect fig05 (workload mix), fig09/fig10 (QPU diversity), fig22 (coordination
penalty), fig49 (batch vs stream), all from `plot_figures.py`.

**4. Scheduler overhead is negligible.**
"Scheduler overhead fraction" section in `analyze_ci.py`. Visualised in
fig02 and fig40.

---

## Utility Scripts

**`summarize_records.py`** — Inspect any `records.csv` directly:

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
[BibTeX entry to be added after publication]
```