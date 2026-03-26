# DQC Scheduler Phase 4 Validation Matrix

This matrix is meant to validate when A/B/C plans are selected and how communication and CPU reconstruction contention affect those choices.

## Common baseline flags

Use these in all runs unless a row overrides them:

```bash
QDC_QPU_SOURCE=ibm_fake QDC_QPU_TIMING_MODE=backend_profile QDC_EXCLUDE_QPU_C=0 QDC_FORCE_CUT_WIDE=1 QDC_WIDE_WIDTHS=16 QDC_WIDE_MAX_CUTS=3 QDC_N_JOBS=40 QDC_MAX_TO_SCHEDULE=6 QDC_MAX_STEPS=12000 QDC_STEP_WALL_CAP_S=10
```

## Sweep 1: workload mix / baseline selection

Goal: show that A/B/C selection changes sensibly as the workload becomes more wide-job-heavy.

Run four cases with:

- `QDC_PCT_WIDE=0.25`
- `QDC_PCT_WIDE=0.50`
- `QDC_PCT_WIDE=0.75`
- `QDC_PCT_WIDE=0.90`

Use medium comm / recon settings:

```bash
QDC_CPU_RECON_WORKERS=2 QDC_HOST_RECON_BASE_S=0.5 QDC_HOST_RECON_PER_EXEC_S=0.2 QDC_HOST_RECON_PER_SAMPLE_S=0.01 QDC_COMM_WORKERS=2 QDC_COMM_BASE_S=0.5 QDC_COMM_PER_EXEC_S=0.05 QDC_COMM_PER_SAMPLE_S=0.005 QDC_COMM_COORD_PER_EXTRA_QPU_S=0.1
```

Measure:
- plan histogram
- mean/median `end_to_end_s` by plan kind
- mean `comm_queue_delay_s` and `cpu_queue_delay_s`
- mean `sim_comm_service_s` and `sim_recon_service_s`

## Sweep 2: communication sensitivity

Goal: show that increasing comm cost shifts some jobs from C back toward B when communication becomes expensive.

Keep:
- `QDC_PCT_WIDE=0.90`
- `QDC_CPU_RECON_WORKERS=2`

Run three comm tiers:

Low:
```bash
QDC_COMM_WORKERS=2 QDC_COMM_BASE_S=0.1 QDC_COMM_PER_EXEC_S=0.01 QDC_COMM_PER_SAMPLE_S=0.001 QDC_COMM_COORD_PER_EXTRA_QPU_S=0.02
```

Medium:
```bash
QDC_COMM_WORKERS=2 QDC_COMM_BASE_S=0.5 QDC_COMM_PER_EXEC_S=0.05 QDC_COMM_PER_SAMPLE_S=0.005 QDC_COMM_COORD_PER_EXTRA_QPU_S=0.1
```

High:
```bash
QDC_COMM_WORKERS=2 QDC_COMM_BASE_S=1.0 QDC_COMM_PER_EXEC_S=0.1 QDC_COMM_PER_SAMPLE_S=0.01 QDC_COMM_COORD_PER_EXTRA_QPU_S=0.2
```

Measure:
- fraction of C jobs
- cut-job `comm_queue_delay_s`
- cut-job `end_to_end_s`
- jobs whose selected plan flips between tiers

## Sweep 3: CPU reconstruction sensitivity

Goal: show how host-side reconstruction contention affects end-to-end latency and possibly plan choice.

Keep:
- `QDC_PCT_WIDE=0.90`
- medium comm tier

Run:
- `QDC_CPU_RECON_WORKERS=1`
- `QDC_CPU_RECON_WORKERS=2`
- `QDC_CPU_RECON_WORKERS=4`

Optional heavier recon tier:
```bash
QDC_HOST_RECON_BASE_S=1.0 QDC_HOST_RECON_PER_EXEC_S=0.4 QDC_HOST_RECON_PER_SAMPLE_S=0.02
```

Measure:
- `cpu_queue_delay_s`
- `sim_recon_service_s`
- `end_to_end_s`
- any histogram shifts in A/B/C

## Suggested naming convention

Use output directories like:

- `results/sweep_mix_wide25`
- `results/sweep_mix_wide50`
- `results/sweep_mix_wide75`
- `results/sweep_mix_wide90`
- `results/sweep_comm_low`
- `results/sweep_comm_med`
- `results/sweep_comm_high`
- `results/sweep_cpu_w1`
- `results/sweep_cpu_w2`
- `results/sweep_cpu_w4`

## Example summary commands

Single file:

```bash
python /mnt/data/summarize_records.py results/sweep_comm_med/records.csv
```

Cut jobs only:

```bash
python /mnt/data/summarize_records.py --cut-only results/sweep_comm_med/records.csv
```

Compare two runs directly:

```bash
python /mnt/data/summarize_records.py --cut-only --compare results/sweep_comm_low/records.csv results/sweep_comm_high/records.csv
```
