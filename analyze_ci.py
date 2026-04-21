"""analyze_ci.py — Bootstrap CIs and Wilcoxon tests for key paper claims.

Analyses covered
----------------
1. Main latency result          (E1)  P90 no_cut vs cut_multi        — CI + p-value
2. Scheduler overhead fraction  (E1)  schedule_wall_s / e2e           — CI only
3. Reconstruction share         (E1)  analytic recon / (recon + comm) — CI only
4. Batch vs stream              (E20) P90 batch_all vs stream_medium  — CI + p-value
5. SLO completion rate          (E11) no_slo vs slo_05s completion %  — CI + p-value
6. Stream composition sweep     (E19) P90 low-heavy vs high-heavy     — CI + p-value

Usage
-----
    python analyze_ci.py

Configure the ROOT paths below to point at your repeated-seed directories.
Each seed lives in a subdirectory named seed_N/, e.g.:
    results/paper_ci/seed_0/e1_plan_comparison.csv
    results/paper_ci/seed_1/e1_plan_comparison.csv
    ...
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------------
# Configure paths
# ---------------------------------------------------------------------------

E1_ROOT   = Path("results/paper_ci")       # seed_*/e1_plan_comparison.csv
E11_ROOT  = Path("results/paper_ci")       # seed_*/e11_slo_constrained.csv
E19_ROOT  = Path("results/paper_ci")       # seed_*/e19_stream_composition.csv
E20_ROOT  = Path("results/batch_ci")       # seed_*/e20_batch_stream.csv

E1_FILE   = "e1_plan_comparison.csv"
E11_FILE  = "e11_slo_constrained.csv"
E19_FILE  = "e19_stream_composition.csv"
E20_FILE  = "e20_batch_stream.csv"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values, stat_fn=np.mean, n_boot=10_000, ci=95, seed=1234):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    boots = [stat_fn(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    alpha = (100 - ci) / 2
    return float(np.percentile(boots, alpha)), float(np.percentile(boots, 100 - alpha))


def extract_seed(path: Path) -> int | None:
    m = re.search(r"seed_(\d+)", str(path))
    return int(m.group(1)) if m else None


def _header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def _report_ci(label, values, pct=False):
    values = np.asarray(values, dtype=float)
    est = float(np.mean(values))
    lo, hi = bootstrap_ci(values)
    if pct:
        print(f"  {label}: {100*est:.2f}%  (95% CI [{100*lo:.2f}%, {100*hi:.2f}%])")
    else:
        print(f"  {label}: {est:.4f} s  (95% CI [{lo:.4f}, {hi:.4f}] s)")


def _report_wilcoxon(a_vals, b_vals, label_a, label_b, alternative="greater"):
    if len(a_vals) < 2:
        print(f"  Wilcoxon: insufficient data (n={len(a_vals)})")
        return float("nan")
    stat, pval = wilcoxon(a_vals, b_vals, alternative=alternative)
    print(f"  Wilcoxon signed-rank ({label_a} {alternative} {label_b}): "
          f"stat={stat:.1f}, p={pval:.2e}")
    return float(pval)


def _load_seeds(root: Path, filename: str) -> list[tuple[int, pd.DataFrame]]:
    """Return [(seed, df), ...] for all matching seed directories."""
    out = []
    for p in sorted(root.glob(f"seed_*/{filename}")):
        seed = extract_seed(p)
        try:
            out.append((seed, pd.read_csv(p)))
        except Exception as e:
            print(f"  [WARN] could not read {p}: {e}")
    if not out:
        print(f"  [SKIP] no files found at {root}/seed_*/{filename}")
    return out


# ---------------------------------------------------------------------------
# 1. Main latency result: no_cut vs cut_multi P90 (E1)
# ---------------------------------------------------------------------------

def analyze_e1_main_latency(root: Path = E1_ROOT, filename: str = E1_FILE):
    _header("1. Main latency result — no_cut vs cut_multi P90 (E1)")

    rows = []
    for seed, df in _load_seeds(root, filename):
        sub = df[df["condition"].isin(["no_cut", "cut_multi"])]
        if sub.empty:
            continue
        for cond, g in sub.groupby("condition"):
            rows.append(dict(seed=seed, condition=cond,
                             p90=float(g["end_to_end_s"].quantile(0.9))))

    if not rows:
        print("  No data.")
        return None

    pivot = (pd.DataFrame(rows)
             .pivot(index="seed", columns="condition", values="p90")
             .dropna())
    print(f"  Seeds: {len(pivot)}")

    pivot["diff_s"]  = pivot["no_cut"] - pivot["cut_multi"]
    pivot["pct_red"] = 100.0 * pivot["diff_s"] / pivot["no_cut"]

    _report_ci("P90 reduction (s)",  pivot["diff_s"].values)
    _report_ci("P90 reduction (%)",  pivot["pct_red"].values, pct=False)
    _report_wilcoxon(pivot["no_cut"].values, pivot["cut_multi"].values,
                     "no_cut", "cut_multi", alternative="greater")

    return pivot


# ---------------------------------------------------------------------------
# 2. Scheduler overhead fraction (E1, cut_multi)
# ---------------------------------------------------------------------------

def analyze_scheduler_overhead(root: Path = E1_ROOT, filename: str = E1_FILE,
                                condition: str = "cut_multi"):
    _header("2. Scheduler overhead fraction of e2e latency (E1)")

    rows = []
    for seed, df in _load_seeds(root, filename):
        sub = df[df["condition"] == condition].copy()
        if sub.empty:
            continue

        sched_col = next((c for c in ["schedule_wall_s", "t_schedule_s"]
                          if c in sub.columns), None)
        if sched_col is None:
            print(f"  [WARN] seed {seed}: no scheduler-time column found")
            continue

        frac = (sub[sched_col].fillna(0) / sub["end_to_end_s"]
                ).replace([np.inf, -np.inf], np.nan).dropna()
        if not frac.empty:
            rows.append(dict(seed=seed, mean_frac=float(frac.mean()),
                             max_frac=float(frac.max())))

    if not rows:
        print("  No data.")
        return None

    out = pd.DataFrame(rows)
    print(f"  Seeds: {len(out)}   Condition: {condition}")
    _report_ci("Mean overhead fraction", out["mean_frac"].values, pct=True)
    print(f"  Max per-job fraction seen: {100*out['max_frac'].max():.4f}%")
    return out


# ---------------------------------------------------------------------------
# 3. Reconstruction share of cutting overhead (E1, cut_multi)
#
#    Reconstruction is NOT stored as a column — it is computed analytically
#    the same way the paper figures do:
#        recon_s = 0.005 + 0.002 * sampling_overhead   (if sampling_overhead > 0)
#    Communication uses charged_comm_s (actual per-job comm cost).
# ---------------------------------------------------------------------------

def analyze_reconstruction_share(root: Path = E1_ROOT, filename: str = E1_FILE,
                                  condition: str = "cut_multi"):
    _header("3. Reconstruction share of cutting overhead (E1)")

    def _recon(row):
        samp = float(row.get("sampling_overhead", 0) or 0)
        return 0.005 + 0.002 * samp if samp > 0 else 0.0

    comm_candidates = ["charged_comm_s", "sim_comm_s", "model_comm_s", "pred_comm_s"]

    rows = []
    for seed, df in _load_seeds(root, filename):
        sub = df[df["condition"] == condition].copy()
        if sub.empty:
            continue

        # Analytic reconstruction per job
        sub["recon_s"] = sub.apply(_recon, axis=1)

        # Best available communication column
        comm_col = next((c for c in comm_candidates if c in sub.columns
                         and sub[c].fillna(0).sum() > 0), None)
        if comm_col is None:
            print(f"  [WARN] seed {seed}: no communication column found — skipping")
            continue

        sub["comm_s"] = pd.to_numeric(sub[comm_col], errors="coerce").fillna(0)
        denom = sub["recon_s"] + sub["comm_s"]

        # Per-job share, then median across jobs (avoid 0/0)
        valid = denom > 0
        if not valid.any():
            print(f"  [WARN] seed {seed}: recon+comm == 0 for all rows")
            continue

        share = float((sub.loc[valid, "recon_s"] / denom[valid]).mean())
        rows.append(dict(seed=seed, comm_col=comm_col,
                         recon_total=float(sub["recon_s"].sum()),
                         comm_total=float(sub["comm_s"].sum()),
                         share=share))

    if not rows:
        print("  No data.")
        return None

    out = pd.DataFrame(rows)
    print(f"  Seeds: {len(out)}   Condition: {condition}")
    print(f"  Communication column used: {out['comm_col'].iloc[0]}")
    _report_ci("Mean recon share", out["share"].values, pct=True)
    return out


# ---------------------------------------------------------------------------
# 4. Batch vs stream P90 latency (E20)
# ---------------------------------------------------------------------------

def analyze_batch_vs_stream(root: Path = E20_ROOT, filename: str = E20_FILE,
                             stream_cond: str = "stream_medium",
                             batch_cond:  str = "batch_all"):
    _header(f"4. Batch vs stream — P90 latency: {batch_cond} vs {stream_cond} (E20)")

    rows = []
    for seed, df in _load_seeds(root, filename):
        sub = df[df["condition"].isin([stream_cond, batch_cond])]
        if sub.empty:
            continue
        for cond, g in sub.groupby("condition"):
            rows.append(dict(seed=seed, condition=cond,
                             p90=float(g["end_to_end_s"].quantile(0.9))))

    if not rows:
        print("  No data.")
        return None

    pivot = (pd.DataFrame(rows)
             .pivot(index="seed", columns="condition", values="p90")
             .dropna())
    print(f"  Seeds: {len(pivot)}")

    pivot["diff_s"]    = pivot[batch_cond] - pivot[stream_cond]
    pivot["pct_incr"]  = 100.0 * pivot["diff_s"] / pivot[stream_cond]

    _report_ci("P90 increase batch over stream (s)", pivot["diff_s"].values)
    _report_ci("P90 increase (%)", pivot["pct_incr"].values, pct=False)
    _report_wilcoxon(pivot[batch_cond].values, pivot[stream_cond].values,
                     batch_cond, stream_cond, alternative="greater")

    return pivot


# ---------------------------------------------------------------------------
# 5. SLO completion rate: no_slo vs slo_05s (E11)
# ---------------------------------------------------------------------------

def analyze_slo_completion(root: Path = E11_ROOT, filename: str = E11_FILE):
    _header("5. SLO completion rate — no_slo vs slo_05s (E11)")

    rows = []
    for seed, df in _load_seeds(root, filename):
        sub = df[df["condition"].isin(["no_slo", "slo_05s"])]
        if sub.empty:
            continue
        for cond, g in sub.groupby("condition"):
            # Completion rate = jobs with valid e2e > 0 / total rows for condition
            n_total = len(g)
            n_done  = int((g["end_to_end_s"].fillna(0) > 0).sum())
            rows.append(dict(seed=seed, condition=cond,
                             completion_rate=n_done / max(n_total, 1)))

    if not rows:
        print("  No data.")
        return None

    pivot = (pd.DataFrame(rows)
             .pivot(index="seed", columns="condition", values="completion_rate")
             .dropna())
    print(f"  Seeds: {len(pivot)}")

    _report_ci("Completion rate — no_slo",  pivot["no_slo"].values,  pct=True)
    _report_ci("Completion rate — slo_05s", pivot["slo_05s"].values, pct=True)

    pivot["diff"] = pivot["no_slo"] - pivot["slo_05s"]
    _report_ci("Difference (no_slo − slo_05s)", pivot["diff"].values, pct=True)
    _report_wilcoxon(pivot["no_slo"].values, pivot["slo_05s"].values,
                     "no_slo", "slo_05s", alternative="greater")

    return pivot


# ---------------------------------------------------------------------------
# 6. Stream composition sweep: low-heavy vs high-heavy P90 (E19)
# ---------------------------------------------------------------------------

def analyze_stream_composition(root: Path = E19_ROOT, filename: str = E19_FILE,
                                low_cond:  str = "heavy_00pct",
                                high_cond: str = "heavy_100pct"):
    _header(f"6. Stream composition — P90 latency: {high_cond} vs {low_cond} (E19)")

    rows = []
    for seed, df in _load_seeds(root, filename):
        sub = df[df["condition"].isin([low_cond, high_cond])]
        if sub.empty:
            continue
        for cond, g in sub.groupby("condition"):
            e2e = g["end_to_end_s"].dropna()
            e2e = e2e[e2e > 0]
            if e2e.empty:
                continue
            rows.append(dict(seed=seed, condition=cond,
                             p90=float(e2e.quantile(0.9)),
                             fid_med=float(g["fidelity_proxy"].dropna().median())
                                     if "fidelity_proxy" in g.columns else float("nan")))

    if not rows:
        print("  No data.")
        return None

    pivot_p90 = (pd.DataFrame(rows)
                 .pivot(index="seed", columns="condition", values="p90")
                 .dropna())
    pivot_fid = (pd.DataFrame(rows)
                 .pivot(index="seed", columns="condition", values="fid_med")
                 .dropna())

    print(f"  Seeds (P90): {len(pivot_p90)}")
    pivot_p90["diff_s"]   = pivot_p90[high_cond] - pivot_p90[low_cond]
    pivot_p90["pct_incr"] = 100.0 * pivot_p90["diff_s"] / pivot_p90[low_cond]

    _report_ci("P90 increase high- vs low-heavy (s)", pivot_p90["diff_s"].values)
    _report_ci("P90 increase (%)", pivot_p90["pct_incr"].values, pct=False)
    _report_wilcoxon(pivot_p90[high_cond].values, pivot_p90[low_cond].values,
                     high_cond, low_cond, alternative="greater")

    if not pivot_fid.empty and not pivot_fid[high_cond].isna().all():
        print(f"\n  Seeds (fidelity): {len(pivot_fid)}")
        pivot_fid["fid_drop"] = pivot_fid[low_cond] - pivot_fid[high_cond]
        _report_ci("Fidelity drop (low − high heavy)", pivot_fid["fid_drop"].values)

    return pivot_p90


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    analyze_e1_main_latency()
    analyze_scheduler_overhead()
    analyze_reconstruction_share()
    analyze_batch_vs_stream()
    analyze_slo_completion()
    analyze_stream_composition()
    print("\n\nDone.")


if __name__ == "__main__":
    main()
