"""compare_runs.py — summarise and diff two scheduler result CSVs.

Usage:
    python3 compare_runs.py results/baseline/records.csv results/pandora/records.csv
    python3 compare_runs.py results/demo_workload/records.csv  # single-run summary only
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def summarise(rows: list[dict], label: str) -> dict:
    plan_counts: dict[str, int] = defaultdict(int)
    qpu_jobs: dict[str, int] = defaultdict(int)        # qpu_id -> job count
    qpu_by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    e2e: list[float] = []
    fid: list[float] = []
    exec_t: list[float] = []
    cut_e2e: list[float] = []
    cut_exec: list[float] = []

    for r in rows:
        kind = r.get("plan_kind", "?")
        qpu = r.get("qpu_id") or "multi"
        plan_counts[kind] += 1
        qpu_jobs[qpu] += 1
        qpu_by_kind[kind][qpu] += 1
        e2e.append(_f(r.get("end_to_end_s")))
        fid.append(_f(r.get("fidelity_proxy")))
        exec_t.append(_f(r.get("t_execution_s")))
        if kind in ("B_CUT_SINGLE_SEQ", "C_CUT_MULTI_QPU"):
            cut_e2e.append(_f(r.get("end_to_end_s")))
            cut_exec.append(_f(r.get("t_execution_s")))

    def avg(lst):
        valid = [x for x in lst if x == x]  # drop NaN
        return sum(valid) / len(valid) if valid else float("nan")

    def pct(kind):
        return 100.0 * plan_counts.get(kind, 0) / len(rows) if rows else 0.0

    print(f"\n{'─'*50}")
    print(f"  {label}  ({len(rows)} jobs)")
    print(f"{'─'*50}")
    print(f"  Plan mix:")
    for kind in ("A_NO_CUT_SINGLE", "B_CUT_SINGLE_SEQ", "C_CUT_MULTI_QPU", "D_WAIT"):
        n = plan_counts.get(kind, 0)
        if n or kind != "D_WAIT":
            routing = dict(qpu_by_kind.get(kind, {}))
            routing_str = "  " + str(routing) if routing else ""
            print(f"    {kind:<22s}  {n:3d}  ({pct(kind):5.1f}%){routing_str}")
    print(f"  QPU utilisation (jobs routed):")
    for qpu in sorted(qpu_jobs):
        print(f"    {qpu:<12s}  {qpu_jobs[qpu]:3d}")
    print(f"  End-to-end latency (all jobs):  avg={avg(e2e):.3f}s")
    print(f"  Execution time     (all jobs):  avg={avg(exec_t):.3f}s")
    print(f"  Fidelity proxy     (all jobs):  avg={avg(fid):.4f}")
    if cut_e2e:
        print(f"  Cut jobs only:")
        print(f"    e2e avg={avg(cut_e2e):.3f}s   exec avg={avg(cut_exec):.3f}s   n={len(cut_e2e)}")

    return {
        "plan_counts": dict(plan_counts),
        "qpu_jobs": dict(qpu_jobs),
        "avg_e2e": avg(e2e),
        "avg_exec": avg(exec_t),
        "avg_fid": avg(fid),
        "avg_cut_e2e": avg(cut_e2e),
        "avg_cut_exec": avg(cut_exec),
        "n_cut": len(cut_e2e),
        "n": len(rows),
    }


def diff(a: dict, b: dict, label_a: str, label_b: str) -> None:
    def delta(key, fmt=".3f", pct=False):
        va, vb = a[key], b[key]
        if va != va or vb != vb:
            return "n/a"
        d = vb - va
        sign = "+" if d >= 0 else ""
        if pct:
            rel = 100.0 * d / va if va else float("nan")
            return f"{sign}{d:{fmt}}  ({sign}{rel:.1f}%)"
        return f"{sign}{d:{fmt}}"

    print(f"\n{'═'*50}")
    print(f"  DIFF  {label_a}  →  {label_b}")
    print(f"{'═'*50}")
    print(f"  avg end-to-end latency:  {delta('avg_e2e', pct=True)}")
    print(f"  avg execution time:      {delta('avg_exec', pct=True)}")
    print(f"  avg fidelity proxy:      {delta('avg_fid', '.4f')}")
    if a["n_cut"] or b["n_cut"]:
        print(f"  cut-job e2e avg:         {delta('avg_cut_e2e', pct=True)}")
        print(f"  cut-job exec avg:        {delta('avg_cut_exec', pct=True)}")
    print(f"  plan mix changes:")
    all_kinds = set(a["plan_counts"]) | set(b["plan_counts"])
    for kind in sorted(all_kinds):
        na = a["plan_counts"].get(kind, 0)
        nb = b["plan_counts"].get(kind, 0)
        d = nb - na
        tag = f"  (+{d})" if d > 0 else f"  ({d})" if d < 0 else "  (unchanged)"
        print(f"    {kind:<22s}  {na} → {nb}{tag}")
    print(f"  QPU routing changes:")
    all_qpus = set(a["qpu_jobs"]) | set(b["qpu_jobs"])
    for qpu in sorted(all_qpus):
        na = a["qpu_jobs"].get(qpu, 0)
        nb = b["qpu_jobs"].get(qpu, 0)
        d = nb - na
        tag = f"  (+{d})" if d > 0 else f"  ({d})" if d < 0 else "  (unchanged)"
        print(f"    {qpu:<12s}  {na} → {nb}{tag}")


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 compare_runs.py <baseline.csv> [pandora.csv]")
        sys.exit(1)

    a = load(args[0])
    sa = summarise(a, Path(args[0]).parent.name + "/" + Path(args[0]).name)

    if len(args) >= 2:
        b = load(args[1])
        sb = summarise(b, Path(args[1]).parent.name + "/" + Path(args[1]).name)
        diff(sa, sb, Path(args[0]).parent.name, Path(args[1]).parent.name)


if __name__ == "__main__":
    main()
