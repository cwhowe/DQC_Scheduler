#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

NUMERIC_FIELDS = [
    "end_to_end_s",
    "t_execution_s",
    "t_reconstruction_s",
    "comm_queue_delay_s",
    "comm_busy_time_s",
    "cpu_queue_delay_s",
    "cpu_busy_time_s",
    "sim_comm_queue_s",
    "sim_comm_service_s",
    "sim_comm_s",
    "sim_recon_queue_s",
    "sim_recon_service_s",
    "sim_recon_s",
    "pred_comm_queue_s",
    "pred_comm_service_s",
    "pred_comm_s",
]

PLAN_ORDER = ["A_NO_CUT_SINGLE", "B_CUT_SINGLE_SEQ", "C_CUT_MULTI_QPU"]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def load_details(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("details_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def first_present(*values: Any) -> Any:
    for v in values:
        if v is not None:
            if isinstance(v, str) and not v.strip():
                continue
            return v
    return None


def collect_num(row: Dict[str, Any], details: Dict[str, Any], key: str) -> Optional[float]:
    top = safe_float(row.get(key))
    if top is not None:
        return top
    return safe_float(details.get(key))


def collect_str(row: Dict[str, Any], details: Dict[str, Any], *keys: str) -> Optional[str]:
    for k in keys:
        val = first_present(row.get(k), details.get(k))
        if val is not None:
            return str(val)
    return None


@dataclass
class Record:
    source: str
    job_id: str
    plan_kind: str
    qpu_id: Optional[str]
    row: Dict[str, Any]
    details: Dict[str, Any]
    nums: Dict[str, Optional[float]]



def load_records(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            details = load_details(row)
            plan_kind = collect_str(row, details, "plan_kind", "plan") or "UNKNOWN"
            qpu_id = collect_str(row, details, "qpu_id", "primary_qpu_id")
            nums = {k: collect_num(row, details, k) for k in NUMERIC_FIELDS}
            records.append(
                Record(
                    source=str(path),
                    job_id=str(row.get("job_id", details.get("job_id", ""))),
                    plan_kind=plan_kind,
                    qpu_id=qpu_id,
                    row=row,
                    details=details,
                    nums=nums,
                )
            )
    return records



def summarize_values(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "max": max(values),
    }



def fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.6f}"



def plan_sort_key(plan: str):
    return (PLAN_ORDER.index(plan) if plan in PLAN_ORDER else 999, plan)



def print_group_summary(records: List[Record], title: str) -> None:
    print(f"\n=== {title} ===")
    total = len(records)
    print(f"jobs: {total}")
    counts: Dict[str, int] = {}
    for r in records:
        counts[r.plan_kind] = counts.get(r.plan_kind, 0) + 1
    print("plan histogram:")
    for plan in sorted(counts, key=plan_sort_key):
        n = counts[plan]
        pct = 100.0 * n / max(1, total)
        print(f"  {plan:20s} {n:4d}  ({pct:5.1f}%)")

    metrics_to_show = [
        "end_to_end_s",
        "t_execution_s",
        "sim_comm_service_s",
        "comm_queue_delay_s",
        "sim_recon_service_s",
        "cpu_queue_delay_s",
        "pred_comm_service_s",
        "pred_comm_queue_s",
    ]
    print("\nper-plan metrics (mean / median / max):")
    header = f"{'plan_kind':20s} {'metric':22s} {'mean':>12s} {'median':>12s} {'max':>12s}"
    print(header)
    print("-" * len(header))
    for plan in sorted(counts, key=plan_sort_key):
        plan_recs = [r for r in records if r.plan_kind == plan]
        for metric in metrics_to_show:
            vals = [r.nums[metric] for r in plan_recs if r.nums.get(metric) is not None]
            stats = summarize_values([v for v in vals if v is not None])
            print(f"{plan:20s} {metric:22s} {fmt(stats['mean']):>12s} {fmt(stats['median']):>12s} {fmt(stats['max']):>12s}")



def compare_runs(left: List[Record], right: List[Record], left_name: str, right_name: str, cut_only: bool) -> None:
    if cut_only:
        left = [r for r in left if r.plan_kind != "A_NO_CUT_SINGLE"]
        right = [r for r in right if r.plan_kind != "A_NO_CUT_SINGLE"]

    print(f"\n=== Comparison: {left_name} vs {right_name}{' (cut jobs only)' if cut_only else ''} ===")
    metrics = [
        "end_to_end_s",
        "comm_queue_delay_s",
        "sim_comm_service_s",
        "cpu_queue_delay_s",
        "sim_recon_service_s",
        "pred_comm_service_s",
        "pred_comm_queue_s",
    ]
    header = f"{'metric':22s} {left_name:>14s} {right_name:>14s} {'delta(right-left)':>18s}"
    print(header)
    print("-" * len(header))
    for metric in metrics:
        lv = [r.nums[metric] for r in left if r.nums.get(metric) is not None]
        rv = [r.nums[metric] for r in right if r.nums.get(metric) is not None]
        lmean = mean(lv) if lv else None
        rmean = mean(rv) if rv else None
        delta = None if lmean is None or rmean is None else (rmean - lmean)
        print(f"{metric:22s} {fmt(lmean):>14s} {fmt(rmean):>14s} {fmt(delta):>18s}")



def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize DQC Scheduler records.csv outputs.")
    ap.add_argument("paths", nargs="+", help="One or more records.csv files")
    ap.add_argument("--cut-only", action="store_true", help="Filter to non-A plans for the main summaries")
    ap.add_argument("--compare", action="store_true", help="If exactly two files are provided, print a direct comparison table")
    args = ap.parse_args()

    loaded: List[tuple[str, List[Record]]] = []
    for p in args.paths:
        path = Path(p)
        recs = load_records(path)
        if args.cut_only:
            recs = [r for r in recs if r.plan_kind != "A_NO_CUT_SINGLE"]
        loaded.append((path.stem, recs))

    for name, recs in loaded:
        print_group_summary(recs, name)

    if args.compare and len(loaded) == 2:
        compare_runs(loaded[0][1], loaded[1][1], loaded[0][0], loaded[1][0], args.cut_only)


if __name__ == "__main__":
    main()
