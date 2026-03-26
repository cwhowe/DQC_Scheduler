from __future__ import annotations

import sys
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def _safe_float(x: Any) -> float:
    try:
        if x in (None, "", "None"):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    idx = min(len(xs2) - 1, max(0, int(round((len(xs2) - 1) * q))))
    return float(xs2[idx])


def _jain_index(xs: List[float]) -> float:
    xs = [float(x) for x in xs if float(x) >= 0.0]
    if not xs:
        return 0.0
    s = sum(xs)
    ss = sum(x * x for x in xs)
    if ss == 0.0:
        return 0.0
    return (s * s) / (len(xs) * ss)


def _load_csv(path: Path):
    # Some runs store very large JSON blobs in details_json / metadata_json.
    # Raise the csv parser field limit so DictReader can handle them.
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = max_int // 10

    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _load_details(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(row.get("details_json") or "{}")
    except Exception:
        return {}


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    records = _load_csv(run_dir / "records.csv")
    events = _load_csv(run_dir / "events.csv")
    jobs = _load_csv(run_dir / "job_manifest.csv")

    job_by_id = {row["job_id"]: row for row in jobs}
    plan_counts = Counter(r.get("plan_kind", "UNKNOWN") for r in records)

    latencies = []
    sched_times = []
    part_times = []
    exec_times = []
    recon_times = []
    wall_total_times = []
    queue_waits = []
    exec_spans = []
    fidelity_proxy = []
    fidelity_est = []

    per_job_details: Dict[str, Dict[str, Any]] = {}
    for r in records:
        det = _load_details(r)
        jid = r.get("job_id", "")
        per_job_details[jid] = det
        latencies.append(_safe_float(r.get("end_to_end_s") or det.get("sim_latency_s")))
        sched_times.append(_safe_float(r.get("t_schedule_s") or det.get("schedule_wall_s")))
        part_times.append(_safe_float(r.get("t_partition_s")))
        exec_times.append(_safe_float(r.get("t_execution_s")))
        recon_times.append(_safe_float(r.get("t_reconstruction_s")))
        wall_total_times.append(_safe_float(r.get("wall_end_to_end_s") or det.get("wall_end_to_end_s")))
        queue_waits.append(_safe_float(r.get("sim_queue_wait_s") or det.get("sim_queue_wait_s") or det.get("queue_wait_s")))
        exec_spans.append(_safe_float(r.get("sim_execution_span_s") or det.get("sim_execution_span_s")))
        if r.get("fidelity_proxy") not in (None, "", "None"):
            fidelity_proxy.append(_safe_float(r.get("fidelity_proxy")))
        if r.get("fidelity_estimated") not in (None, "", "None"):
            fidelity_est.append(_safe_float(r.get("fidelity_estimated")))

    makespan = _safe_float(meta.get("sim_completion_time_s") or meta.get("sim_now_s"))
    if makespan <= 0.0 and events:
        makespan = max(_safe_float(e.get("end_s")) for e in events)
    if makespan <= 0.0 and jobs and records:
        makespan = max(_safe_float(r.get("submit_time_s")) + _safe_float(r.get("end_to_end_s")) for r in records)

    qpu_busy = defaultdict(float)
    comm_total = 0.0
    recon_total = 0.0
    kind_counts = Counter()
    family_counts = Counter()
    for e in events:
        kind = str(e.get("kind", ""))
        kind_counts[kind] += 1
        dur = max(0.0, _safe_float(e.get("end_s")) - _safe_float(e.get("start_s")))
        qid = e.get("qpu_id")
        if kind == "quantum" and qid not in (None, "", "None"):
            qpu_busy[str(qid)] += dur
        elif kind == "communication":
            comm_total += dur
        elif kind == "reconstruction":
            recon_total += dur

    for jid in job_by_id:
        fam = job_by_id[jid].get("family", "unknown")
        family_counts[fam] += 1

    total_busy = sum(qpu_busy.values())
    qpu_count = max(1.0, float(len(qpu_busy) if qpu_busy else 3.0))
    utilization = (total_busy / (makespan * qpu_count)) if makespan > 0 else 0.0
    fairness = _jain_index(list(qpu_busy.values()))

    heavy_lat = []
    light_lat = []
    for r in records:
        jid = r.get("job_id", "")
        bucket = job_by_id.get(jid, {}).get("bucket", "")
        lat = _safe_float(r.get("end_to_end_s"))
        if bucket == "heavy":
            heavy_lat.append(lat)
        elif bucket == "light":
            light_lat.append(lat)

    return {
        "run_name": meta["run_name"],
        "suite": meta["suite"],
        "seed": meta["seed"],
        "method": meta["method"]["name"],
        "cut_strategy": meta["method"]["cut_strategy"],
        "workload": meta["workload"]["name"],
        "full_eval": meta["full_eval"],
        "jobs_total": len(jobs),
        "jobs_completed": len(records),
        "completion_rate": (len(records) / len(jobs)) if jobs else 0.0,
        "makespan_s": makespan,
        "lat_mean_s": mean(latencies) if latencies else 0.0,
        "lat_p50_s": _quantile(latencies, 0.5),
        "lat_p95_s": _quantile(latencies, 0.95),
        "lat_heavy_mean_s": mean(heavy_lat) if heavy_lat else 0.0,
        "lat_light_mean_s": mean(light_lat) if light_lat else 0.0,
        "schedule_mean_s": mean(sched_times) if sched_times else 0.0,
        "partition_mean_s": mean(part_times) if part_times else 0.0,
        "exec_mean_s": mean(exec_times) if exec_times else 0.0,
        "recon_mean_s": mean(recon_times) if recon_times else 0.0,
        "queue_wait_mean_s": mean(queue_waits) if queue_waits else 0.0,
        "sim_exec_span_mean_s": mean(exec_spans) if exec_spans else 0.0,
        "wall_end_to_end_mean_s": mean(wall_total_times) if wall_total_times else 0.0,
        "comm_total_s": comm_total,
        "recon_total_s": recon_total,
        "utilization": utilization,
        "fairness_jain": fairness,
        "fidelity_proxy_mean": mean(fidelity_proxy) if fidelity_proxy else 0.0,
        "fidelity_est_mean": mean(fidelity_est) if fidelity_est else 0.0,
        "plan_hist_json": json.dumps(dict(plan_counts), sort_keys=True),
        "qpu_busy_json": json.dumps(dict(qpu_busy), sort_keys=True),
        "family_hist_json": json.dumps(dict(family_counts), sort_keys=True),
        "task_kind_hist_json": json.dumps(dict(kind_counts), sort_keys=True),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize scheduler experiment suite results.")
    ap.add_argument("suite_dir")
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir)
    run_dirs = sorted([p for p in suite_dir.iterdir() if p.is_dir() and (p / "run_meta.json").exists()])
    rows = [analyze_run(run_dir) for run_dir in run_dirs]

    out_csv = suite_dir / "summary_runs.csv"
    with out_csv.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["workload"], row["method"])].append(row)

    agg_rows = []
    numeric_fields = [
        "completion_rate", "makespan_s", "lat_mean_s", "lat_p50_s", "lat_p95_s", "lat_heavy_mean_s", "lat_light_mean_s",
        "schedule_mean_s", "partition_mean_s", "exec_mean_s", "recon_mean_s", "queue_wait_mean_s", "sim_exec_span_mean_s",
        "wall_end_to_end_mean_s", "comm_total_s", "recon_total_s", "utilization", "fairness_jain", "fidelity_proxy_mean", "fidelity_est_mean"
    ]
    for (suite, workload, method), grp in sorted(grouped.items()):
        row = {"suite": suite, "workload": workload, "method": method, "n_seeds": len(grp)}
        for nf in numeric_fields:
            xs = [_safe_float(g.get(nf)) for g in grp]
            row[f"{nf}_mean"] = mean(xs) if xs else 0.0
            row[f"{nf}_std"] = math.sqrt(mean([(x - row[f"{nf}_mean"]) ** 2 for x in xs])) if xs else 0.0
        agg_rows.append(row)

    out_agg = suite_dir / "summary_aggregated.csv"
    with out_agg.open("w", newline="") as f:
        fieldnames = list(agg_rows[0].keys()) if agg_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(agg_rows)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_agg}")


if __name__ == "__main__":
    main()