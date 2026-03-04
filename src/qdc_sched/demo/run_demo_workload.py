from __future__ import annotations

import os
import random
import time
import csv
import json
from collections import Counter
from typing import List, Tuple, Dict, Callable, Set

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qdc_sched.core.hardware import HardwareProfile, QPUState
from qdc_sched.core.quality import QualityModel
from qdc_sched.core.scheduler import Scheduler, SchedulerConfig
from qdc_sched.core.executor import ExecConfig
from qdc_sched.core.types import Job, RunToggles, JobConstraints


def ghz(n: int, measure: bool = True) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    if measure:
        qc.measure_all()
    return qc


def random_cx(n: int, depth: int, measure: bool = True) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    rng = random.Random(1234 + 17 * n + depth)
    for _ in range(depth):
        a = rng.randrange(n)
        b = rng.randrange(n)
        while b == a:
            b = rng.randrange(n)
        qc.cx(a, b)
        qc.h(a)
    if measure:
        qc.measure_all()
    return qc


def z_observable(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_list([("Z" * n, 1.0)])


def make_line_qpu(qpu_id: str, n: int, base_queue_delay_s: float) -> QPUState:
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(i, i + 1) for i in range(n - 1)])
    hp = HardwareProfile(
        qpu_id=qpu_id,
        num_qubits=n,
        coupling_graph=g,
        base_queue_delay_s=base_queue_delay_s,
    )
    return QPUState(hp)


def _parse_int_list(env_name: str, default_csv: str) -> List[int]:
    raw = os.getenv(env_name, default_csv)
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def make_workload(
    n_jobs: int | None = None,
    pct_expect: float | None = None,
    pct_wide: float | None = None,
) -> Tuple[List[Tuple[float, Job]], Set[str]]:
    """Generate a mixed workload.

    - Counts jobs (random CX circuits) always fit.
    - Expectation jobs use GHZ without measurement + Z^{\otimes n} observable.
    - A small fraction are "wide slots" which can exceed any single QPU (to exercise cutting).
    """
    if n_jobs is None:
        n_jobs = int(os.getenv("QDC_N_JOBS", "160"))
    if pct_expect is None:
        pct_expect = float(os.getenv("QDC_PCT_EXPECT", "0.40"))
    if pct_wide is None:
        pct_wide = float(os.getenv("QDC_PCT_WIDE", "0.15"))

    wl: List[Tuple[float, Job]] = []
    wide_ids: Set[str] = set()
    t = 0.0
    rng = random.Random(2026)

    n_wide = int(round(n_jobs * pct_wide))
    wide_slots = set(rng.sample(range(n_jobs), k=n_wide)) if n_wide > 0 else set()

    wide_widths = _parse_int_list("QDC_WIDE_WIDTHS", "10,12,14,16")

    for i in range(n_jobs):
        t += rng.choice([0.0, 0.1, 0.2, 0.5, 1.0])
        jid = f"J{i:03d}"

        # Wide-slot jobs: always expectation; can exceed any single QPU to ensure cutting paths get exercised
        if i in wide_slots:
            width = rng.choice(wide_widths)
            shots = rng.choice([100, 200, 500])
            qc = ghz(width, measure=False)
            job = Job(
                job_id=jid,
                circuit=qc,
                task_type="expectation",
                observables=z_observable(width),
                shots=shots,
                submit_time_s=t,
                constraints=JobConstraints(
                    allow_cutting=True,
                    force_cutting=(os.getenv("QDC_FORCE_CUT_WIDE", "1") == "1"),
                    allow_multi_qpu=True,
                    max_cuts=int(os.getenv("QDC_WIDE_MAX_CUTS", "2")),
                    comm_overhead_s=float(os.getenv("QDC_WIDE_COMM_OVERHEAD_S", "0.08")),
                ),
            )
            wl.append((t, job))
            wide_ids.add(jid)
            continue

        if rng.random() < (1.0 - pct_expect):
            width = rng.choice([3, 5, 7])
            depth = rng.choice([3, 6, 9])
            shots = rng.choice([500, 1000, 2000])
            qc = random_cx(width, depth, measure=True)
            job = Job(job_id=jid, circuit=qc, task_type="counts", shots=shots, submit_time_s=t)
            wl.append((t, job))
            continue

        width = rng.choice(wide_widths)
        shots = rng.choice([100, 200, 500])
        qc = ghz(width, measure=False)

        force_cut = (os.getenv("QDC_FORCE_CUT_MED", "0") == "1")
        if width > int(os.getenv("QDC_FORCE_CUT_IF_GT", "9999")):
            force_cut = True

        job = Job(
            job_id=jid,
            circuit=qc,
            task_type="expectation",
            observables=z_observable(width),
            shots=shots,
            submit_time_s=t,
            constraints=JobConstraints(
                allow_cutting=True,
                force_cutting=force_cut,
                allow_multi_qpu=True,
                max_cuts=int(os.getenv("QDC_MAX_CUTS", "6")),
                comm_overhead_s=float(os.getenv("QDC_COMM_OVERHEAD_S", "0.02")),
            ),
        )
        wl.append((t, job))

    return wl, wide_ids


def _install_reserve_counters(qpus: Dict[str, QPUState]):
    reserve_counts = Counter()
    for qid, st in qpus.items():
        orig_reserve = getattr(st, "reserve", None)
        if orig_reserve is None or not callable(orig_reserve):
            continue

        def make_wrapped(qid_local: str, orig_fn: Callable):
            def wrapped(job_id, qubits, start_s, duration_s, *args, **kwargs):
                reserve_counts[qid_local] += 1
                return orig_fn(job_id, qubits, start_s, duration_s, *args, **kwargs)
            return wrapped

        st.reserve = make_wrapped(qid, orig_reserve)
    return reserve_counts


def inject_congestion_bursts(qpus: Dict[str, QPUState]) -> None:
    burst_dur = float(os.getenv("QDC_CONGEST_BURST_S", "15.0"))
    for i, t0 in enumerate([15.0, 40.0]):
        k = qpus["qpu_B"].profile.num_qubits
        qpus["qpu_B"].reserve(f"BLOCK_B_{i}", list(range(k)), start_s=t0, duration_s=burst_dur)


def _install_plan_debug(sched: Scheduler, wide_job_ids: Set[str]) -> Counter:
    ctr: Counter = Counter()
    exec_obj = getattr(sched, "executor", None)
    if exec_obj is None:
        return ctr

    orig = exec_obj.run_job_plan
    wide_debug_budget = {"n": int(os.getenv("QDC_WIDE_DEBUG_N", "10"))}

    def wrapped(job, plan, now_s, toggles):
        kind = str(getattr(plan, "kind", "UNKNOWN"))
        ctr[kind] += 1
        if getattr(job, "job_id", None) in wide_job_ids and wide_debug_budget["n"] > 0:
            wide_debug_budget["n"] -= 1
            details = getattr(plan, "details", None) or {}
            ps = {}
            if isinstance(details, dict) and isinstance(details.get("plan_scores"), dict):
                for k, v in details["plan_scores"].items():
                    if isinstance(v, dict) and "score" in v:
                        ps[k] = v.get("score")
                    else:
                        ps[k] = None
            psk = details.get("plan_skips") if isinstance(details, dict) else None
            if isinstance(psk, dict):
                psk_summary = {bk: [x.get("reason") for x in vals][-3:] for bk, vals in psk.items() if vals}
            else:
                psk_summary = None
            print(f"[WIDE-DBG] job={job.job_id} kind={kind} qpu_id={getattr(plan,'qpu_id',None)} scores={ps} skips={psk_summary}")
        return orig(job, plan, now_s, toggles)

    exec_obj.run_job_plan = wrapped
    return ctr


def _print_histogram(ctr: Counter) -> None:
    total = sum(ctr.values())
    print("\n================ Plan histogram ================")
    for kind, n in sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = 100.0 * n / total if total else 0.0
        print(f"{kind:20s}  {n:4d}  ({pct:5.1f}%)")
    print(f"{'TOTAL':20s}  {total:4d}  (100.0%)")


def _export_results(sched: Scheduler, outdir: str) -> None:
    """Export scheduler metrics + task log to CSVs."""
    os.makedirs(outdir, exist_ok=True)

    # ---- records.csv ----
    records = []
    try:
        records = getattr(getattr(sched, "metrics", None), "records", []) or []
    except Exception:
        records = []

    rec_path = os.path.join(outdir, "records.csv")
    with open(rec_path, "w", newline="") as f:
        fieldnames = [
            "job_id","qpu_id","plan_kind","submit_time_s",
            "t_schedule_s","t_partition_s","t_mapping_s","t_execution_s",
            "t_reconstruction_s","end_to_end_s","fidelity_proxy",
            "fidelity_estimated","details_json"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({
                "job_id": getattr(r, "job_id", None),
                "qpu_id": getattr(r, "qpu_id", None),
                "plan_kind": getattr(r, "plan_kind", None),
                "submit_time_s": getattr(r, "submit_time_s", None),
                "t_schedule_s": getattr(r, "t_schedule_s", None),
                "t_partition_s": getattr(r, "t_partition_s", None),
                "t_mapping_s": getattr(r, "t_mapping_s", None),
                "t_execution_s": getattr(r, "t_execution_s", None),
                "t_reconstruction_s": getattr(r, "t_reconstruction_s", None),
                "end_to_end_s": getattr(r, "end_to_end_s", None),
                "fidelity_proxy": getattr(r, "fidelity_proxy", None),
                "fidelity_estimated": getattr(r, "fidelity_estimated", None),
                "details_json": json.dumps(getattr(r, "details", {}) or {}, default=str),
            })

    # ---- events.csv ----
    events = []
    try:
        events = getattr(sched, "task_log", []) or []
    except Exception:
        events = []

    evt_path = os.path.join(outdir, "events.csv")
    with open(evt_path, "w", newline="") as f:
        fieldnames = [
            "task_id","job_id","kind","start_s","end_s","qpu_id","qubits","label","depends_on","metadata_json"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in events:
            w.writerow({
                "task_id": getattr(t, "task_id", None),
                "job_id": getattr(t, "job_id", None),
                "kind": str(getattr(t, "kind", None)),
                "start_s": getattr(t, "start_s", None),
                "end_s": getattr(t, "end_s", None),
                "qpu_id": getattr(t, "qpu_id", None),
                "qubits": json.dumps(getattr(t, "qubits", None), default=str),
                "label": getattr(t, "label", None),
                "depends_on": json.dumps(getattr(t, "depends_on", None), default=str),
                "metadata_json": json.dumps(getattr(t, "metadata", {}) or {}, default=str),
            })

    print(f"[EXPORT] wrote {rec_path} and {evt_path}")


def run_workload(full_eval: bool = False) -> None:
    exclude_c = os.getenv("QDC_EXCLUDE_QPU_C", "1") == "1"
    base_A = float(os.getenv("QDC_QPU_A_BASE_DELAY", "1.5"))
    base_B = float(os.getenv("QDC_QPU_B_BASE_DELAY", "0.2"))
    base_C = float(os.getenv("QDC_QPU_C_BASE_DELAY", "0.3"))

    qpus: Dict[str, QPUState] = {
        "qpu_A": make_line_qpu("qpu_A", 7, base_queue_delay_s=base_A),
        "qpu_B": make_line_qpu("qpu_B", 7, base_queue_delay_s=base_B),
    }
    if not exclude_c:
        qpus["qpu_C"] = make_line_qpu("qpu_C", 14, base_queue_delay_s=base_C)

    reserve_counts = _install_reserve_counters(qpus)
    inject_congestion_bursts(qpus)

    cfg = SchedulerConfig()
    timing_mode_env = os.getenv("QDC_TIMING_MODE", "analytic").strip().lower()
    timing_mode = "aer" if timing_mode_env in ("aer", "aer_timing") else "analytic"
    cfg.exec_cfg = ExecConfig(
        reserve_nonsim=True,
        timing_mode=timing_mode,
        aer_timing_repeats=int(os.getenv("QDC_AER_TIMING_REPEATS", "1")),
        aer_timing_use_noise=(os.getenv("QDC_AER_TIMING_USE_NOISE", "0") == "1"),
        aer_timing_include_transpile=(os.getenv("QDC_AER_TIMING_INCLUDE_TRANSPILE", "0") == "1"),
    )

    sched = Scheduler(qpus=qpus, quality=QualityModel(noise_models={}), cfg=cfg)

    wl, wide_ids = make_workload()
    plan_ctr = _install_plan_debug(sched, wide_ids)

    print("[DEMO] reserve_nonsim =", bool(getattr(sched.executor.cfg, "reserve_nonsim", False)))
    print("[DEMO] QDC_EXCLUDE_QPU_C =", exclude_c)

    print("\n================ REAL WORKLOAD RUN ================")
    print("full_eval:", full_eval)
    print("timing_mode:", getattr(sched.executor.cfg, "timing_mode", None))
    print("aer_repeats:", os.getenv("QDC_AER_TIMING_REPEATS", "1"))
    print("jobs:", len(wl))
    print("congestion_burst_s:", float(os.getenv("QDC_CONGEST_BURST_S", "15.0")))
    print("wide_jobs:", len(wide_ids))

    toggles = RunToggles(
        compute_estimated_fidelity=False,
        compute_expectation=bool(full_eval),
        simulate_only=False,
    )

    now = 0.0
    next_idx = 0
    idle_steps = 0
    max_steps = int(os.getenv("QDC_MAX_STEPS", "50000"))
    stop_reason = None
    max_to_schedule = int(os.getenv("QDC_MAX_TO_SCHEDULE", "2"))

    step_wall_cap_s = float(os.getenv("QDC_STEP_WALL_CAP_S", "15.0"))

    for step in range(max_steps):
        while next_idx < len(wl) and wl[next_idx][0] <= now + 1e-12:
            _, job = wl[next_idx]
            sched.submit(job, toggles)
            next_idx += 1

        t0 = time.time()
        max_to_schedule = int(os.getenv("QDC_MAX_TO_SCHEDULE", str(max_to_schedule)))
        started = sched.step(now_s=now, max_to_schedule=max_to_schedule)
        wall = time.time() - t0
        if wall > step_wall_cap_s:
            print(f"[WATCHDOG] sched.step took {wall:.2f}s at t={now:.2f} (cap={step_wall_cap_s:.2f}s).")
            action = os.getenv("QDC_WATCHDOG_ACTION", "adapt").strip().lower()
            if action == "break":
                print("If using Aer timing, try: QDC_AER_TIMING_REPEATS=1 or lower shots.")
                stop_reason = f"watchdog wall={wall:.2f}s at t={now:.2f}"
                break
            max_to_schedule = max(1, max_to_schedule // 2)
            os.environ["QDC_MAX_TO_SCHEDULE"] = str(max_to_schedule)
            print(f"[WATCHDOG] adapting: setting QDC_MAX_TO_SCHEDULE={max_to_schedule} and continuing.")
            sched.tick(0.5)
            now += 0.5
            continue

        if started:
            idle_steps = 0
        else:
            idle_steps += 1

        dt = 0.5 if started else (max(0.5, float(wl[next_idx][0] - now)) if next_idx < len(wl) else 1.0)
        sched.tick(dt)
        now += dt

        if step % 5 == 0:
            qlen = len(getattr(sched, "queue", []))
            pend = len(getattr(sched, "_pending", []))
            print(f"[t={now:7.2f}] reserve_calls={dict(reserve_counts)} | queue={qlen} pending={pend}")

        if next_idx >= len(wl):
            qlen = len(getattr(sched, "queue", []))
            pend = len(getattr(sched, "_pending", []))
            if qlen == 0 and pend == 0 and idle_steps >= 300:
                break

    else:
        if stop_reason is None:
            stop_reason = "max_steps reached"

    completed = len(getattr(sched.metrics, "records", [])) if hasattr(sched, "metrics") else 0
    qlen = len(getattr(sched, "queue", []))
    pend = len(getattr(sched, "_pending", []))
    print(f"[DEMO] completed={completed} / total_jobs={len(wl)} submitted={next_idx} queue={qlen} pending={pend}")
    if stop_reason:
        print("[DEMO] stop_reason:", stop_reason)

    _print_histogram(plan_ctr)

    outdir = os.getenv("QDC_OUTDIR", "results/demo_workload")
    _export_results(sched, outdir=outdir)
    print("\nDone.")


def main():
    full_eval = os.getenv("QDC_FULL_EVAL", "0") == "1"
    run_workload(full_eval=full_eval)


if __name__ == "__main__":
    main()