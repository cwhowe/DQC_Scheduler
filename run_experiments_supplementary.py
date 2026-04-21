"""run_experiments.py — Supplementary DQC Scheduler experiments (E17–E24).

Runs any combination of the following experiments in a single invocation:

  E17  Congestion × arrival-rate joint sweep (4×3 = 12 conditions)
       Output: e17_congestion_arrival.csv
  E18  Utilisation–throughput Pareto frontier (pool-size × lambda grid)
       Output: e18_utilization_pareto.csv
  E19  Job-stream composition sweep (heavy-circuit fraction 0 → 100%)
       Output: e19_stream_composition.csv
  E20  Batch vs stream scheduling (7 submission strategies)
       Output: e20_batch_stream.csv
  E21  Throughput scaling vs job load (heterogeneous 3-QPU pool)
       Output: e21_throughput_scaling.json
  E24  QPU idle fraction vs arrival rate (homogeneous 3-QPU pool)
       Output: e24_idle_fraction.json

Usage
-----
    # Run everything
    python run_experiments.py --outdir results/experiments --seed 2026

    # Run a subset
    python run_experiments.py --experiments E17,E19,E20 --seed 2026

    # Quick smoke test
    python run_experiments.py --fast --seed 2026

    # Check API compatibility before a full run
    python run_experiments.py --diagnose

Note: E21 and E24 use a lightweight self-contained scheduler that does not
import qdc_sched, so they run even if the full package is unavailable.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qdc_sched.core.hardware import HardwareProfile, QPUState
from qdc_sched.core.quality import QualityModel
from qdc_sched.core.scheduler import Scheduler, SchedulerConfig
from qdc_sched.core.executor import ExecConfig
from qdc_sched.core.planner import PlannerConfig
from qdc_sched.core.types import Job, JobConstraints, RunToggles
from qdc_sched.core.metrics import JobRunRecord
from qdc_sched.cutting.base import CutConstraints
from qdc_sched.cutting.fitcut import FitCutCutStrategy

# ---------------------------------------------------------------------------
# Shared circuit factories
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def circ_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"ghz_{n}")
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    return qc


def circ_random(n: int, depth: int, seed: int = 0) -> QuantumCircuit:
    rng = _rng(seed)
    qc = QuantumCircuit(n, name=f"rand_{n}_d{depth}")
    for _ in range(depth):
        a = rng.randrange(n)
        b = rng.randrange(n)
        while b == a:
            b = rng.randrange(n)
        qc.cx(a, b)
        qc.h(a)
    return qc


def circ_qaoa(n: int, p: int = 1, seed: int = 0) -> QuantumCircuit:
    rng = _rng(seed)
    qc = QuantumCircuit(n, name=f"qaoa_{n}_p{p}")
    for i in range(n):
        qc.h(i)
    for _ in range(p):
        for i in range(0, n - 1, 2):
            gamma = rng.uniform(0, 3.14)
            qc.cx(i, i + 1)
            qc.rz(gamma, i + 1)
            qc.cx(i, i + 1)
        for i in range(n):
            beta = rng.uniform(0, 3.14)
            qc.rx(2 * beta, i)
    return qc


def circ_vqe(n: int, layers: int = 2, seed: int = 0) -> QuantumCircuit:
    rng = _rng(seed)
    qc = QuantumCircuit(n, name=f"vqe_{n}_l{layers}")
    for layer in range(layers):
        for i in range(n):
            qc.ry(rng.uniform(0, 3.14), i)
        offset = layer % 2
        for i in range(offset, n - 1, 2):
            qc.cx(i, i + 1)
    for i in range(n):
        qc.ry(rng.uniform(0, 3.14), i)
    return qc


def z_obs(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_list([("Z" * n, 1.0)])


# ---------------------------------------------------------------------------
# Shared QPU / scheduler / tick-loop helpers (used by E17–E20)
# ---------------------------------------------------------------------------

def _line_graph(n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(i, i + 1) for i in range(n - 1)])
    return G


def make_qpu(
    qpu_id: str,
    n: int,
    twoq_error: float = 2.0e-2,
    base_queue_delay_s: float = 0.3,
    oneq_error: float = 1.2e-3,
    twoq_gate_time_s: float = 300e-9,
    oneq_gate_time_s: float = 35e-9,
    meas_time_s: float = 1_000e-9,
    shot_overhead_s: float = 250e-6,
) -> QPUState:
    G = _line_graph(n)
    prof = HardwareProfile(
        qpu_id=qpu_id,
        num_qubits=len(G.nodes),
        coupling_graph=G,
        base_queue_delay_s=base_queue_delay_s,
        oneq_gate_time_s=oneq_gate_time_s,
        twoq_gate_time_s=twoq_gate_time_s,
        meas_time_s=meas_time_s,
        shot_overhead_s=shot_overhead_s,
        oneq_error=oneq_error,
        twoq_error=twoq_error,
        readout_error=2.0e-2,
    )
    return QPUState(prof)


def build_scheduler(
    qpus: Dict[str, QPUState],
    *,
    allow_cutting: bool = True,
    allow_multi_qpu: bool = True,
    max_cuts: int = 2,
    timing_mode: str = "analytic",
) -> Scheduler:
    strategy = FitCutCutStrategy()
    cc = CutConstraints(
        max_cuts=max_cuts,
        allow_wire_cuts=True,
        allow_gate_cuts=True,
        reconstruction_target="expectation",
        target_labels_cap=4,
        seed_tries=2,
    )
    planner_cfg = PlannerConfig(cut_constraints=cc, cut_strategy=strategy)
    exec_cfg = ExecConfig(reserve_nonsim=True, timing_mode=timing_mode)
    cfg = SchedulerConfig(planner=planner_cfg, exec_cfg=exec_cfg)
    quality = QualityModel(
        noise_models={},
        qpu_profiles={qid: st.profile for qid, st in qpus.items()},
    )
    return Scheduler(qpus=qpus, quality=quality, cfg=cfg)


def run_tick_loop(
    sched: Scheduler,
    workload: List[Tuple[float, Job]],
    toggles: RunToggles,
    *,
    max_steps: int = 60_000,
    step_dt_s: float = 0.5,
    idle_break: int = 20,
    max_job_age_s: float = 120.0,
) -> List[JobRunRecord]:
    now = 0.0
    next_idx = 0
    idle = 0

    for _ in range(max_steps):
        while next_idx < len(workload) and workload[next_idx][0] <= now + 1e-12:
            sched.submit(workload[next_idx][1], toggles)
            next_idx += 1

        if max_job_age_s > 0 and hasattr(sched, "expire_pending"):
            sched.expire_pending(now_s=now, max_age_s=max_job_age_s)

        started = sched.step(now_s=now)
        idle = 0 if started else idle + 1

        pending = sched.pending_count()
        active = sum(
            sum(1 for r in st.reservations if r.end_s > now)
            for st in sched.qpus.values()
        )

        if next_idx >= len(workload) and pending == 0 and active == 0:
            break
        if next_idx >= len(workload) and pending == 0 and idle >= idle_break:
            break
        if next_idx >= len(workload) and idle >= idle_break * 4:
            break

        if started:
            dt = step_dt_s
        elif next_idx < len(workload):
            dt = max(step_dt_s, workload[next_idx][0] - now)
        else:
            dt = step_dt_s

        sched.tick(dt)
        now += dt

    return list(getattr(sched.metrics, "records", []))


# ---------------------------------------------------------------------------
# Shared CSV output helpers (used by E17–E20)
# ---------------------------------------------------------------------------

def _safe(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 9)
    return v


def _det(rec: JobRunRecord, key: str, default: Any = "") -> Any:
    try:
        return (rec.details or {}).get(key, default)
    except Exception:
        return default


def record_to_row(rec: JobRunRecord, experiment: str, condition: str) -> Dict[str, Any]:
    d  = rec.details or {}
    tw = d.get("timing_wall_s", {}) or {}
    tm = d.get("timing_model_s", {}) or {}
    return {
        "experiment":             experiment,
        "condition":              condition,
        "job_id":                 rec.job_id,
        "qpu_id":                 _safe(rec.qpu_id),
        "plan_kind":              rec.plan_kind,
        "submit_time_s":          _safe(rec.submit_time_s),
        "end_to_end_s":           _safe(rec.end_to_end_s),
        "t_schedule_s":           _safe(rec.t_schedule_s),
        "t_execution_s":          _safe(rec.t_execution_s),
        "t_reconstruction_s":     _safe(rec.t_reconstruction_s),
        "fidelity_proxy":         _safe(rec.fidelity_proxy),
        "fidelity_estimated":     _safe(rec.fidelity_estimated),
        "pred_exec_s":            _safe(_det(rec, "pred_exec_time_s")),
        "pred_queue_s":           _safe(_det(rec, "pred_queue_delay_s")),
        "pred_recon_s":           _safe(_det(rec, "pred_recon_s")),
        "pred_comm_s":            _safe(_det(rec, "pred_comm_s")),
        "sim_queue_wait_s":       _safe(_det(rec, "sim_queue_wait_s")),
        "sim_execution_span_s":   _safe(_det(rec, "sim_execution_span_s")),
        "sim_comm_s":             _safe(_det(rec, "sim_comm_s")),
        "sim_recon_s":            _safe(_det(rec, "sim_recon_s")),
        "sim_latency_s":          _safe(_det(rec, "sim_latency_s")),
        "charged_comm_s":         _safe(_det(rec, "charged_comm_s")),
        "charged_recon_s":        _safe(_det(rec, "charged_recon_s")),
        "wall_exec_s":            _safe(tw.get("execute_s")),
        "wall_recon_s":           _safe(tw.get("reconstruct_s")),
        "wall_total_s":           _safe(tw.get("total_s")),
        "model_exec_s":           _safe(tm.get("execute_s")),
        "model_comm_s":           _safe(tm.get("communication_s")),
        "model_recon_s":          _safe(tm.get("reconstruct_s")),
        "sampling_overhead":      _safe(_det(rec, "sampling_overhead")),
        "labels_used":            _safe(_det(rec, "labels_used")),
        "k_wire":                 _safe(_det(rec, "k_wire")),
        "k_gate":                 _safe(_det(rec, "k_gate")),
        "pending_attempts":       _safe(_det(rec, "pending_attempts")),
        "schedule_wall_s":        _safe(_det(rec, "schedule_wall_s")),
        "queue_wait_s":           _safe(_det(rec, "queue_wait_s")),
        "score_qpu_completion_s": _safe((_det(rec, "score_components") or {}).get("qpu_completion_s")),
        "score_frag_penalty_s":   _safe((_det(rec, "score_components") or {}).get("frag_penalty_s")),
        "score_coord_penalty_s":  _safe((_det(rec, "score_components") or {}).get("coord_penalty_s")),
        "score_quality_term":     _safe((_det(rec, "score_components") or {}).get("quality_term")),
        "score_total":            _safe((_det(rec, "score_components") or {}).get("total_score")),
    }


def append_rows(path: str, rows: List[Dict], write_header: bool) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Shared workload builders (used by E17–E20)
# ---------------------------------------------------------------------------

def _wide_job(job_id: str, t: float, rng: random.Random) -> Tuple[float, Job]:
    n     = rng.choice([9, 10, 11, 12])
    shots = rng.choice([500, 1000, 2000])
    fam   = rng.choice(["qaoa", "vqe", "random"])
    s     = rng.randint(0, 9999)
    qc    = (circ_qaoa(n, p=rng.choice([1, 2]), seed=s) if fam == "qaoa"
             else circ_vqe(n, layers=rng.choice([2, 3]), seed=s) if fam == "vqe"
             else circ_random(n, depth=max(8, n), seed=s))
    job = Job(
        job_id=job_id, circuit=qc, task_type="expectation", observables=z_obs(n),
        shots=shots, submit_time_s=t,
        constraints=JobConstraints(
            allow_cutting=True, allow_multi_qpu=True, force_cutting=True, max_cuts=2,
        ),
    )
    return (t, job)


def wide_stream(n_jobs: int, arrival_rate: float = 1.0, seed: int = 42) -> List[Tuple[float, Job]]:
    """Poisson-arrival stream of wide circuits — every job forces cutting."""
    rng = _rng(seed)
    wl: List[Tuple[float, Job]] = []
    t = 0.0
    for i in range(n_jobs):
        t += rng.expovariate(arrival_rate)
        wl.append(_wide_job(f"W{i:04d}", t, rng))
    return wl


def mixed_stream(
    n_jobs: int,
    pct_heavy: float = 0.25,
    arrival_rate: float = 1.0,
    seed: int = 42,
) -> List[Tuple[float, Job]]:
    """Poisson-arrival stream with configurable heavy/light mix."""
    rng = _rng(seed)
    wl: List[Tuple[float, Job]] = []
    t = 0.0
    for i in range(n_jobs):
        t += rng.expovariate(arrival_rate)
        if rng.random() < pct_heavy:
            wl.append(_wide_job(f"J{i:04d}", t, rng))
        else:
            n     = rng.choice([3, 4, 5, 6, 7])
            shots = rng.choice([500, 1000])
            fam   = rng.choice(["ghz", "random", "qaoa"])
            s     = rng.randint(0, 9999)
            qc    = (circ_ghz(n) if fam == "ghz"
                     else circ_qaoa(n, p=1, seed=s) if fam == "qaoa"
                     else circ_random(n, depth=max(4, n), seed=s))
            job = Job(
                job_id=f"J{i:04d}", circuit=qc, task_type="expectation",
                observables=z_obs(n), shots=shots, submit_time_s=t,
                constraints=JobConstraints(
                    allow_cutting=True, allow_multi_qpu=True, force_cutting=False, max_cuts=2,
                ),
            )
            wl.append((t, job))
    return wl


def _narrow_job(jid: str, t: float, rng: random.Random) -> Job:
    n     = rng.choice([3, 4, 5, 6])
    shots = rng.choice([500, 1000])
    fam   = rng.choice(["ghz", "random"])
    qc    = circ_ghz(n) if fam == "ghz" else circ_random(n, max(3, n), rng.randint(0, 9999))
    return Job(
        job_id=jid, circuit=qc, task_type="expectation", observables=z_obs(n),
        shots=shots, submit_time_s=t,
        constraints=JobConstraints(allow_cutting=True, allow_multi_qpu=True,
                                   force_cutting=False, max_cuts=2),
    )


def _wide_job_e20(jid: str, t: float, rng: random.Random, urgent: bool = False) -> Job:
    n     = rng.choice([9, 10, 11, 12])
    shots = rng.choice([500, 1000])
    fam   = rng.choice(["qaoa", "random"])
    qc    = (circ_qaoa(n, p=1, seed=rng.randint(0, 9999)) if fam == "qaoa"
             else circ_random(n, max(8, n), rng.randint(0, 9999)))
    return Job(
        job_id=jid, circuit=qc, task_type="expectation", observables=z_obs(n),
        shots=shots, submit_time_s=t,
        constraints=JobConstraints(allow_cutting=True, allow_multi_qpu=True,
                                   force_cutting=True, max_cuts=2,
                                   slo_s=1.0 if urgent else None),
    )


def _mixed_jobs_e20(n_jobs: int, seed: int) -> List[Job]:
    rng = _rng(seed)
    jobs = []
    for i in range(n_jobs):
        jid = f"M{i:04d}"
        jobs.append(_narrow_job(jid, 0.0, rng) if i % 2 == 0
                    else _wide_job_e20(jid, 0.0, rng))
    return jobs


def poisson_stream_e20(
    n_jobs: int, lam: float, seed: int, pct_urgent: float = 0.0
) -> List[Tuple[float, Job]]:
    rng = _rng(seed)
    t = 0.0
    wl = []
    base = _mixed_jobs_e20(n_jobs, seed)
    for i, _ in enumerate(base):
        t += rng.expovariate(lam)
        urgent = (rng.random() < pct_urgent)
        jid = f"P{i:04d}"
        j = (_narrow_job(jid, t, rng) if i % 2 == 0
             else _wide_job_e20(jid, t, rng, urgent=urgent))
        wl.append((t, j))
    return wl


def micro_batch_stream(
    n_jobs: int, batch_size: int, inter_batch_s: float, seed: int
) -> List[Tuple[float, Job]]:
    rng = _rng(seed)
    base = _mixed_jobs_e20(n_jobs, seed)
    wl = []
    t = 0.0
    for i, _ in enumerate(base):
        if i > 0 and i % batch_size == 0:
            t += inter_batch_s
        jid = f"B{i:04d}"
        j = _narrow_job(jid, t, rng) if i % 2 == 0 else _wide_job_e20(jid, t, rng)
        wl.append((t, j))
    return wl


def batch_all_e20(n_jobs: int, seed: int) -> List[Tuple[float, Job]]:
    rng = _rng(seed)
    base = _mixed_jobs_e20(n_jobs, seed)
    wl = []
    for i, _ in enumerate(base):
        jid = f"A{i:04d}"
        j = _narrow_job(jid, 0.0, rng) if i % 2 == 0 else _wide_job_e20(jid, 0.0, rng)
        wl.append((0.0, j))
    return wl


# ---------------------------------------------------------------------------
# E17: Congestion × arrival-rate joint sweep
# ---------------------------------------------------------------------------

def exp_e17(args: argparse.Namespace, path: str) -> None:
    print("\n=== E17: Congestion × Arrival-rate sweep ===")
    cong_levels  = [("cong_0pct", 0.00), ("cong_33pct", 0.33),
                    ("cong_66pct", 0.66), ("cong_100pct", 1.00)]
    arrival_rates = [("lam_0p5", 0.5), ("lam_2p0", 2.0), ("lam_5p0", 5.0)]

    first = True
    for cong_key, cong_frac in cong_levels:
        for lam_key, lam in arrival_rates:
            cond = f"{cong_key}__{lam_key}"
            print(f"  {cond}")
            qpus = {
                "qpu_A": make_qpu("qpu_A", 7, twoq_error=5.0e-2),
                "qpu_B": make_qpu("qpu_B", 7, twoq_error=2.0e-2),
                "qpu_C": make_qpu("qpu_C", 7, twoq_error=0.5e-2),
            }
            if cong_frac > 0:
                st_c = qpus["qpu_C"]
                n_block = max(1, int(round(st_c.profile.num_qubits * cong_frac)))
                for t0 in [0.0, 15.0, 30.0, 60.0, 90.0, 120.0]:
                    st_c.reserve(f"CONG_{cond}_{t0:.0f}", list(range(n_block)),
                                 start_s=t0, duration_s=15.0)
            sched   = build_scheduler(qpus)
            wl      = wide_stream(args.n_jobs, arrival_rate=lam,
                                  seed=args.seed + hash(cond) % 1000)
            records = run_tick_loop(sched, wl, RunToggles(simulate_only=False))
            rows    = [record_to_row(r, "E17_congestion_arrival", cond) for r in records]
            append_rows(path, rows, write_header=first)
            first = False
            e2e = [r.end_to_end_s for r in records if r.end_to_end_s]
            print(f"    -> {len(records)} jobs  e2e_med={statistics.median(e2e) if e2e else float('nan'):.3f}s")


# ---------------------------------------------------------------------------
# E18: Utilisation–throughput Pareto frontier
# ---------------------------------------------------------------------------

def exp_e18(args: argparse.Namespace, path: str) -> None:
    print("\n=== E18: Utilisation–throughput Pareto frontier ===")
    pool_sizes    = [2, 3, 4, 6, 8]
    arrival_rates = [0.5, 1.0, 2.0, 4.0]
    if args.fast:
        pool_sizes    = [2, 3, 4]
        arrival_rates = [0.5, 2.0]

    first = True
    for n_qpu in pool_sizes:
        for lam in arrival_rates:
            lam_str = f"{lam:.1f}".replace(".", "p")
            cond    = f"n{n_qpu}qpu__lam{lam_str}"
            print(f"  {cond}")
            qpus = {
                f"qpu_{chr(65 + i)}": make_qpu(f"qpu_{chr(65 + i)}", 7, twoq_error=2.0e-2)
                for i in range(n_qpu)
            }
            sched   = build_scheduler(qpus)
            wl      = wide_stream(args.n_jobs, arrival_rate=lam,
                                  seed=args.seed + hash(cond) % 1000)
            records = run_tick_loop(sched, wl, RunToggles(simulate_only=False),
                                    idle_break=25, max_steps=60_000)
            rows    = [record_to_row(r, "E18_utilization_pareto", cond) for r in records]
            append_rows(path, rows, write_header=first)
            first = False
            e2e = [r.end_to_end_s for r in records if r.end_to_end_s]
            pct_c = 100 * sum(1 for r in records if r.plan_kind == "C_CUT_MULTI_QPU") / max(len(records), 1)
            print(f"    -> {len(records)} jobs  C:{pct_c:.0f}%  p90={e2e[int(0.9*len(e2e))]:.3f}s" if e2e else f"    -> {len(records)} jobs")


# ---------------------------------------------------------------------------
# E19: Job-stream composition sweep
# ---------------------------------------------------------------------------

def exp_e19(args: argparse.Namespace, path: str) -> None:
    print("\n=== E19: Job-stream composition sweep ===")
    heavy_fractions = [
        ("heavy_00pct", 0.00), ("heavy_20pct", 0.20), ("heavy_40pct", 0.40),
        ("heavy_60pct", 0.60), ("heavy_80pct", 0.80), ("heavy_100pct", 1.00),
    ]

    first = True
    for cname, pct_heavy in heavy_fractions:
        print(f"  {cname}  (heavy={int(pct_heavy * 100)}%)")
        qpus = {
            "qpu_A": make_qpu("qpu_A", 7, twoq_error=5.0e-2),
            "qpu_B": make_qpu("qpu_B", 7, twoq_error=0.5e-2),
        }
        sched   = build_scheduler(qpus)
        wl      = mixed_stream(args.n_jobs, pct_heavy=pct_heavy, arrival_rate=1.0,
                               seed=args.seed + hash(cname) % 1000)
        records = run_tick_loop(sched, wl, RunToggles(simulate_only=False),
                                idle_break=20, max_steps=50_000)
        rows    = [record_to_row(r, "E19_stream_composition", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        e2e  = sorted(r.end_to_end_s for r in records if r.end_to_end_s)
        fids = [r.fidelity_proxy for r in records if r.fidelity_proxy]
        p90  = e2e[int(0.9 * len(e2e))] if e2e else float("nan")
        fmed = statistics.median(fids) if fids else float("nan")
        pct_c = 100 * sum(1 for r in records if r.plan_kind == "C_CUT_MULTI_QPU") / max(len(records), 1)
        print(f"    -> {len(records)} jobs  C:{pct_c:.0f}%  fid={fmed:.3f}  p90={p90:.3f}s")


# ---------------------------------------------------------------------------
# E20: Batch vs stream scheduling
# ---------------------------------------------------------------------------

def exp_e20(args: argparse.Namespace, path: str) -> None:
    print("\n=== E20: Batch vs stream scheduling ===")

    def _pool() -> Dict[str, QPUState]:
        return {
            "qpu_A": make_qpu("qpu_A", 7, twoq_error=5.0e-2),
            "qpu_B": make_qpu("qpu_B", 7, twoq_error=2.0e-2),
            "qpu_C": make_qpu("qpu_C", 7, twoq_error=0.5e-2),
        }

    conditions = [
        ("stream_slow",   lambda: poisson_stream_e20(args.n_jobs, 0.5, args.seed),           "λ=0.5"),
        ("stream_medium", lambda: poisson_stream_e20(args.n_jobs, 2.0, args.seed),           "λ=2.0"),
        ("stream_fast",   lambda: poisson_stream_e20(args.n_jobs, 5.0, args.seed),           "λ=5.0"),
        ("micro_batch_5", lambda: micro_batch_stream(args.n_jobs, 5, 5.0, args.seed),        "burst=5"),
        ("micro_batch_20",lambda: micro_batch_stream(args.n_jobs, 20, 10.0, args.seed),      "burst=20"),
        ("batch_all",     lambda: batch_all_e20(args.n_jobs, args.seed),                     "all-at-once"),
        ("priority_mix",  lambda: poisson_stream_e20(args.n_jobs, 2.0, args.seed, 0.20),     "λ=2.0, 20% urgent"),
    ]
    if args.fast:
        conditions = conditions[:4]

    first = True
    for cname, wl_fn, desc in conditions:
        print(f"  {cname}  ({desc})")
        sched   = build_scheduler(_pool())
        records = run_tick_loop(sched, wl_fn(), RunToggles(simulate_only=False))
        rows    = [record_to_row(r, "E20_batch_stream", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        plans = collections.Counter(r.plan_kind for r in records)
        e2e   = sorted(r.end_to_end_s for r in records if r.end_to_end_s)
        p90   = e2e[int(0.9 * len(e2e))] if e2e else float("nan")
        print(f"    -> {len(records)} jobs  "
              f"A:{plans.get('A_NO_CUT_SINGLE',0)}  "
              f"B:{plans.get('B_CUT_SINGLE_SEQ',0)}  "
              f"C:{plans.get('C_CUT_MULTI_QPU',0)}  p90={p90:.3f}s")


# ---------------------------------------------------------------------------
# E21 / E24 — lightweight self-contained scheduler (no qdc_sched dependency)
# ---------------------------------------------------------------------------

@dataclass
class _QPU:
    qpu_id: str
    n_qubits: int
    gate_time_us: float
    t1_us: float

    @property
    def throughput_factor(self) -> float:
        return self.n_qubits * (1.0 / self.gate_time_us)


@dataclass
class _Job:
    job_id: str
    n_qubits: int
    n_gates: int
    arrival_time: float
    slo_s: float
    priority: int = 1


@dataclass
class _Result:
    job_id: str
    plan: str
    start_time: float
    exec_time_s: float
    queue_wait_s: float
    comm_overhead_s: float
    overhead_s: float
    finish_time: float
    met_slo: bool
    score: float


_W_LAT, _W_UTIL, _W_FID, _W_SLO = 0.35, 0.25, 0.25, 0.15


def _score(plan: str, job: _Job, exec_t: float, queue_w: float, comm_s: float, oh_s: float) -> float:
    lat = queue_w + exec_t + comm_s + oh_s
    lat_score = max(0.0, 1.0 - lat / max(job.slo_s, 1e-9))
    util = {"A": 0.70, "B": 0.55, "C": 0.85}[plan]
    fid  = {"A": 0.90, "B": 0.75, "C": 0.70}[plan]
    slo  = 1.0 if lat <= job.slo_s else 0.0
    return _W_LAT * lat_score + _W_UTIL * util + _W_FID * fid + _W_SLO * slo


class _SimpleScheduler:
    def __init__(self, qpus: List[_QPU], seed: int = 42):
        self.qpus = qpus
        self.rng  = random.Random(seed)
        self.qpu_free: Dict[str, float] = {q.qpu_id: 0.0 for q in qpus}

    def _exec(self, job: _Job, qpu: _QPU) -> float:
        return job.n_gates * qpu.gate_time_us * 1e-6

    def _comm(self, n_cuts: int) -> float:
        return n_cuts * 0.002

    def schedule(self, job: _Job) -> _Result:
        cands = []
        for qpu in self.qpus:
            ready = max(self.qpu_free[qpu.qpu_id], job.arrival_time)
            if qpu.n_qubits >= job.n_qubits:
                et = self._exec(job, qpu); ct = 0.0; oh = 0.001
                qw = max(0.0, ready - job.arrival_time)
                cands.append(("A", [qpu.qpu_id], ready, et, qw, ct, oh,
                               _score("A", job, et, qw, ct, oh)))
            n_cuts = max(1, job.n_qubits // qpu.n_qubits)
            et = self._exec(job, qpu) * (1 + 0.15 * n_cuts)
            ct = self._comm(n_cuts); oh = 0.002 + 0.001 * n_cuts
            qw = max(0.0, ready - job.arrival_time)
            cands.append(("B", [qpu.qpu_id], ready, et, qw, ct, oh,
                           _score("B", job, et, qw, ct, oh)))
        if len(self.qpus) >= 2:
            ready = max(job.arrival_time, min(self.qpu_free[q.qpu_id] for q in self.qpus))
            n_par = min(len(self.qpus), max(2, job.n_qubits // 8))
            chosen = sorted(self.qpus, key=lambda q: self.qpu_free[q.qpu_id])[:n_par]
            et = self._exec(job, chosen[0]) / n_par * 1.2
            ct = self._comm(n_par - 1) * 1.5; oh = 0.003 + 0.002 * (n_par - 1)
            qw = max(0.0, ready - job.arrival_time)
            cands.append(("C", [q.qpu_id for q in chosen], ready, et, qw, ct, oh,
                           _score("C", job, et, qw, ct, oh)))
        if not cands:
            qpu = self.qpus[0]
            ready = max(self.qpu_free[qpu.qpu_id], job.arrival_time)
            et = self._exec(job, qpu); ct = 0.002; oh = 0.002
            qw = max(0.0, ready - job.arrival_time)
            cands.append(("B", [qpu.qpu_id], ready, et, qw, ct, oh,
                           _score("B", job, et, qw, ct, oh)))
        plan, qpu_ids, start, et, qw, ct, oh, score = max(cands, key=lambda c: c[7])
        finish = start + et + ct + oh
        for qid in qpu_ids:
            self.qpu_free[qid] = finish
        return _Result(job.job_id, plan, start, et, qw, ct, oh, finish,
                       (qw + et + ct + oh) <= job.slo_s, score)


def _gen_jobs(n: int, lam: float, seed: int, slo_s: float = 5.0) -> List[_Job]:
    rng = random.Random(seed)
    jobs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(lam)
        n_q = rng.choice([4, 6, 8, 12, 16, 20, 27])
        n_g = rng.randint(n_q * 5, n_q * 40)
        jobs.append(_Job(f"J{i:04d}", n_q, n_g, t, slo_s, rng.randint(1, 3)))
    return jobs


def _metrics(jobs: List[_Job], results: List[_Result], wall: float) -> dict:
    n = len(results)
    lats = [r.queue_wait_s + r.exec_time_s + r.comm_overhead_s + r.overhead_s for r in results]
    lats_s = sorted(lats)
    pc = {"A": 0, "B": 0, "C": 0}
    for r in results:
        pc[r.plan] = pc.get(r.plan, 0) + 1
    tex  = sum(r.exec_time_s for r in results)
    twait = sum(r.queue_wait_s for r in results)
    tcomm = sum(r.comm_overhead_s for r in results)
    toh   = sum(r.overhead_s for r in results)
    tot  = sum(lats)
    return dict(
        n_jobs=n,
        throughput_jobs_per_s=n / max(wall, 1e-9),
        slo_rate=sum(r.met_slo for r in results) / n,
        latency_p50_s=lats_s[int(0.50 * n)],
        latency_p90_s=lats_s[min(int(0.90 * n), n - 1)],
        latency_p99_s=lats_s[min(int(0.99 * n), n - 1)],
        plan_A_count=pc.get("A", 0), plan_B_count=pc.get("B", 0), plan_C_count=pc.get("C", 0),
        plan_C_frac=pc.get("C", 0) / n,
        exec_frac=tex / max(tot, 1e-9),
        idle_frac=1.0 - tex / max(tot, 1e-9),
        queue_wait_frac=twait / max(tot, 1e-9),
        mean_exec_s=tex / n,
        mean_wait_s=twait / n,
        mean_comm_s=tcomm / n,
        mean_overhead_s=toh / n,
    )


def exp_e21(args: argparse.Namespace, outdir: Path) -> None:
    print("\n=== E21: Throughput scaling vs job load ===")
    n_jobs_sweep = [10, 20, 40, 80, 160] if not args.fast else [10, 20, 40]
    pool = [_QPU("QPU-0", 27, 50, 100_000), _QPU("QPU-1", 16, 35, 80_000), _QPU("QPU-2", 8, 20, 60_000)]

    records = []
    for n_jobs in n_jobs_sweep:
        print(f"  n_jobs={n_jobs}", end=" ", flush=True)
        jobs  = _gen_jobs(n_jobs, 2.0, args.seed)
        sched = _SimpleScheduler(deepcopy(pool), seed=args.seed)
        t0    = time.perf_counter()
        res   = [sched.schedule(j) for j in jobs]
        wall  = time.perf_counter() - t0
        sim_wall = max(r.finish_time for r in res) - min(j.arrival_time for j in jobs)
        m = _metrics(jobs, res, sim_wall)
        m.update(n_jobs_configured=n_jobs, pool_type="heterogeneous", real_wall_s=wall)
        m["lambda"] = 2.0
        records.append(m)
        print(f"throughput={m['throughput_jobs_per_s']:.3f} jobs/s  SLO={m['slo_rate']:.1%}")

    out = outdir / "e21_throughput_scaling.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"  -> {out}")


def exp_e24(args: argparse.Namespace, outdir: Path) -> None:
    print("\n=== E24: Idle fraction vs arrival rate ===")
    lambda_sweep = ([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0] if not args.fast
                    else [0.1, 0.5, 1.0, 2.0, 4.0, 8.0])
    pool = [_QPU("QPU-0", 20, 40, 90_000), _QPU("QPU-1", 20, 40, 90_000), _QPU("QPU-2", 20, 40, 90_000)]

    records = []
    for lam in lambda_sweep:
        print(f"  lambda={lam:.2f}", end=" ", flush=True)
        jobs  = _gen_jobs(args.n_jobs, lam, args.seed)
        sched = _SimpleScheduler(deepcopy(pool), seed=args.seed)
        t0    = time.perf_counter()
        res   = [sched.schedule(j) for j in jobs]
        wall  = time.perf_counter() - t0
        sim_wall = max(r.finish_time for r in res) - min(j.arrival_time for j in jobs)
        m = _metrics(jobs, res, sim_wall)
        m.update(n_jobs_configured=args.n_jobs, pool_type="homogeneous", real_wall_s=wall)
        m["lambda"] = lam
        records.append(m)
        print(f"idle={m['idle_frac']:.3f}  exec={m['exec_frac']:.3f}  SLO={m['slo_rate']:.1%}")

    out = outdir / "e24_idle_fraction.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics() -> None:
    """Smoke-test every API call used in E17–E20. Prints PASS/FAIL per step."""
    import traceback
    steps = []

    def check(name, fn):
        try:
            steps.append((name, "PASS", repr(fn())[:80]))
        except Exception:
            steps.append((name, "FAIL", traceback.format_exc().strip().splitlines()[-1]))

    check("circ_ghz(5)",          lambda: circ_ghz(5))
    check("circ_random(5,4)",     lambda: circ_random(5, 4, seed=0))
    check("circ_qaoa(6,p=1)",     lambda: circ_qaoa(6, p=1, seed=0))
    check("circ_vqe(6,layers=2)", lambda: circ_vqe(6, layers=2, seed=0))
    check("z_obs(6)",             lambda: z_obs(6))
    check("make_qpu",             lambda: make_qpu("qpu_A", 7))
    check("build_scheduler",      lambda: build_scheduler(
        {"qpu_A": make_qpu("qpu_A", 7), "qpu_B": make_qpu("qpu_B", 7)}))
    check("wide_stream(2)",       lambda: wide_stream(2, seed=0))
    check("mixed_stream(2)",      lambda: mixed_stream(2, seed=0))
    check("_SimpleScheduler",     lambda: _SimpleScheduler(
        [_QPU("Q0", 7, 40, 90_000)], seed=0).schedule(
        _gen_jobs(1, 1.0, 0)[0]))

    print("\n=== DIAGNOSTICS ===")
    any_fail = False
    for name, status, detail in steps:
        print(f"  [{'OK  ' if status == 'PASS' else 'FAIL'}] {name}")
        if status == "FAIL":
            print(f"         {detail}")
            any_fail = True
    print()
    if any_fail:
        print("Fix the FAIL items above before running experiments.")
    else:
        print("All checks passed — safe to run.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_EXPERIMENTS = ["E17", "E18", "E19", "E20", "E21", "E24"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--experiments", default=",".join(ALL_EXPERIMENTS),
        help=f"Comma-separated experiments to run (default: all). "
             f"Choices: {', '.join(ALL_EXPERIMENTS)}",
    )
    p.add_argument("--outdir",  default="results/experiments",
                   help="Output directory for CSV and JSON files (default: results/experiments)")
    p.add_argument("--n-jobs",  type=int, default=40,
                   help="Jobs per condition (default: 40)")
    p.add_argument("--seed",    type=int, default=2026,
                   help="Random seed (default: 2026)")
    p.add_argument("--fast",    action="store_true",
                   help="Reduced sweep sizes for quick smoke test")
    p.add_argument("--diagnose", action="store_true",
                   help="Smoke-test API calls and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.diagnose:
        run_diagnostics()
        return

    wanted = {e.strip().upper() for e in args.experiments.split(",")}
    unknown = wanted - set(ALL_EXPERIMENTS)
    if unknown:
        print(f"[ERROR] Unknown experiments: {unknown}. Valid: {ALL_EXPERIMENTS}")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[run_experiments]  outdir={outdir}  n_jobs={args.n_jobs}  seed={args.seed}")
    print(f"[run_experiments]  running: {sorted(wanted)}  fast={args.fast}\n")
    t0_wall = time.time()

    dispatch_csv = {
        "E17": (exp_e17, "e17_congestion_arrival.csv"),
        "E18": (exp_e18, "e18_utilization_pareto.csv"),
        "E19": (exp_e19, "e19_stream_composition.csv"),
        "E20": (exp_e20, "e20_batch_stream.csv"),
    }
    dispatch_json = {
        "E21": exp_e21,
        "E24": exp_e24,
    }

    for key in ["E17", "E18", "E19", "E20"]:
        if key not in wanted:
            continue
        fn, fname = dispatch_csv[key]
        path = str(outdir / fname)
        print(f"--- {key} -> {fname} ---")
        fn(args, path)

    for key in ["E21", "E24"]:
        if key not in wanted:
            continue
        print(f"--- {key} ---")
        dispatch_json[key](args, outdir)

    print(f"\n[run_experiments]  done in {time.time() - t0_wall:.1f}s")
    print(f"[run_experiments]  outputs -> {outdir}/")


if __name__ == "__main__":
    main()