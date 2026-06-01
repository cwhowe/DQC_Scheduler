"""run_experiments.py — DQC Scheduler experiment harness.

Runs all five experiment groups and writes per-experiment CSVs to --outdir.
Each experiment is fully self-contained: builds its own QPU set, workload,
and scheduler configuration, then runs the tick loop and exports records.

Experiment groups
-----------------
E1  Plan A/B/C comparison          — overhead, fidelity, utilisation per plan kind
E2  Workload variation             — light vs heavy circuits, width/depth sweep
E3  Cutting algorithm comparison   — FitCut vs no-cut vs qiskit-addon baseline
E4  QPU diversity & congestion     — homogeneous vs heterogeneous pools, congestion
E5  Batch vs streaming submission  — all-at-once vs Poisson arrival

Usage
-----
    python run_experiments.py
    python run_experiments.py --outdir results/paper --n-jobs 80 --seed 42
    python run_experiments.py --experiments E1,E3   # run subset
    python run_experiments.py --fast                # smaller workloads for CI
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
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
from qdc_sched.cutting.qiskit_addon import QiskitAddonCutStrategy


# ---------------------------------------------------------------------------
# Circuit factories
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def circ_ghz(n: int, measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"ghz_{n}")
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    if measure:
        qc.measure_all()
    return qc


def circ_qft(n: int) -> QuantumCircuit:
    qc = QFT(n, do_swaps=False, name=f"qft_{n}")
    return qc.decompose()


def circ_random(n: int, depth: int, seed: int = 0, measure: bool = False) -> QuantumCircuit:
    rng = _rng(seed)
    qc = QuantumCircuit(n, name=f"rand_{n}_d{depth}")
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


def circ_qaoa(n: int, p: int = 1, seed: int = 0) -> QuantumCircuit:
    """QAOA-style ansatz: p rounds of alternating cost/mixer layers."""
    rng = _rng(seed)
    qc = QuantumCircuit(n, name=f"qaoa_{n}_p{p}")
    for i in range(n):
        qc.h(i)
    for _ in range(p):
        # cost layer: random ZZ couplings
        for i in range(0, n - 1, 2):
            gamma = rng.uniform(0, 3.14)
            qc.cx(i, i + 1)
            qc.rz(gamma, i + 1)
            qc.cx(i, i + 1)
        # mixer layer
        for i in range(n):
            beta = rng.uniform(0, 3.14)
            qc.rx(2 * beta, i)
    return qc


def circ_vqe(n: int, layers: int = 2, seed: int = 0) -> QuantumCircuit:
    """Hardware-efficient VQE ansatz: RY + entangling CX brickwork."""
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


def circ_qv(n: int, depth: int | None = None, seed: int = 0) -> QuantumCircuit:
    """Quantum Volume circuit: random SU(4) layers (approximated with CX+RZ+SX)."""
    d = depth or n
    rng = _rng(seed)
    qc = QuantumCircuit(n, name=f"qv_{n}_d{d}")
    for _ in range(d):
        perm = list(range(n))
        rng.shuffle(perm)
        for k in range(0, n - 1, 2):
            a, b = perm[k], perm[k + 1]
            qc.cx(a, b)
            qc.rz(rng.uniform(0, 3.14), a)
            qc.sx(b)
            qc.cx(b, a)
    return qc


def z_obs(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_list([("Z" * n, 1.0)])


# ---------------------------------------------------------------------------
# QPU factory helpers
# ---------------------------------------------------------------------------

def _line_graph(n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(i, i + 1) for i in range(n - 1)])
    return G


def _grid_graph(r: int, c: int) -> nx.Graph:
    G = nx.grid_2d_graph(r, c)
    return nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes())})


def _heavy_hex_graph(n: int) -> nx.Graph:
    """Approximate heavy-hex connectivity: line with every-other bridge edge."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    for i in range(0, n - 2, 4):
        if i + 2 < n:
            G.add_edge(i, i + 2)
    return G


def make_qpu(
    qpu_id: str,
    n: int,
    topology: str = "line",
    base_queue_delay_s: float = 0.3,
    twoq_error: float = 2.0e-2,
    oneq_error: float = 1.2e-3,
    twoq_gate_time_s: float = 300e-9,
    oneq_gate_time_s: float = 35e-9,
    meas_time_s: float = 1_000e-9,
    shot_overhead_s: float = 250e-6,
) -> QPUState:
    if topology == "grid":
        r = max(2, int(n ** 0.5))
        c = (n + r - 1) // r
        G = _grid_graph(r, c)
    elif topology == "heavy_hex":
        G = _heavy_hex_graph(n)
    else:
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


def default_qpu_pool(n_qpus: int = 2, n_qubits: int = 7, **qpu_kwargs) -> Dict[str, QPUState]:
    """Homogeneous line-topology pool — baseline for most experiments."""
    return {
        f"qpu_{chr(65+i)}": make_qpu(f"qpu_{chr(65+i)}", n_qubits, **qpu_kwargs)
        for i in range(n_qpus)
    }


# ---------------------------------------------------------------------------
# Scheduler builder
# ---------------------------------------------------------------------------

def build_scheduler(
    qpus: Dict[str, QPUState],
    *,
    allow_cutting: bool = True,
    allow_multi_qpu: bool = True,
    cut_strategy: str = "fitcut",   # "fitcut" | "qiskit_addon" | "pandora_optimized" | "pandora_widgetizer" | "none"
    max_cuts: int = 3,
    timing_mode: str = "analytic",
) -> Scheduler:
    if cut_strategy == "qiskit_addon":
        strategy = QiskitAddonCutStrategy()
    elif cut_strategy == "pandora_optimized":
        from qdc_sched.cutting.pandora_bridge import PandoraBridge
        from qdc_sched.cutting.pandora_optimizer import PandoraOptimizedCutStrategy
        bridge = PandoraBridge(config_path=os.environ.get("PANDORA_CONFIG_PATH", ""))
        strategy = PandoraOptimizedCutStrategy(bridge=bridge)
    elif cut_strategy == "pandora_widgetizer":
        from qdc_sched.cutting.pandora_bridge import PandoraBridge
        from qdc_sched.cutting.pandora_widgetizer import PandoraWidgetizerStrategy
        bridge = PandoraBridge(config_path=os.environ.get("PANDORA_CONFIG_PATH", ""))
        strategy = PandoraWidgetizerStrategy(bridge=bridge)
    elif cut_strategy == "none" or not allow_cutting:
        strategy = FitCutCutStrategy()   # won't be called when allow_cutting=False
    else:
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


# ---------------------------------------------------------------------------
# Tick loop
# ---------------------------------------------------------------------------

def run_tick_loop(
    sched: Scheduler,
    workload: List[Tuple[float, Job]],
    toggles: RunToggles,
    *,
    max_steps: int = 20000,
    step_dt_s: float = 0.5,
    idle_break: int = 5,
    max_job_age_s: float = 30.0,   # drop jobs that have been pending this long
) -> List[JobRunRecord]:
    now = 0.0
    next_idx = 0
    idle = 0
    job_submit_time: Dict[str, float] = {}

    for _ in range(max_steps):
        while next_idx < len(workload) and workload[next_idx][0] <= now + 1e-12:
            job = workload[next_idx][1]
            sched.submit(job, toggles)
            job_submit_time[job.job_id] = workload[next_idx][0]
            next_idx += 1

        # Expire jobs that have been pending too long (e.g. D_WAIT with tight SLO)
        if max_job_age_s > 0 and hasattr(sched, 'expire_pending'):
            sched.expire_pending(now_s=now, max_age_s=max_job_age_s)

        started = sched.step(now_s=now)

        if started:
            idle = 0
        else:
            idle += 1

        pending = sched.pending_count()
        active = sum(
            sum(1 for r in st.reservations if r.end_s > now)
            for st in sched.qpus.values()
        )

        if next_idx >= len(workload) and pending == 0 and active == 0:
            break
        if next_idx >= len(workload) and pending == 0 and idle >= idle_break:
            break
        # Safety: if all jobs submitted and we've been idle a long time, stop
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
# Record serialisation
# ---------------------------------------------------------------------------

def _safe(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 9)
    return v


def _det(rec: JobRunRecord, key: str, default: Any = "") -> Any:
    try:
        d = rec.details or {}
        return d.get(key, default)
    except Exception:
        return default


def record_to_row(rec: JobRunRecord, experiment: str, condition: str) -> Dict[str, Any]:
    """Flatten a JobRunRecord + details into a CSV-ready dict."""
    d = rec.details or {}
    tw = d.get("timing_wall_s", {}) or {}
    tm = d.get("timing_model_s", {}) or {}
    return {
        "experiment":          experiment,
        "condition":           condition,
        "job_id":              rec.job_id,
        "qpu_id":              _safe(rec.qpu_id),
        "plan_kind":           rec.plan_kind,
        "submit_time_s":       _safe(rec.submit_time_s),
        "end_to_end_s":        _safe(rec.end_to_end_s),
        "t_schedule_s":        _safe(rec.t_schedule_s),
        "t_execution_s":       _safe(rec.t_execution_s),
        "t_reconstruction_s":  _safe(rec.t_reconstruction_s),
        "fidelity_proxy":      _safe(rec.fidelity_proxy),
        "fidelity_estimated":  _safe(rec.fidelity_estimated),
        # timing breakdown from details
        "pred_exec_s":         _safe(_det(rec, "pred_exec_time_s")),
        "pred_queue_s":        _safe(_det(rec, "pred_queue_delay_s")),
        "pred_recon_s":        _safe(_det(rec, "pred_recon_s")),
        "pred_comm_s":         _safe(_det(rec, "pred_comm_s")),
        "sim_queue_wait_s":    _safe(_det(rec, "sim_queue_wait_s")),
        "sim_execution_span_s":_safe(_det(rec, "sim_execution_span_s")),
        "sim_comm_s":          _safe(_det(rec, "sim_comm_s")),
        "sim_recon_s":         _safe(_det(rec, "sim_recon_s")),
        "sim_latency_s":       _safe(_det(rec, "sim_latency_s")),
        "charged_comm_s":      _safe(_det(rec, "charged_comm_s")),
        "charged_recon_s":     _safe(_det(rec, "charged_recon_s")),
        "wall_exec_s":         _safe(tw.get("execute_s")),
        "wall_recon_s":        _safe(tw.get("reconstruct_s")),
        "wall_total_s":        _safe(tw.get("total_s")),
        "model_exec_s":        _safe(tm.get("execute_s")),
        "model_comm_s":        _safe(tm.get("communication_s")),
        "model_recon_s":       _safe(tm.get("reconstruct_s")),
        "sampling_overhead":   _safe(_det(rec, "sampling_overhead")),
        "labels_used":         _safe(_det(rec, "labels_used")),
        "k_wire":              _safe(_det(rec, "k_wire")),
        "k_gate":              _safe(_det(rec, "k_gate")),
        "pending_attempts":    _safe(_det(rec, "pending_attempts")),
        "schedule_wall_s":     _safe(_det(rec, "schedule_wall_s")),
        "queue_wait_s":        _safe(_det(rec, "queue_wait_s")),
        # Explicit scoring components from the three-term objective
        "score_qpu_completion_s": _safe((_det(rec, "score_components") or {}).get("qpu_completion_s")),
        "score_frag_penalty_s":   _safe((_det(rec, "score_components") or {}).get("frag_penalty_s")),
        "score_coord_penalty_s":  _safe((_det(rec, "score_components") or {}).get("coord_penalty_s")),
        "score_quality_term":     _safe((_det(rec, "score_components") or {}).get("quality_term")),
        "score_total":            _safe((_det(rec, "score_components") or {}).get("total_score")),
    }


FIELDNAMES = list(record_to_row(
    JobRunRecord("", None, "", 0, 0, 0, 0, 0, 0, 0, 0.0), "", ""
).keys())


def append_rows(path: str, rows: List[Dict], write_header: bool) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Workload builders
# ---------------------------------------------------------------------------

def light_workload(
    n_jobs: int,
    seed: int = 42,
    widths: List[int] = None,
    depths: List[int] = None,
    shots_choices: List[int] = None,
    families: List[str] = None,
    allow_cutting: bool = True,
    allow_multi_qpu: bool = True,
    poisson_rate: float = 2.0,     # jobs per second arrival rate
) -> List[Tuple[float, Job]]:
    """Mixed light workload: GHZ, QFT, random, QV circuits."""
    rng = _rng(seed)
    widths = widths or [3, 5, 7]
    depths = depths or [4, 6, 8]
    shots_choices = shots_choices or [500, 1000]
    families = families or ["ghz", "qft", "random", "qv"]

    wl = []
    t = 0.0
    for i in range(n_jobs):
        t += rng.expovariate(poisson_rate)
        fam = rng.choice(families)
        n = rng.choice(widths)
        d = rng.choice(depths)
        shots = rng.choice(shots_choices)
        s = rng.randint(0, 9999)

        if fam == "ghz":
            qc = circ_ghz(n)
        elif fam == "qft":
            qc = circ_qft(min(n, 7))
        elif fam == "qv":
            qc = circ_qv(n, seed=s)
        else:
            qc = circ_random(n, d, seed=s)

        qn = qc.num_qubits
        job = Job(
            job_id=f"L{i:04d}",
            circuit=qc,
            task_type="expectation",
            observables=z_obs(qn),
            shots=shots,
            submit_time_s=t,
            constraints=JobConstraints(
                allow_cutting=allow_cutting,
                allow_multi_qpu=allow_multi_qpu,
                max_cuts=3,
            ),
        )
        wl.append((t, job))
    return wl


def heavy_workload(
    n_jobs: int,
    seed: int = 42,
    widths: List[int] = None,
    allow_cutting: bool = True,
    allow_multi_qpu: bool = True,
    poisson_rate: float = 0.5,
) -> List[Tuple[float, Job]]:
    """Heavy workload: QAOA + VQE circuits, larger widths."""
    rng = _rng(seed)
    widths = widths or [8, 10, 12, 14]
    shots_choices = [500, 1000, 2000]

    wl = []
    t = 0.0
    for i in range(n_jobs):
        t += rng.expovariate(poisson_rate)
        n = rng.choice(widths)
        shots = rng.choice(shots_choices)
        s = rng.randint(0, 9999)
        fam = rng.choice(["qaoa", "vqe"])

        if fam == "qaoa":
            qc = circ_qaoa(n, p=rng.choice([1, 2]), seed=s)
        else:
            qc = circ_vqe(n, layers=rng.choice([2, 3]), seed=s)

        job = Job(
            job_id=f"H{i:04d}",
            circuit=qc,
            task_type="expectation",
            observables=z_obs(n),
            shots=shots,
            submit_time_s=t,
            constraints=JobConstraints(
                allow_cutting=allow_cutting,
                allow_multi_qpu=allow_multi_qpu,
                force_cutting=True,
                max_cuts=3,
            ),
        )
        wl.append((t, job))
    return wl


def mixed_workload(
    n_jobs: int,
    pct_heavy: float = 0.25,
    seed: int = 42,
    allow_cutting: bool = True,
    allow_multi_qpu: bool = True,
) -> List[Tuple[float, Job]]:
    rng = _rng(seed)
    n_heavy = int(round(n_jobs * pct_heavy))
    n_light = n_jobs - n_heavy
    wl = (
        light_workload(n_light, seed=seed, allow_cutting=allow_cutting, allow_multi_qpu=allow_multi_qpu)
        + heavy_workload(n_heavy, seed=seed+1, allow_cutting=allow_cutting, allow_multi_qpu=allow_multi_qpu)
    )
    # re-sort by submit time
    wl.sort(key=lambda x: x[0])
    return wl


def batch_workload(
    n_jobs: int,
    seed: int = 42,
    allow_cutting: bool = True,
) -> List[Tuple[float, Job]]:
    """All jobs submitted at t=0 simultaneously."""
    wl = light_workload(n_jobs, seed=seed, allow_cutting=allow_cutting, poisson_rate=1e9)
    return [(0.0, job) for _, job in wl]


def inject_congestion(qpus: Dict[str, QPUState], burst_starts: List[float], burst_dur: float = 12.0) -> None:
    """Block a fraction of each QPU at burst_starts to simulate congestion."""
    for i, (qid, st) in enumerate(qpus.items()):
        if i == 0:
            continue  # leave first QPU free
        k = max(1, st.profile.num_qubits // 2)
        for j, t0 in enumerate(burst_starts):
            st.reserve(f"CONGEST_{qid}_{j}", list(range(k)), start_s=t0, duration_s=burst_dur)


# ---------------------------------------------------------------------------
# Experiment E1: Plan A/B/C comparison
# ---------------------------------------------------------------------------

def exp_e1_plan_comparison(args, path: str) -> None:
    print("\n=== E1: Plan A/B/C comparison ===")
    n_jobs = args.n_jobs
    seed = args.seed

    conditions = [
        ("no_cut",    False, False),
        ("cut_single", True, False),
        ("cut_multi",  True, True),
    ]

    first = True
    for cname, allow_cut, allow_multi in conditions:
        print(f"  Running condition: {cname}")
        qpus = default_qpu_pool(n_qpus=2, n_qubits=7)
        sched = build_scheduler(qpus, allow_cutting=allow_cut, allow_multi_qpu=allow_multi)
        wl = mixed_workload(n_jobs, seed=seed, allow_cutting=allow_cut, allow_multi_qpu=allow_multi)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E1_plan_comparison", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        print(f"    → {len(records)} jobs completed")


# ---------------------------------------------------------------------------
# Experiment E2: Workload variation
# ---------------------------------------------------------------------------

def exp_e2_workload_variation(args, path: str) -> None:
    print("\n=== E2: Workload variation ===")

    conditions = [
        ("light_narrow",  light_workload(args.n_jobs, seed=args.seed, widths=[3, 5, 7])),
        ("light_wide",    light_workload(args.n_jobs, seed=args.seed, widths=[7, 9, 11])),
        ("heavy_qaoa_vqe",heavy_workload(args.n_jobs // 2, seed=args.seed)),
        ("mixed_25pct",   mixed_workload(args.n_jobs, pct_heavy=0.25, seed=args.seed)),
        ("mixed_50pct",   mixed_workload(args.n_jobs, pct_heavy=0.50, seed=args.seed)),
        ("ghz_only",      light_workload(args.n_jobs, seed=args.seed, families=["ghz"])),
        ("qft_only",      light_workload(args.n_jobs, seed=args.seed, families=["qft"])),
        ("random_only",   light_workload(args.n_jobs, seed=args.seed, families=["random"])),
    ]

    first = True
    for cname, wl in conditions:
        print(f"  Running condition: {cname}")
        qpus = default_qpu_pool(n_qpus=2, n_qubits=9)
        sched = build_scheduler(qpus)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E2_workload_variation", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        print(f"    → {len(records)} jobs completed")


# ---------------------------------------------------------------------------
# Experiment E3: Cutting algorithm comparison
# ---------------------------------------------------------------------------

def cutting_stress_workload(
    n_jobs: int,
    seed: int = 42,
    allow_cutting: bool = True,
    allow_multi_qpu: bool = True,
) -> List[Tuple[float, Job]]:
    """Circuits wider than a single 7-qubit QPU — forces the cutting path on every job.

    Width range 9-14 means no job fits on a 7-qubit QPU uncut. This ensures
    E3 comparisons between cutting strategies are actually meaningful.
    max_cuts=2 caps sampling_overhead at 4^2=16, keeping reconstruction times realistic.
    """
    rng = _rng(seed)
    wl = []
    t = 0.0
    for i in range(n_jobs):
        t += rng.expovariate(1.0)
        n = rng.choice([9, 10, 11, 12, 13, 14])
        shots = rng.choice([500, 1000])
        fam = rng.choice(["random", "qaoa", "ghz"])
        s = rng.randint(0, 9999)
        if fam == "qaoa":
            qc = circ_qaoa(n, p=1, seed=s)
        elif fam == "ghz":
            qc = circ_ghz(n)
        else:
            qc = circ_random(n, depth=max(6, n), seed=s)
        job = Job(
            job_id=f"CS{i:04d}",
            circuit=qc,
            task_type="expectation",
            observables=z_obs(n),
            shots=shots,
            submit_time_s=t,
            constraints=JobConstraints(
                allow_cutting=allow_cutting,
                allow_multi_qpu=allow_multi_qpu,
                force_cutting=allow_cutting,
                max_cuts=2,
            ),
        )
        wl.append((t, job))
    return wl


def _build_pandora_block(qc: "QuantumCircuit", qubits: list, rng: random.Random, depth: int, n_pairs: int) -> None:
    """Append a random FT-basis layer followed by within-block cancellable pairs onto qubits."""
    n = len(qubits)
    for _ in range(depth):
        q = qubits[rng.randint(0, n - 1)]
        r = rng.random()
        if r < 0.30:
            qc.t(q)
        elif r < 0.55:
            qc.tdg(q)
        elif r < 0.68:
            qc.h(q)
        elif r < 0.80:
            qi = rng.randint(0, n - 1)
            qj = (qi + 1) % n
            qc.cx(qubits[qi], qubits[qj])
        elif r < 0.90:
            qc.s(q)
        else:
            qc.sdg(q)

    for _ in range(n_pairs):
        # Single-qubit pairs only — guaranteed within this block after any cut.
        q = qubits[rng.randint(0, n - 1)]
        r = rng.random()
        if r < 0.40:
            qc.t(q);   qc.tdg(q)    # T / Tdg  → cancel
        elif r < 0.70:
            qc.h(q);   qc.h(q)      # H / H    → cancel
        else:
            qc.s(q);   qc.sdg(q)    # S / Sdg  → cancel


def pandora_stress_workload(n_jobs: int, seed: int = 42) -> List[Tuple[float, Job]]:
    """Blocked circuits with within-partition cancellable pairs.

    Each circuit has two qubit blocks connected by a small number of CX gates
    that serve as the natural cut boundary for FitCut:

        Block A (qubits 0..split-1)        Block B (qubits split..n-1)
        ┌─ FT basis + cancel pairs ─┐      ┌─ FT basis + cancel pairs ─┐
        │  T/Tdg, H/H, S/Sdg pairs  │──CX──│  T/Tdg, H/H, S/Sdg pairs  │
        └───────────────────────────┘      └───────────────────────────┘

    After FitCut splits at the inter-block CX, each subcircuit retains its own
    single-qubit cancellable pairs.  Pandora can then cancel them within each
    piece, giving measurable gate reduction and latency improvement.

    Only single-qubit pairs are used in the cancellation layer — they are always
    confined to one qubit and therefore always survive the cut intact.
    """
    rng = random.Random(seed)
    wl = []
    t = 0.0
    for i in range(n_jobs):
        t += rng.expovariate(1.0)
        n = rng.choice([10, 12, 14])          # even widths simplify equal split
        shots = rng.choice([500, 1000])
        split = n // 2                         # equal blocks
        qc = QuantumCircuit(n)

        block_a = list(range(split))
        block_b = list(range(split, n))

        # Each block: random FT-basis gates + cancellable pairs
        _build_pandora_block(qc, block_a, rng, depth=rng.randint(8, 14), n_pairs=rng.randint(5, 9))
        _build_pandora_block(qc, block_b, rng, depth=rng.randint(8, 14), n_pairs=rng.randint(5, 9))

        # Inter-block entanglement: 1-2 CX gates → FitCut's natural cut boundary
        for _ in range(rng.randint(1, 2)):
            qa = rng.choice(block_a)
            qb = rng.choice(block_b)
            qc.cx(qa, qb)

        job = Job(
            job_id=f"PS{i:04d}",
            circuit=qc,
            task_type="expectation",
            observables=z_obs(n),
            shots=shots,
            submit_time_s=t,
            constraints=JobConstraints(
                allow_cutting=True,
                allow_multi_qpu=True,
                force_cutting=True,
                max_cuts=2,
            ),
        )
        wl.append((t, job))
    return wl


def exp_e3_algorithm_comparison(args, path: str) -> None:
    print("\n=== E3: Cutting algorithm comparison ===")
    print("  Workload: all circuits width 9-14 (wider than 7-qubit QPUs), force_cutting=True")
    print("  max_cuts=2 so sampling_overhead stays at 4^2=16 max")

    conditions = [
        ("our_fitcut",         "fitcut",              True,  True),
        ("qiskit_addon",       "qiskit_addon",        True,  True),
        ("pandora_optimized",  "pandora_optimized",   True,  True),
        ("pandora_widgetizer", "pandora_widgetizer",  True,  True),
        ("no_cut_baseline",    "none",                False, False),
    ]

    first = True
    for cname, strategy, allow_cut, allow_multi in conditions:
        print(f"  Running condition: {cname}")
        qpus = default_qpu_pool(n_qpus=2, n_qubits=7)
        sched = build_scheduler(
            qpus,
            allow_cutting=allow_cut,
            allow_multi_qpu=allow_multi,
            cut_strategy=strategy,
            max_cuts=2,
        )
        wl = cutting_stress_workload(
            args.n_jobs, seed=args.seed,
            allow_cutting=allow_cut,
            allow_multi_qpu=allow_multi,
        )
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E3_algorithm_comparison", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        print(f"    → {len(records)} jobs completed")


# ---------------------------------------------------------------------------
# Experiment E3P: Pandora stress — cancellable-pair workload
# ---------------------------------------------------------------------------

def exp_e3p_pandora_stress(args, path: str) -> None:
    """E3P: Compare cutting strategies on circuits designed for Pandora optimization.

    Uses pandora_stress_workload() which generates two-block circuits where each
    block contains single-qubit cancellable pairs (T/Tdg, H/H, S/Sdg).  Blocks
    are connected by 1-2 inter-block CX gates that form FitCut's natural cut
    boundary.  After cutting, each subcircuit retains its own cancellable pairs
    so Pandora can reduce gate count within each piece.

    Also writes e3p_gate_counts.csv with per-circuit gate/depth before vs after
    Pandora optimization, so the raw optimization effect is separately visible.
    """
    print("\n=== E3P: Pandora stress workload (cancellable-pair circuits) ===")
    print("  Workload: blocked circuits; each partition has within-block T/Tdg, H/H, S/Sdg pairs")
    print("  Pandora cancels pairs within each subcircuit after FitCut splits at inter-block CX")

    gate_counts_path = os.path.join(os.path.dirname(path), "e3p_gate_counts.csv")
    # Always start fresh so stale data from previous runs doesn't corrupt the schema.
    if os.path.exists(gate_counts_path):
        os.remove(gate_counts_path)

    conditions = [
        ("fitcut_baseline",   "fitcut",            True, True),
        ("pandora_optimized", "pandora_optimized", True, True),
    ]

    timing_mode = getattr(args, "timing_mode", "analytic")
    print(f"  Timing mode: {timing_mode}")

    # shot_overhead_s=0 isolates gate-time differences; default 250µs/shot would
    # otherwise dominate and mask any Pandora gate-reduction benefit.
    first = True
    for cname, strategy_name, allow_cut, allow_multi in conditions:
        print(f"  Running condition: {cname}")
        qpus = default_qpu_pool(n_qpus=2, n_qubits=7)

        # Build strategy explicitly for pandora_optimized so we can read gate stats
        pandora_strategy_obj = None
        if strategy_name == "pandora_optimized":
            from qdc_sched.cutting.pandora_bridge import PandoraBridge
            from qdc_sched.cutting.pandora_optimizer import PandoraOptimizedCutStrategy
            bridge = PandoraBridge(config_path=os.environ.get("PANDORA_CONFIG_PATH", ""))
            pandora_strategy_obj = PandoraOptimizedCutStrategy(bridge=bridge)
            sched = build_scheduler(qpus, allow_cutting=allow_cut,
                                    allow_multi_qpu=allow_multi, cut_strategy="fitcut",
                                    max_cuts=2, timing_mode=timing_mode)
            sched.cfg.planner.cut_strategy = pandora_strategy_obj
        else:
            sched = build_scheduler(qpus, allow_cutting=allow_cut,
                                    allow_multi_qpu=allow_multi,
                                    cut_strategy=strategy_name, max_cuts=2,
                                    timing_mode=timing_mode)

        wl = pandora_stress_workload(args.n_jobs, seed=args.seed)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E3P_pandora_stress", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        print(f"    → {len(records)} jobs completed")

        # Write gate count stats for pandora_optimized
        if pandora_strategy_obj is not None and pandora_strategy_obj.gate_count_log:
            log = pandora_strategy_obj.gate_count_log
            gc_rows = [
                {
                    "circuit_idx": i,
                    "gates_before": b,
                    "gates_after": a,
                    "gates_cancelled": b - a,
                    "pct_reduction": round(100 * (b - a) / b, 2) if b else 0,
                    "depth_before": db,
                    "depth_after": da,
                    "depth_reduction": db - da,
                }
                for i, (b, a, db, da) in enumerate(log)
            ]
            gc_fieldnames = ["circuit_idx", "gates_before", "gates_after",
                             "gates_cancelled", "pct_reduction",
                             "depth_before", "depth_after", "depth_reduction"]
            gc_first = not os.path.exists(gate_counts_path)
            with open(gate_counts_path, "w" if gc_first else "a", newline="") as _f:
                _w = csv.DictWriter(_f, fieldnames=gc_fieldnames)
                if gc_first:
                    _w.writeheader()
                _w.writerows(gc_rows)
            avg_gates = sum(b for b, a, *_ in log) / len(log)
            avg_after = sum(a for b, a, *_ in log) / len(log)
            avg_db = sum(db for *_, db, da in log) / len(log)
            avg_da = sum(da for *_, db, da in log) / len(log)
            print(f"    gate reduction: {avg_gates:.1f} → {avg_after:.1f} "
                  f"({100*(avg_gates-avg_after)/avg_gates:.1f}%)")
            print(f"    depth reduction: {avg_db:.1f} → {avg_da:.1f} "
                  f"({100*(avg_db-avg_da)/avg_db:.1f}%)")


# ---------------------------------------------------------------------------
# Experiment E3PFT: Pandora in fault-tolerant regime (T-gate overhead = 50)
# ---------------------------------------------------------------------------

def exp_e3p_fault_tolerant(args, path: str) -> None:
    """E3PFT: Same workload as E3P but with fault-tolerant hardware model.

    Sets T-gate overhead = 50 (magic-state distillation cost), shot_overhead = 0,
    and communication/reconstruction overhead = 0 so that gate time alone
    determines latency.  In a real fault-tolerant machine the classical coordination
    model from the near-term DQC regime does not apply; this models the idealised
    case where T-gate cost (magic-state factories) dominates.

    The contrast with E3P (near-term, shot overhead dominates, no latency gain)
    illustrates where Pandora's T-gate reduction delivers value.
    """
    print("\n=== E3PFT: Pandora stress — fault-tolerant hardware model ===")
    print("  T-gate overhead = 50x (magic-state distillation), shot/comm/recon overhead = 0")
    print("  Pure gate-time regime: T-gate reduction directly lowers latency")

    conditions = [
        ("fitcut_baseline",   "fitcut",            True, True),
        ("pandora_optimized", "pandora_optimized", True, True),
    ]

    # Zero all non-gate overheads so gate time dominates.
    # Shot overhead is per-QPU; comm/recon overheads are read from env vars.
    _ft_env = {
        "QDC_COMM_BASE_S": "0.0",
        "QDC_COMM_PER_EXEC_S": "0.0",
        "QDC_COMM_PER_SAMPLE_S": "0.0",
        "QDC_COMM_COORD_PER_EXTRA_QPU_S": "0.0",
        "QDC_HOST_RECON_BASE_S": "0.0",
        "QDC_HOST_RECON_PER_EXEC_S": "0.0",
    }
    _saved_env = {k: os.environ.get(k) for k in _ft_env}
    os.environ.update(_ft_env)

    try:
        first = True
        for cname, strategy_name, allow_cut, allow_multi in conditions:
            print(f"  Running condition: {cname}")
            qpus = default_qpu_pool(n_qpus=2, n_qubits=7, shot_overhead_s=0.0)
            # Apply T-gate overhead to every QPU profile
            import dataclasses
            for qstate in qpus.values():
                qstate.profile = dataclasses.replace(qstate.profile, t_gate_overhead=50.0)

            pandora_strategy_obj = None
            if strategy_name == "pandora_optimized":
                from qdc_sched.cutting.pandora_bridge import PandoraBridge
                from qdc_sched.cutting.pandora_optimizer import PandoraOptimizedCutStrategy
                bridge = PandoraBridge(config_path=os.environ.get("PANDORA_CONFIG_PATH", ""))
                pandora_strategy_obj = PandoraOptimizedCutStrategy(bridge=bridge)
                sched = build_scheduler(qpus, allow_cutting=allow_cut,
                                        allow_multi_qpu=allow_multi, cut_strategy="fitcut",
                                        max_cuts=2, timing_mode="analytic")
                sched.cfg.planner.cut_strategy = pandora_strategy_obj
            else:
                sched = build_scheduler(qpus, allow_cutting=allow_cut,
                                        allow_multi_qpu=allow_multi,
                                        cut_strategy=strategy_name, max_cuts=2,
                                        timing_mode="analytic")

            wl = pandora_stress_workload(args.n_jobs, seed=args.seed)
            toggles = RunToggles(simulate_only=False)
            records = run_tick_loop(sched, wl, toggles)
            rows = [record_to_row(r, "E3PFT_pandora_stress", cname) for r in records]
            append_rows(path, rows, write_header=first)
            first = False
            print(f"    → {len(records)} jobs completed")

            if pandora_strategy_obj is not None and pandora_strategy_obj.gate_count_log:
                log = pandora_strategy_obj.gate_count_log
                avg_b = sum(b for b, a, *_ in log) / len(log)
                avg_a = sum(a for b, a, *_ in log) / len(log)
                print(f"    gate reduction: {avg_b:.1f} → {avg_a:.1f} "
                      f"({100*(avg_b-avg_a)/avg_b:.1f}%)")
    finally:
        # Restore original env vars
        for k, v in _saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Experiment E4: QPU diversity and congestion
# ---------------------------------------------------------------------------

def exp_e4_qpu_diversity(args, path: str) -> None:
    """E4 redesigned: aggressive error-rate sweep so quality-aware routing
    produces a large, clearly visible fidelity signal.

    Previous design: error rates 1.6%-2.4% -> fidelity diff ~0.026 (too small).
    New design: error rates 0.5%-5.0% -> fidelity diff ~0.20+ (clearly visible).

    Conditions:
      homog_uniform    — 3× QPU at 2.0% error. Baseline. Scheduler spreads load.
      heterog_quality  — 3× QPU at 0.5%, 2.0%, 5.0% error. Scheduler should
                         route 90%+ jobs to the 0.5% QPU -> large fidelity gain.
      heterog_capacity — QPUs differ in capacity: 5Q, 7Q, 11Q (all 2.0% error).
                         Scheduler routes wide circuits to largest QPU.
      congestion_best  — heterog_quality pool but the best QPU (0.5% error) is
                         partially congested. Forces use of 2nd-best QPU.
                         Shows graceful degradation.

    All conditions use force_cutting=True (wide circuits) so routing matters.
    """
    print("\n=== E4: QPU quality routing (redesigned) ===")
    print("  Error sweep: 0.5% vs 2.0% vs 5.0% — large fidelity signal")
    print("  Wide-circuit workload (force_cutting) so quality routing is exercised")

    def _homog_uniform():
        return {
            "qpu_A": make_qpu("qpu_A", 7, topology="line", twoq_error=2.0e-2),
            "qpu_B": make_qpu("qpu_B", 7, topology="line", twoq_error=2.0e-2),
            "qpu_C": make_qpu("qpu_C", 7, topology="line", twoq_error=2.0e-2),
        }

    def _heterog_quality():
        return {
            "qpu_A": make_qpu("qpu_A", 7, topology="line", twoq_error=5.0e-2),   # worst
            "qpu_B": make_qpu("qpu_B", 7, topology="line", twoq_error=2.0e-2),   # mid
            "qpu_C": make_qpu("qpu_C", 7, topology="line", twoq_error=0.5e-2),   # best
        }

    def _heterog_capacity():
        return {
            "qpu_A": make_qpu("qpu_A",  5, topology="line", twoq_error=2.0e-2),   # small
            "qpu_B": make_qpu("qpu_B",  7, topology="line", twoq_error=2.0e-2),   # medium
            "qpu_C": make_qpu("qpu_C", 11, topology="heavy_hex", twoq_error=2.0e-2), # large
        }

    pool_configs = [
        ("homog_uniform",    _homog_uniform,    False),
        ("heterog_quality",  _heterog_quality,  False),
        ("heterog_capacity", _heterog_capacity, False),
        ("congestion_best",  _heterog_quality,  True),   # best QPU partially blocked
    ]

    first = True
    for cname, pool_fn, with_congestion in pool_configs:
        print(f"  Running condition: {cname}")
        qpus = pool_fn()
        if with_congestion:
            # Block the BEST QPU (qpu_C) to force rerouting to qpu_B
            st = qpus["qpu_C"]
            k = max(1, st.profile.num_qubits // 2)
            for j, t0 in enumerate([1.0, 6.0, 12.0]):
                st.reserve(f"CONGEST_C_{j}", list(range(k)), start_s=t0, duration_s=5.0)
        sched = build_scheduler(qpus, allow_cutting=True, allow_multi_qpu=True, max_cuts=2)

        if cname == "heterog_capacity":
            # For capacity experiment: use mixed-width workload WITHOUT force_cutting
            # so that the 11Q QPU can run 9-10Q circuits as Plan A (its key advantage).
            # force_cutting=True would force every circuit to cut even when it fits.
            wl = cutting_stress_workload(args.n_jobs, seed=args.seed,
                                         allow_cutting=True, allow_multi_qpu=True)
            # Override: allow no-cut for circuits that fit on large QPU
            wl2 = []
            for t, job in wl:
                from dataclasses import replace
                new_constraints = replace(job.constraints, force_cutting=False)
                wl2.append((t, replace(job, constraints=new_constraints)))
            wl = wl2
        else:
            # Quality conditions: use wide circuits so routing to best QPU matters
            wl = cutting_stress_workload(args.n_jobs, seed=args.seed,
                                         allow_cutting=True, allow_multi_qpu=True)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E4_qpu_diversity", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        plans = collections.Counter(r.plan_kind for r in records)
        fids = [r.fidelity_proxy for r in records if r.fidelity_proxy]
        import statistics as _st
        print(f"    → {len(records)} jobs  plans={dict(plans)}"
              f"  fid_med={_st.median(fids) if fids else 0:.3f}")


# ---------------------------------------------------------------------------
# Experiment E5: Batch vs streaming
# ---------------------------------------------------------------------------

def exp_e5_batch_vs_stream(args, path: str) -> None:
    print("\n=== E5: Batch vs streaming submission ===")

    conditions = [
        ("stream_fast",   light_workload(args.n_jobs, seed=args.seed, poisson_rate=4.0)),
        ("stream_slow",   light_workload(args.n_jobs, seed=args.seed, poisson_rate=0.5)),
        ("batch_all",     batch_workload(args.n_jobs, seed=args.seed)),
        ("mixed_stream",  mixed_workload(args.n_jobs, seed=args.seed)),
    ]

    first = True
    for cname, wl in conditions:
        print(f"  Running condition: {cname}")
        qpus = default_qpu_pool(n_qpus=2, n_qubits=7)
        sched = build_scheduler(qpus)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E5_batch_vs_stream", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        print(f"    → {len(records)} jobs completed")


# ---------------------------------------------------------------------------
# Utilisation summary (written alongside main CSV)
# ---------------------------------------------------------------------------

def write_utilisation(
    path: str,
    experiment: str,
    condition: str,
    qpus: Dict[str, QPUState],
    records: List[JobRunRecord],
) -> None:
    """Write per-QPU utilisation rows to a separate CSV."""
    upath = path.replace(".csv", "_utilisation.csv")
    fields = ["experiment","condition","qpu_id","n_jobs","total_exec_s","active_span_s","utilisation_pct"]

    # Per-QPU total execution time from records
    qpu_exec: Dict[str, float] = {}
    for rec in records:
        qid = str(rec.qpu_id or "")
        if not qid or qid == "MULTI":
            continue
        qpu_exec[qid] = qpu_exec.get(qid, 0.0) + float(rec.t_execution_s or 0.0)

    total_span = max((float(r.end_to_end_s or 0) + float(r.submit_time_s or 0) for r in records), default=1.0)

    rows = []
    for qid in qpus:
        exec_s = qpu_exec.get(qid, 0.0)
        util = 100.0 * exec_s / total_span if total_span > 0 else 0.0
        rows.append({
            "experiment": experiment, "condition": condition,
            "qpu_id": qid,
            "n_jobs": sum(1 for r in records if str(r.qpu_id or "") == qid),
            "total_exec_s": round(exec_s, 4),
            "active_span_s": round(total_span, 4),
            "utilisation_pct": round(util, 2),
        })

    write_header = not os.path.exists(upath)
    os.makedirs(os.path.dirname(upath) if os.path.dirname(upath) else ".", exist_ok=True)
    with open(upath, "a" if not write_header else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows(rows)



# ---------------------------------------------------------------------------
# Experiment E6: Circuit width sweep
# ---------------------------------------------------------------------------

def exp_e6_width_sweep(args, path: str) -> None:
    """Sweep circuit width from 3 to 14 qubits on a fixed 5-qubit QPU pool.

    At widths <= 7: circuits fit without cutting (Plan A).
    At widths > 7: cutting is required (Plans B/C).
    Shows exactly where the cut boundary falls and the cost of crossing it.
    """
    print("\n=== E6: Circuit width sweep ===")
    print("  Fixed pool: 2× 5-qubit QPUs | widths 3,5,6,8,10,12 | 15 jobs each")

    widths = [3, 5, 6, 8, 10, 12]  # 5Q boundary between 5Q and 6Q
    rng = _rng(args.seed)
    first = True

    for width in widths:
        cname = f"width_{width:02d}q"
        print(f"  width={width}q...")
        qpus = default_qpu_pool(n_qpus=2, n_qubits=5)
        sched = build_scheduler(qpus, allow_cutting=True, allow_multi_qpu=True, max_cuts=2)

        # Build width-specific workload
        wl = []
        t = 0.0
        n_jobs_width = max(15, args.n_jobs // 6)
        for i in range(n_jobs_width):
            t += rng.expovariate(1.5)
            fam = rng.choice(["random", "ghz", "qaoa"] if width >= 6 else ["ghz", "random"])
            s = rng.randint(0, 9999)
            shots = rng.choice([500, 1000])
            if fam == "qaoa":
                qc = circ_qaoa(width, p=1, seed=s)
            elif fam == "ghz":
                qc = circ_ghz(width)
            else:
                qc = circ_random(width, depth=max(4, width), seed=s)
            job = Job(
                job_id=f"W{width}_{i:03d}",
                circuit=qc,
                task_type="expectation",
                observables=z_obs(width),
                shots=shots,
                submit_time_s=t,
                constraints=JobConstraints(
                    allow_cutting=True,
                    allow_multi_qpu=True,
                    force_cutting=(width > 5),
                    max_cuts=2,
                ),
            )
            wl.append((t, job))

        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E6_width_sweep", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        plans = collections.Counter(r.plan_kind for r in records)
        print(f"    → {len(records)} jobs: {dict(plans)}")







# ---------------------------------------------------------------------------
# Experiment E7: Objective weight sensitivity
# ---------------------------------------------------------------------------

def exp_e7_weight_sensitivity(args, path: str) -> None:
    """Show how each term in the three-term objective changes plan selection.

    Sweeps weight configurations to demonstrate that each penalty term has
    a measurable effect on plan choice, latency, fidelity, and utilization.
    This directly validates the objective formulation in the paper.

    Conditions:
      baseline    — default weights (w_time=1, w_frag=1, w_coord=1, w_qual=1)
      frag_heavy  — w_frag=5: strongly penalises over-fragmented plans (fewer labels)
      coord_heavy — w_coord=5: strongly penalises Plan C (prefers Plan B)
      quality_heavy — w_qual=5: routes to highest-fidelity QPUs
      time_heavy  — w_time=5: aggressively minimises QPU completion time
      no_penalties — w_frag=0, w_coord=0: reverts to simple time+quality
    """
    print("\n=== E7: Objective weight sensitivity ===")

    conditions = [
        ("baseline",      1.0, 1.0, 1.0, 1.0),
        ("frag_heavy",    1.0, 5.0, 1.0, 1.0),
        ("coord_heavy",   1.0, 1.0, 5.0, 1.0),
        ("quality_heavy", 1.0, 1.0, 1.0, 5.0),
        ("time_heavy",    5.0, 1.0, 1.0, 1.0),
        ("no_penalties",  1.0, 0.0, 0.0, 1.0),
    ]

    wl = mixed_workload(args.n_jobs, seed=args.seed)

    first = True
    for cname, w_time, w_frag, w_coord, w_qual in conditions:
        print(f"  condition={cname}: w_time={w_time} w_frag={w_frag} w_coord={w_coord} w_qual={w_qual}")
        qpus = {
            "qpu_A": make_qpu("qpu_A", 7, topology="line",     twoq_error=2.4e-2),
            "qpu_B": make_qpu("qpu_B", 9, topology="grid",     twoq_error=2.0e-2),
            "qpu_C": make_qpu("qpu_C", 11,topology="heavy_hex",twoq_error=1.6e-2),
        }
        quality = QualityModel(
            noise_models={},
            qpu_profiles={qid: st.profile for qid, st in qpus.items()},
        )
        from qdc_sched.core.planner import PlannerConfig
        from qdc_sched.cutting.base import CutConstraints
        from qdc_sched.cutting.fitcut import FitCutCutStrategy
        planner_cfg = PlannerConfig(
            weight_time=w_time,
            weight_frag=w_frag,
            weight_coord=w_coord,
            weight_quality=w_qual,
            cut_constraints=CutConstraints(max_cuts=2, target_labels_cap=4, seed_tries=2),
            cut_strategy=FitCutCutStrategy(),
        )
        from qdc_sched.core.scheduler import SchedulerConfig
        from qdc_sched.core.executor import ExecConfig
        cfg = SchedulerConfig(planner=planner_cfg, exec_cfg=ExecConfig(reserve_nonsim=True))
        sched = Scheduler(qpus=qpus, quality=quality, cfg=cfg)

        job_wl = mixed_workload(args.n_jobs, seed=args.seed)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, job_wl, toggles)
        rows = [record_to_row(r, "E7_weight_sensitivity", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False

        plans = collections.Counter(r.plan_kind for r in records)
        e2e_vals = [r.end_to_end_s for r in records if r.end_to_end_s]
        import statistics as _st
        e2e_med = _st.median(e2e_vals) if e2e_vals else 0
        print(f"    -> {len(records)} jobs  plans={dict(plans)}  e2e_med={e2e_med:.3f}s")


# ---------------------------------------------------------------------------
# Experiment E8: Fragmentation penalty effect
# ---------------------------------------------------------------------------

def exp_e8_fragmentation(args, path: str) -> None:
    """Isolate the fragmentation penalty's effect on cutting decisions.

    Uses only wide circuits (width 10-14) that require cutting, so every
    job must choose between 2-label and 4-label partitions. The fragmentation
    penalty directly controls which is chosen.

    Conditions:
      no_frag_penalty  — w_frag=0: chooses labels purely based on QPU completion
      low_frag_penalty — w_frag=0.5: mild preference for fewer labels
      default_frag     — w_frag=1.0: balanced (default)
      high_frag_penalty— w_frag=3.0: strongly prefers 2-label cuts
    """
    print("\n=== E8: Fragmentation penalty effect ===")

    conditions = [
        ("no_frag_penalty",   0.0),
        ("low_frag_penalty",  0.5),
        ("default_frag",      1.0),
        ("high_frag_penalty", 3.0),
    ]

    first = True
    for cname, w_frag in conditions:
        print(f"  condition={cname}: w_frag={w_frag}")
        qpus = default_qpu_pool(n_qpus=2, n_qubits=7)
        quality = QualityModel(
            noise_models={},
            qpu_profiles={qid: st.profile for qid, st in qpus.items()},
        )
        from qdc_sched.core.planner import PlannerConfig
        from qdc_sched.cutting.base import CutConstraints
        from qdc_sched.cutting.fitcut import FitCutCutStrategy
        planner_cfg = PlannerConfig(
            weight_frag=w_frag,
            cut_constraints=CutConstraints(max_cuts=2, target_labels_cap=4, seed_tries=2),
            cut_strategy=FitCutCutStrategy(),
        )
        from qdc_sched.core.scheduler import SchedulerConfig
        from qdc_sched.core.executor import ExecConfig
        cfg = SchedulerConfig(planner=planner_cfg, exec_cfg=ExecConfig(reserve_nonsim=True))
        sched = Scheduler(qpus=qpus, quality=quality, cfg=cfg)

        # Wide-only workload: all circuits need cutting
        wl = cutting_stress_workload(args.n_jobs, seed=args.seed,
                                     allow_cutting=True, allow_multi_qpu=True)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E8_fragmentation", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False

        plans = collections.Counter(r.plan_kind for r in records)
        labels_used = [r.details.get("labels_used", 0) for r in records
                       if isinstance(r.details, dict) and r.details.get("labels_used")]
        import statistics as _st
        lbl_med = _st.median(labels_used) if labels_used else 0
        print(f"    -> {len(records)} jobs  plans={dict(plans)}  median_labels={lbl_med:.1f}")


# ---------------------------------------------------------------------------
# Experiment E9: Coordination penalty — Plan B vs C preference
# ---------------------------------------------------------------------------

def exp_e9_coordination(args, path: str) -> None:
    """Show how the coordination penalty controls Plan B vs C selection.

    With low coord penalty, Plan C (multi-QPU) is preferred whenever it
    offers lower QPU completion time. With high coord penalty, Plan B
    (single-QPU sequential) is preferred even if it is slower, because
    multi-QPU coordination cost offsets the parallelism benefit.

    This experiment directly answers: 'When is single-QPU cutting better
    than multi-QPU cutting?' The answer is when coord_penalty > parallelism_gain.
    """
    print("\n=== E9: Coordination penalty — Plan B vs C ===")

    conditions = [
        ("prefer_multi",   0.0),    # no coord penalty: always use multi-QPU if faster
        ("mild_coord",     0.03),   # 30ms per extra QPU
        ("default_coord",  0.05),   # 50ms per extra QPU (default)
        ("high_coord",     0.20),   # 200ms: strongly prefer single-QPU
        ("extreme_coord",  0.50),   # 500ms: almost never use Plan C
    ]

    first = True
    for cname, coord_per_extra in conditions:
        print(f"  condition={cname}: coord_per_extra_qpu={coord_per_extra:.2f}s")
        qpus = {
            "qpu_A": make_qpu("qpu_A", 7, topology="line"),
            "qpu_B": make_qpu("qpu_B", 7, topology="line"),
        }
        quality = QualityModel(
            noise_models={},
            qpu_profiles={qid: st.profile for qid, st in qpus.items()},
        )
        from qdc_sched.core.planner import PlannerConfig
        from qdc_sched.cutting.base import CutConstraints
        from qdc_sched.cutting.fitcut import FitCutCutStrategy
        planner_cfg = PlannerConfig(
            coord_per_extra_qpu=coord_per_extra,
            cut_constraints=CutConstraints(max_cuts=2, target_labels_cap=4, seed_tries=2),
            cut_strategy=FitCutCutStrategy(),
        )
        from qdc_sched.core.scheduler import SchedulerConfig
        from qdc_sched.core.executor import ExecConfig
        cfg = SchedulerConfig(planner=planner_cfg, exec_cfg=ExecConfig(reserve_nonsim=True))
        sched = Scheduler(qpus=qpus, quality=quality, cfg=cfg)

        wl = cutting_stress_workload(args.n_jobs, seed=args.seed,
                                     allow_cutting=True, allow_multi_qpu=True)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E9_coordination", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False

        plans = collections.Counter(r.plan_kind for r in records)
        plan_b = plans.get("B_CUT_SINGLE_SEQ", 0)
        plan_c = plans.get("C_CUT_MULTI_QPU", 0)
        total_cut = plan_b + plan_c
        pct_c = 100 * plan_c / total_cut if total_cut > 0 else 0
        print(f"    -> {len(records)} jobs  B={plan_b} C={plan_c}  Plan C rate={pct_c:.0f}%")





# ---------------------------------------------------------------------------
# Experiment E10: Weight sensitivity on wide-circuit workload
# ---------------------------------------------------------------------------

def exp_e10_weight_sensitivity_wide(args, path: str) -> None:
    """E10: same weight conditions as E7 but all circuits require cutting.
    Isolates penalty effects by removing Plan A jobs from the picture.
    Expected: no_penalties shows latency spike; coord_heavy reduces Plan C.
    """
    print("\n=== E10: Weight sensitivity (wide-circuit workload) ===")
    conditions = [
        ("baseline",      1.0, 1.0, 1.0, 1.0),
        ("coord_heavy",   1.0, 1.0, 5.0, 1.0),
        ("quality_heavy", 1.0, 1.0, 1.0, 5.0),
        ("no_penalties",  1.0, 0.0, 0.0, 1.0),
    ]
    first = True
    for cname, w_time, w_frag, w_coord, w_qual in conditions:
        print(f"  {cname}: w_time={w_time} w_frag={w_frag} w_coord={w_coord} w_qual={w_qual}")
        qpus = {
            "qpu_A": make_qpu("qpu_A", 7, topology="line",     twoq_error=2.4e-2),
            "qpu_B": make_qpu("qpu_B", 9, topology="grid",     twoq_error=2.0e-2),
            "qpu_C": make_qpu("qpu_C", 11,topology="heavy_hex",twoq_error=1.6e-2),
        }
        quality = QualityModel(
            noise_models={},
            qpu_profiles={qid: st.profile for qid, st in qpus.items()},
        )
        from qdc_sched.core.planner import PlannerConfig
        from qdc_sched.cutting.base import CutConstraints
        from qdc_sched.cutting.fitcut import FitCutCutStrategy
        planner_cfg = PlannerConfig(
            weight_time=w_time, weight_frag=w_frag,
            weight_coord=w_coord, weight_quality=w_qual,
            cut_constraints=CutConstraints(max_cuts=2, target_labels_cap=4, seed_tries=2),
            cut_strategy=FitCutCutStrategy(),
        )
        from qdc_sched.core.scheduler import SchedulerConfig
        from qdc_sched.core.executor import ExecConfig
        cfg = SchedulerConfig(planner=planner_cfg, exec_cfg=ExecConfig(reserve_nonsim=True))
        sched = Scheduler(qpus=qpus, quality=quality, cfg=cfg)
        # Wide-circuit workload: all jobs require cutting
        wl = cutting_stress_workload(args.n_jobs, seed=args.seed,
                                     allow_cutting=True, allow_multi_qpu=True)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E10_weight_wide", cname) for r in records]
        append_rows(path, rows, write_header=first); first = False
        import statistics as _st
        e2e_vals = [r.end_to_end_s for r in records if r.end_to_end_s]
        plans = collections.Counter(r.plan_kind for r in records)
        print(f"    -> {len(records)} jobs  plans={dict(plans)}  e2e_med={_st.median(e2e_vals):.3f}s")


# ---------------------------------------------------------------------------
# Experiment E11: SLO-constrained scheduling
# ---------------------------------------------------------------------------

def exp_e11_slo_constrained(args, path: str) -> None:
    """E11: Show how strict SLOs change plan selection and compliance.
    Tighter deadlines force the scheduler toward Plan A or D_WAIT,
    demonstrating graceful degradation under latency constraints.

    Uses explicit PlannerConfig so predicted_total_time_s is computed
    correctly (not inflated by sampling_overhead multiplier).
    """
    print("\n=== E11: SLO-constrained scheduling ===")
    # SLO thresholds calibrated against actual job latencies:
    # no_slo: no constraint
    # slo_3s: loose -- most jobs pass
    # slo_1s: tight -- cut jobs borderline
    # slo_0s: very tight -- only fastest Plan A jobs pass
    conditions = [
        ("no_slo",   None),
        ("slo_3s",   3.0),
        ("slo_1s",   1.0),
        ("slo_05s",  0.5),
    ]
    first = True
    for cname, slo in conditions:
        print(f"  {cname}: slo={slo}")
        qpus = {
            "qpu_A": make_qpu("qpu_A", 7, topology="line",     twoq_error=2.4e-2),
            "qpu_B": make_qpu("qpu_B", 9, topology="grid",     twoq_error=2.0e-2),
        }
        quality = QualityModel(
            noise_models={},
            qpu_profiles={qid: st.profile for qid, st in qpus.items()},
        )
        from qdc_sched.core.planner import PlannerConfig
        from qdc_sched.cutting.base import CutConstraints
        from qdc_sched.cutting.fitcut import FitCutCutStrategy
        planner_cfg = PlannerConfig(
            cut_constraints=CutConstraints(max_cuts=2, target_labels_cap=4, seed_tries=2),
            cut_strategy=FitCutCutStrategy(),
        )
        from qdc_sched.core.scheduler import SchedulerConfig
        from qdc_sched.core.executor import ExecConfig
        cfg = SchedulerConfig(planner=planner_cfg, exec_cfg=ExecConfig(reserve_nonsim=True))
        sched = Scheduler(qpus=qpus, quality=quality, cfg=cfg)

        rng = _rng(args.seed)
        wl = []
        t = 0.0
        for i in range(args.n_jobs):
            t += rng.expovariate(1.5)
            width = rng.choice([5, 7, 8, 9, 10, 12])
            fam = rng.choice(["ghz", "random", "qaoa"] if width >= 8 else ["ghz", "random"])
            s = rng.randint(0, 9999)
            shots = rng.choice([500, 1000])
            if fam == "qaoa": qc = circ_qaoa(width, p=1, seed=s)
            elif fam == "ghz": qc = circ_ghz(width)
            else: qc = circ_random(width, depth=max(4, width), seed=s)
            from qdc_sched.core.types import JobConstraints
            constraints = JobConstraints(
                allow_cutting=True, allow_multi_qpu=True, max_cuts=2,
                slo_s=float(slo) if slo is not None else None,
            )
            job = Job(
                job_id=f"SLO_{i:03d}", circuit=qc,
                task_type="expectation", observables=z_obs(width),
                shots=shots, submit_time_s=t, constraints=constraints,
            )
            wl.append((t, job))
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles, idle_break=15, max_job_age_s=20.0)
        rows = [record_to_row(r, "E11_slo_constrained", cname) for r in records]
        append_rows(path, rows, write_header=first); first = False
        plans = collections.Counter(r.plan_kind for r in records)
        import statistics as _st
        e2e = [r.end_to_end_s for r in records if r.end_to_end_s]
        pct_wait = 100*plans.get("D_WAIT",0)/max(len(records),1)
        print(f"    -> {len(records)} jobs  plans={dict(plans)}  wait={pct_wait:.0f}%  "
              f"e2e_med={_st.median(e2e) if e2e else 0:.3f}s")




# ---------------------------------------------------------------------------
# Experiment E12: QPU pool scaling — how throughput and latency scale with
# the number of available QPUs under a heavy cutting workload.
#
# Design:
#   - Sweep n_qpus in [2, 3, 4, 6, 8], each QPU is 7-qubit line topology
#   - Workload: QAOA/VQE circuits width 8–14 (force_cutting=True)
#     → every job must cut, so Plan C opportunities scale with QPU count
#   - Fixed: 60 jobs per condition, seed=args.seed
#   - Metrics: throughput, e2e latency, queue wait, plan mix (B vs C),
#     QPU utilization
#
# Expected findings:
#   - Throughput increases as n_qpus grows (more parallelism for Plan C)
#   - Queue wait drops sharply from 2→4 QPUs then plateaus (saturation)
#   - Plan C fraction rises with n_qpus (more multi-QPU opportunities)
#   - Utilization per QPU drops at large n_qpus (diminishing returns)
# ---------------------------------------------------------------------------

def exp_e12_qpu_scaling(args, path: str) -> None:
    """E12: QPU pool size scaling with a heavy cutting workload."""
    print("\n=== E12: QPU pool scaling ===")
    print("  Workload: QAOA/VQE width 8-14, force_cutting=True, batch submission (t=0)")
    print("  Batch mode stresses QPU pool so n_qpus difference is measurable")
    print("  Pool: n_qpus × 7-qubit line QPUs, homogeneous")

    n_qpu_values = [2, 3, 4, 6, 8]
    first = True

    for n_qpus in n_qpu_values:
        cname = f"n{n_qpus}_qpu"
        print(f"  n_qpus={n_qpus}...")

        # base_queue_delay_s=1.0 simulates realistic cloud-like per-QPU latency.
        # This creates genuine queuing under batch load so that pool size matters:
        # with 2 QPUs each job waits ~11s on average; with 8 QPUs ~3s.
        # Without elevated queue delay, analytic exec times (~0.13s/shot) are so
        # small that even 2 QPUs drain 40 jobs in seconds with no queuing.
        qpus = {
            f"qpu_{chr(65+i)}": make_qpu(
                f"qpu_{chr(65+i)}", 7, topology="line",
                twoq_error=2.0e-2, base_queue_delay_s=1.0,
            )
            for i in range(n_qpus)
        }
        sched = build_scheduler(qpus, allow_cutting=True, allow_multi_qpu=True, max_cuts=2)

        # Batch submission (all at t=0) so QPU pool is the bottleneck.
        wl_raw = heavy_workload(
            args.n_jobs, seed=args.seed + n_qpus,
            widths=[8, 10, 12, 14],
            allow_cutting=True, allow_multi_qpu=True,
        )
        wl = [(0.0, job) for _, job in wl_raw]

        toggles = RunToggles(simulate_only=False)
        # idle_break=50 ensures the loop doesn't exit while jobs are still queued
        records = run_tick_loop(sched, wl, toggles,
                                idle_break=50, max_steps=100000,
                                max_job_age_s=300.0)
        rows = [record_to_row(r, "E12_qpu_scaling", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False

        plans = collections.Counter(r.plan_kind for r in records)
        import statistics as _st
        e2e = [r.end_to_end_s for r in records if r.end_to_end_s]
        submits = [r.submit_time_s or 0 for r in records]
        finish = [s + (r.end_to_end_s or 0) for s, r in zip(submits, records)]
        span = max(finish) - min(submits) if finish else 1
        tp = len(records) / span if span > 0 else 0
        print(f"    → {len(records)} jobs  plans={dict(plans)}")
        print(f"       e2e_med={_st.median(e2e) if e2e else 0:.3f}s  throughput={tp:.2f} jobs/s")
        write_utilisation(path, "E12_qpu_scaling", cname, qpus, records)


# ---------------------------------------------------------------------------
# Experiment E13: Backend configuration comparison
#
# Design:
#   Six pool configurations, all using a wide QAOA/VQE workload (8-12Q).
#   Conditions vary QPU count, size, topology, and error rate.
#   ibm_like_2/4:      IBM heavy_hex 7Q, 2% error
#   mixed_quality_2/4: mixed error rates (0.5% best + 3% mid)
#   homog_small_4:     4× small 5Q QPUs, 2% error
#   homog_large_2:     2× large 11Q QPUs, 2% error
#
# Expected findings:
#   - mixed_quality: scheduler routes to best QPU → higher fidelity
#   - homog_large_2: wide circuits fit without cutting → lower overhead
#   - homog_small_4: more parallelism but must always cut → Plan C dominant
#   - IBM×4 vs IBM×2: more QPUs reduces queue wait
# ---------------------------------------------------------------------------

def exp_e13_backend_comparison(args, path: str) -> None:
    print("\n=== E13: Backend configuration comparison ===")

    def _pool_ibm(n):
        return {f"qpu_{chr(65+i)}": make_qpu(f"qpu_{chr(65+i)}", 7,
                topology="heavy_hex", twoq_error=2.0e-2, base_queue_delay_s=0.5)
                for i in range(n)}

    def _pool_mixed_quality(n):
        # Alternate between good and mid-quality QPUs
        pool = {}
        for i in range(n):
            err = 0.5e-2 if i % 2 == 0 else 3.0e-2
            pool[f"qpu_{chr(65+i)}"] = make_qpu(f"qpu_{chr(65+i)}", 7,
                topology="line", twoq_error=err, base_queue_delay_s=0.5)
        return pool

    def _pool_small(n):
        return {f"qpu_{chr(65+i)}": make_qpu(f"qpu_{chr(65+i)}", 5,
                topology="line", twoq_error=2.0e-2, base_queue_delay_s=0.5)
                for i in range(n)}

    def _pool_large(n):
        return {f"qpu_{chr(65+i)}": make_qpu(f"qpu_{chr(65+i)}", 11,
                topology="heavy_hex", twoq_error=2.0e-2, base_queue_delay_s=0.5)
                for i in range(n)}

    configs = [
        ("ibm_like_2",       lambda: _pool_ibm(2)),
        ("ibm_like_4",       lambda: _pool_ibm(4)),
        ("mixed_quality_2",  lambda: _pool_mixed_quality(2)),
        ("mixed_quality_4",  lambda: _pool_mixed_quality(4)),
        ("homog_small_4",    lambda: _pool_small(4)),
        ("homog_large_2",    lambda: _pool_large(2)),
    ]

    first = True
    for cname, pool_fn in configs:
        print(f"  {cname}...")
        qpus = pool_fn()
        sched = build_scheduler(qpus, allow_cutting=True, allow_multi_qpu=True, max_cuts=2)

        # Mixed workload: circuits 5-14Q wide WITHOUT force_cutting.
        # This lets large QPUs (11Q) serve 5-10Q circuits via Plan A,
        # which is the key differentiator vs small-QPU pools.
        # Batch submission (all t=0) so QPU pool is the bottleneck,
        # making throughput differences across pool configs visible.
        # 50% heavy (8-14Q) so pool differences are visible:
        # small QPUs must cut all heavy circuits, large QPUs run many as Plan A
        wl_raw = mixed_workload(args.n_jobs, pct_heavy=0.5,
                                seed=args.seed + hash(cname) % 1000)
        wl = [(0.0, job) for _, job in wl_raw]  # batch

        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles,
                                idle_break=40, max_steps=100000)
        rows = [record_to_row(r, "E13_backend_comparison", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        plans = collections.Counter(r.plan_kind for r in records)
        fids = [r.fidelity_proxy for r in records if r.fidelity_proxy]
        import statistics as _st
        print(f"    → {len(records)} jobs  plans={dict(plans)}"
              f"  fid_med={_st.median(fids) if fids else 0:.3f}")



# ---------------------------------------------------------------------------
# Experiment E14: Error rate sensitivity
#
# Design:
#   Fixed 3-QPU pool with homogeneous error rate swept across conditions.
#   The scheduler's quality-aware routing is exercised: it always prefers
#   the QPU with lowest error rate when possible.
#   Metrics: fidelity (primary), plan mix, queue wait.
#
# Expected findings:
#   - Higher error rate → lower fidelity (monotonic)
#   - For heterogeneous conditions (mixed error rates), scheduler routes
#     majority of jobs to lowest-error QPU → higher aggregate fidelity
#     than naive round-robin would achieve
#   - Plan mix relatively stable (error rate doesn't change circuit width)
# ---------------------------------------------------------------------------

def exp_e14_noise_sensitivity(args, path: str) -> None:
    """E14: QPU error rate sensitivity and quality-aware routing effect."""
    print("\n=== E14: Noise sensitivity sweep ===")

    # Homogeneous pools at different error rates
    homog_conditions = [
        ("err_0p5pct",  0.5e-2),
        ("err_1pct",    1.0e-2),
        ("err_2pct",    2.0e-2),
        ("err_5pct",    5.0e-2),
        ("err_10pct",  10.0e-2),
    ]
    # Heterogeneous: 3 QPUs at mixed error rates — scheduler routes to best
    heterog_conditions = [
        ("heterog_mild",   [0.5e-2, 2.0e-2, 5.0e-2]),   # mild spread
        ("heterog_wide",   [0.5e-2, 2.0e-2, 10.0e-2]),  # wide spread
        ("heterog_equal",  [2.0e-2, 2.0e-2, 2.0e-2]),   # baseline (same as err_2pct)
    ]

    first = True
    # Homogeneous sweep
    for cname, err in homog_conditions:
        print(f"  {cname} (err={err*100:.1f}%)...")
        qpus = {f"qpu_{chr(65+i)}": make_qpu(f"qpu_{chr(65+i)}", 7,
                topology="line", twoq_error=err, base_queue_delay_s=0.3)
                for i in range(3)}
        sched = build_scheduler(qpus, allow_cutting=True, allow_multi_qpu=True, max_cuts=2)
        wl = mixed_workload(args.n_jobs, seed=args.seed)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E14_noise_sensitivity", cname) for r in records]
        append_rows(path, rows, write_header=first); first = False
        fids = [r.fidelity_proxy for r in records if r.fidelity_proxy]
        import statistics as _st
        print(f"    → {len(records)} jobs  fid_med={_st.median(fids) if fids else 0:.3f}")

    # Heterogeneous pools — key comparison vs homogeneous baseline
    for cname, errs in heterog_conditions:
        print(f"  {cname} (errs={[f'{e*100:.1f}%' for e in errs]})...")
        qpus = {f"qpu_{chr(65+i)}": make_qpu(f"qpu_{chr(65+i)}", 7,
                topology="line", twoq_error=e, base_queue_delay_s=0.3)
                for i, e in enumerate(errs)}
        sched = build_scheduler(qpus, allow_cutting=True, allow_multi_qpu=True, max_cuts=2)
        wl = mixed_workload(args.n_jobs, seed=args.seed)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles)
        rows = [record_to_row(r, "E14_noise_sensitivity", cname) for r in records]
        append_rows(path, rows, write_header=False)
        fids = [r.fidelity_proxy for r in records if r.fidelity_proxy]
        by_qpu = {}
        for r in records:
            qid = getattr(r, 'qpu_id', None)
            if qid and qid != 'MULTI':
                by_qpu[qid] = by_qpu.get(qid, 0) + 1
        import statistics as _st
        print(f"    → {len(records)} jobs  fid_med={_st.median(fids) if fids else 0:.3f}  routing={by_qpu}")

# ---------------------------------------------------------------------------
# Experiment E15: Streaming load vs plan selection
#
# Design:
#   Wide-circuit workload (force_cutting=True, widths 8-12Q on 2x7Q QPUs)
#   so EVERY job must cut. Sweep Poisson arrival rate from slow to batch.
#   At low rate: Plan B sufficient (no queue pressure).
#   At high rate: queue builds -> Plan C (multi-QPU parallel) drains faster.
#
# Expected findings:
#   - Plan C fraction rises with arrival rate
#   - Queue wait rises then stabilises once Plan C engages
#   - Throughput peaks around lambda=2-4 (QPU saturation)
# ---------------------------------------------------------------------------

def exp_e15_streaming_load(args, path: str) -> None:
    """E15: Arrival rate effect on plan selection (wide circuits)."""
    print("\n=== E15: Streaming load sweep (wide circuits, force_cutting) ===")

    conditions = [
        ("lambda_0p25", 0.25),
        ("lambda_0p5",  0.50),
        ("lambda_1p0",  1.00),
        ("lambda_2p0",  2.00),
        ("lambda_4p0",  4.00),
        ("lambda_batch", None),
    ]

    first = True
    for cname, lam in conditions:
        print(f"  {cname} (lam={lam})...")
        qpus = {
            "qpu_A": make_qpu("qpu_A", 7, topology="line",
                              twoq_error=2.0e-2, base_queue_delay_s=0.3),
            "qpu_B": make_qpu("qpu_B", 7, topology="line",
                              twoq_error=2.0e-2, base_queue_delay_s=0.3),
        }
        sched = build_scheduler(qpus, allow_cutting=True, allow_multi_qpu=True, max_cuts=2)

        rng = _rng(args.seed + hash(cname) % 1000)
        wl = []
        t = 0.0
        for i in range(args.n_jobs):
            if lam is None:
                t = 0.0
            else:
                t += rng.expovariate(lam)
            width = rng.choice([8, 9, 10, 11, 12])
            shots = rng.choice([500, 1000])
            fam = rng.choice(["qaoa", "random", "ghz"])
            s = rng.randint(0, 9999)
            if fam == "qaoa":
                qc = circ_qaoa(width, p=1, seed=s)
            elif fam == "ghz":
                qc = circ_ghz(width)
            else:
                qc = circ_random(width, depth=max(6, width), seed=s)
            job = Job(
                job_id=f"S{i:04d}",
                circuit=qc,
                task_type="expectation",
                observables=z_obs(width),
                shots=shots,
                submit_time_s=t,
                constraints=JobConstraints(
                    allow_cutting=True,
                    allow_multi_qpu=True,
                    force_cutting=True,
                    max_cuts=2,
                ),
            )
            wl.append((t, job))

        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles,
                                idle_break=20, max_steps=50000,
                                max_job_age_s=120.0)
        rows = [record_to_row(r, "E15_streaming_load", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        import statistics as _st
        plans = collections.Counter(r.plan_kind for r in records)
        e2e = [r.end_to_end_s for r in records if r.end_to_end_s]
        pct_c = 100 * plans.get("C_CUT_MULTI_QPU", 0) / max(len(records), 1)
        print(f"    -> {len(records)} jobs  C:{pct_c:.0f}%  e2e_med={_st.median(e2e) if e2e else 0:.3f}s")


# ---------------------------------------------------------------------------
# Experiment E16: Congestion sweep
#
# Design:
#   Heterogeneous quality pool (0.5%/2%/5% error). Sweep the fraction of
#   best QPU (qpu_C, 0.5%) that is pre-reserved: 0->100% blocked.
#   Wide circuits (force_cutting) so quality routing always matters.
#
# Expected findings:
#   - At 0%: high fidelity (routes to qpu_C)
#   - Fidelity degrades smoothly as qpu_C fills
#   - Queue wait rises with congestion
#   - Routing shifts: qpu_C -> qpu_B -> qpu_A
# ---------------------------------------------------------------------------

def exp_e16_congestion_sweep(args, path: str) -> None:
    """E16: Graceful fidelity degradation as best QPU fills."""
    print("\n=== E16: Congestion sweep on heterogeneous quality pool ===")

    congestion_levels = [
        ("cong_00pct",  0.00),
        ("cong_25pct",  0.25),
        ("cong_50pct",  0.50),
        ("cong_75pct",  0.75),
        ("cong_100pct", 1.00),
    ]

    first = True
    for cname, frac in congestion_levels:
        print(f"  {cname} (best QPU {int(frac*100)}% blocked)...")
        qpus = {
            "qpu_A": make_qpu("qpu_A", 7, topology="line", twoq_error=5.0e-2),
            "qpu_B": make_qpu("qpu_B", 7, topology="line", twoq_error=2.0e-2),
            "qpu_C": make_qpu("qpu_C", 7, topology="line", twoq_error=0.5e-2),
        }

        if frac > 0:
            st_c = qpus["qpu_C"]
            n_block = max(1, int(round(st_c.profile.num_qubits * frac)))
            for t0 in [0.0, 10.0, 20.0, 40.0, 80.0, 120.0]:
                st_c.reserve(
                    f"CONG_{cname}_{t0:.0f}",
                    list(range(n_block)),
                    start_s=t0,
                    duration_s=20.0,
                )

        sched = build_scheduler(qpus, allow_cutting=True, allow_multi_qpu=True, max_cuts=2)
        wl = cutting_stress_workload(args.n_jobs, seed=args.seed,
                                      allow_cutting=True, allow_multi_qpu=True)
        toggles = RunToggles(simulate_only=False)
        records = run_tick_loop(sched, wl, toggles,
                                idle_break=20, max_steps=50000,
                                max_job_age_s=120.0)
        rows = [record_to_row(r, "E16_congestion_sweep", cname) for r in records]
        append_rows(path, rows, write_header=first)
        first = False
        import statistics as _st
        fids = [r.fidelity_proxy for r in records if r.fidelity_proxy]
        routing = collections.Counter(
            r.qpu_id for r in records if r.qpu_id and r.qpu_id != "MULTI"
        )
        print(f"    -> {len(records)} jobs  fid_med={_st.median(fids) if fids else 0:.3f}  routing={dict(routing)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_seed_list(seed_arg: str) -> list[int]:
    seeds = []
    for part in seed_arg.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds provided to --seeds")
    return seeds


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--outdir", default="results/experiments",
                   help="Output directory for CSVs")
    p.add_argument("--n-jobs", type=int, default=60,
                   help="Jobs per condition (default 60)")
    p.add_argument("--seed", type=int, default=2026,
                   help="Single seed for one run (default 2026)")
    p.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seed list for repeated runs, e.g. 2026,2027,2028",
    )
    p.add_argument(
        "--experiments",
        default="E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,E13,E14,E15,E16",
        help="Comma-separated list of experiments to run",
    )
    p.add_argument("--fast", action="store_true",
                   help="Use n_jobs=20 for quick iteration")
    p.add_argument(
        "--timing-mode",
        default=os.getenv("QDC_QPU_TIMING_MODE", "analytic"),
        choices=["analytic", "aer", "backend_profile"],
        help="QPU execution timing model (default: analytic). "
             "'aer' measures real Aer wall-clock time per subcircuit. "
             "Also controlled by QDC_QPU_TIMING_MODE env var.",
    )
    return p.parse_args()


def run_selected_experiments(args, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    wanted = {e.strip().upper() for e in args.experiments.split(",")}

    dispatch = {
        "E1": (exp_e1_plan_comparison,    "e1_plan_comparison.csv"),
        "E2": (exp_e2_workload_variation, "e2_workload_variation.csv"),
        "E3": (exp_e3_algorithm_comparison, "e3_algorithm_comparison.csv"),
        "E3P":   (exp_e3p_pandora_stress,        "e3p_pandora_stress.csv"),
        "E3PFT": (exp_e3p_fault_tolerant,        "e3p_fault_tolerant.csv"),
        "E4": (exp_e4_qpu_diversity,      "e4_qpu_diversity.csv"),
        "E5": (exp_e5_batch_vs_stream,    "e5_batch_vs_stream.csv"),
        "E6": (exp_e6_width_sweep,        "e6_width_sweep.csv"),
        "E7": (exp_e7_weight_sensitivity, "e7_weight_sensitivity.csv"),
        "E8": (exp_e8_fragmentation,      "e8_fragmentation.csv"),
        "E9": (exp_e9_coordination,       "e9_coordination.csv"),
        "E10": (exp_e10_weight_sensitivity_wide, "e10_weight_sensitivity_wide.csv"),
        "E11": (exp_e11_slo_constrained,  "e11_slo_constrained.csv"),
        "E12": (exp_e12_qpu_scaling,      "e12_qpu_scaling.csv"),
        "E13": (exp_e13_backend_comparison, "e13_backend_comparison.csv"),
        "E14": (exp_e14_noise_sensitivity, "e14_noise_sensitivity.csv"),
        "E15": (exp_e15_streaming_load,   "e15_streaming_load.csv"),
        "E16": (exp_e16_congestion_sweep, "e16_congestion_sweep.csv"),
    }

    print(f"[EXPERIMENTS] outdir={outdir}  n_jobs={args.n_jobs}  seed={args.seed}")
    print(f"[EXPERIMENTS] running: {sorted(wanted)}")
    t0 = time.time()

    for key in ["E1","E2","E3","E3P","E3PFT","E4","E5","E6","E7","E8","E9","E10","E11","E12","E13","E14","E15","E16"]:
        if key not in wanted:
            continue
        fn, fname = dispatch[key]
        path = os.path.join(outdir, fname)
        print(f"\n--- {key} -> {path} ---")
        fn(args, path)
        print(f"  [saved] {path}")

    print(f"\n[EXPERIMENTS] done in {time.time()-t0:.1f}s")
    print(f"[EXPERIMENTS] results in: {outdir}/")

def main():
    args = parse_args()
    if args.fast:
        args.n_jobs = 20

    if args.seeds.strip():
        seeds = parse_seed_list(args.seeds)
        print(f"[EXPERIMENTS] multi-seed mode: {seeds}")

        for seed in seeds:
            args.seed = seed
            seed_outdir = os.path.join(args.outdir, f"seed_{seed}")
            print(f"\n==============================")
            print(f"[EXPERIMENTS] seed={seed}")
            print(f"==============================")
            run_selected_experiments(args, seed_outdir)
    else:
        run_selected_experiments(args, args.outdir)


if __name__ == "__main__":
    main()