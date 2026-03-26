from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

try:
    from qiskit.circuit.library import QFTGate, QuantumVolume
except Exception:
    QFTGate = None
    QuantumVolume = None

try:
    from qiskit.circuit.library import quantum_volume as quantum_volume_fn
except Exception:
    quantum_volume_fn = None

try:
    from qiskit.synthesis.qft import synth_qft_full
except Exception:
    synth_qft_full = None

from qdc_sched.core.hardware import HardwareProfile, QPUState
from qdc_sched.core.quality import QualityModel
from qdc_sched.core.scheduler import Scheduler, SchedulerConfig
from qdc_sched.core.executor import ExecConfig
from qdc_sched.core.planner import PlannerConfig
from qdc_sched.core.types import Job, RunToggles, JobConstraints
from qdc_sched.cutting import FitCutCutStrategy, QiskitAddonCutStrategy
from qdc_sched.cutting.base import CutAnalysis, CutConstraints, CutStrategy, PartitionPlan


# -------------------------
# Circuit families
# -------------------------

def ghz_circuit(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    return qc


def qft_circuit(n: int) -> QuantumCircuit:
    if synth_qft_full is not None:
        try:
            return synth_qft_full(num_qubits=n, do_swaps=False, approximation_degree=0)
        except Exception:
            pass
    if QFTGate is not None:
        try:
            return QuantumCircuit(n).compose(QFTGate(n), inplace=False).decompose()
        except Exception:
            pass
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
    return qc


def random_cx_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(max(1, depth)):
        a = rng.randrange(n)
        b = rng.randrange(n)
        while b == a:
            b = rng.randrange(n)
        qc.h(a)
        qc.cx(a, b)
        if rng.random() < 0.4:
            qc.rx(rng.random() * math.pi, b)
        if rng.random() < 0.3:
            qc.rz(rng.random() * math.pi, a)
    return qc


def qaoa_ring_circuit(n: int, p: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    gammas = [0.15 + 0.2 * rng.random() for _ in range(max(1, p))]
    betas = [0.25 + 0.2 * rng.random() for _ in range(max(1, p))]
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for layer in range(max(1, p)):
        gamma = gammas[layer]
        beta = betas[layer]
        for q in range(n):
            qc.rzz(2.0 * gamma, q, (q + 1) % n)
        for q in range(n):
            qc.rx(2.0 * beta, q)
    return qc


def vqe_hwe_circuit(n: int, reps: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(max(1, reps)):
        for q in range(n):
            qc.ry(rng.random() * math.pi, q)
            qc.rz(rng.random() * math.pi, q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def qv_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    if quantum_volume_fn is not None:
        try:
            return quantum_volume_fn(num_qubits=n, depth=max(1, depth), seed=seed).decompose()
        except Exception:
            pass
    if QuantumVolume is not None:
        try:
            return QuantumVolume(num_qubits=n, depth=max(1, depth), seed=seed).decompose()
        except Exception:
            pass
    return random_cx_circuit(n, depth=max(3, depth * 2), seed=seed)


def z_observable(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_list([("Z" * n, 1.0)])


# -------------------------
# Naive baseline cutting strategy
# -------------------------

class NaiveChunkCutStrategy(CutStrategy):
    name = "naive_chunk"

    def analyze(self, circuit: QuantumCircuit, constraints: CutConstraints, context: Dict[str, Any]) -> CutAnalysis:
        max_local = int((context or {}).get("max_local_qubits", 0) or 0)
        if max_local <= 0:
            return CutAnalysis(feasible=False, reason="missing_max_local_qubits")
        if circuit.num_qubits <= max_local:
            return CutAnalysis(feasible=True, reason="fits_without_cutting", est_executions=1, est_quality_delta=0.0)
        n_parts = math.ceil(circuit.num_qubits / max_local)
        if n_parts - 1 > int(constraints.max_cuts or 0) and int(constraints.max_cuts or 0) > 0:
            return CutAnalysis(feasible=False, reason="max_cuts_exceeded")
        return CutAnalysis(feasible=True, reason="naive_chunk", est_executions=max(1, n_parts * 100), est_quality_delta=0.05 * n_parts)

    def partition(self, circuit: QuantumCircuit, constraints: CutConstraints, context: Dict[str, Any]) -> PartitionPlan:
        max_local = max(1, int((context or {}).get("max_local_qubits", 1) or 1))
        subcircuits: List[QuantumCircuit] = []
        remaining = circuit.num_qubits
        idx = 0
        while remaining > 0:
            k = min(max_local, remaining)
            sub = QuantumCircuit(k, name=f"naive_{idx}")
            # Very cheap approximate structure: local H layer only.
            for q in range(k):
                sub.h(q)
            subcircuits.append(sub)
            remaining -= k
            idx += 1
        return PartitionPlan(
            kind="naive_chunk",
            subcircuits=subcircuits,
            reconstruction={
                "subexperiments": {},
                "coefficients": {},
                "subobservables": {i: [] for i in range(len(subcircuits))},
                "meta": {"baseline": "naive_chunk"},
            },
            est_executions=max(1, len(subcircuits) * 100),
            k_wire=max(0, len(subcircuits) - 1),
            k_gate=0,
        )


# -------------------------
# Experiment presets
# -------------------------

@dataclass
class MethodPreset:
    name: str
    cut_strategy: str
    allow_cutting: bool = True
    allow_multi_qpu: bool = True
    force_cut_wide: bool = False
    planner_env: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class WorkloadPreset:
    name: str
    description: str
    n_jobs: int
    arrival_gap_choices: Sequence[float]
    heavy_fraction: float
    force_wide_fraction: float
    light_widths: Sequence[int]
    heavy_widths: Sequence[int]
    light_depths: Sequence[int]
    heavy_depths: Sequence[int]
    shots_light: Sequence[int]
    shots_heavy: Sequence[int]
    families_light: Sequence[str]
    families_heavy: Sequence[str]
    fits_without_cutting: bool
    align_wide_to_bursts: bool = True
    burst_starts: Sequence[float] = (15.0, 40.0)
    congestion_targets: Sequence[str] = ("qpu_B",)
    congestion_duration_s: float = 15.0
    congestion_block_fraction: float = 1.0


@dataclass
class RunSpec:
    suite: str
    workload: WorkloadPreset
    method: MethodPreset
    seed: int
    full_eval: bool

    @property
    def run_name(self) -> str:
        eval_tag = "full" if self.full_eval else "timing"
        return f"{self.suite}__{self.workload.name}__{self.method.name}__seed{self.seed}__{eval_tag}"


DEFAULT_METHODS: Dict[str, MethodPreset] = {
    "scheduler_aware_fitcut": MethodPreset(
        name="scheduler_aware_fitcut",
        cut_strategy="fitcut",
        allow_cutting=True,
        allow_multi_qpu=True,
        force_cut_wide=True,
        notes="Main proposed method.",
    ),
    "scheduler_aware_addon": MethodPreset(
        name="scheduler_aware_addon",
        cut_strategy="addon",
        allow_cutting=True,
        allow_multi_qpu=True,
        force_cut_wide=True,
        notes="Qiskit addon inside scheduler.",
    ),
    "scheduler_aware_naive": MethodPreset(
        name="scheduler_aware_naive",
        cut_strategy="naive",
        allow_cutting=True,
        allow_multi_qpu=True,
        force_cut_wide=True,
        notes="Naive chunk cutting inside scheduler.",
    ),
    "no_cut_scheduler": MethodPreset(
        name="no_cut_scheduler",
        cut_strategy="fitcut",
        allow_cutting=False,
        allow_multi_qpu=False,
        force_cut_wide=False,
        notes="Same scheduler, cutting disabled.",
    ),
    "cut_single_seq_fitcut": MethodPreset(
        name="cut_single_seq_fitcut",
        cut_strategy="fitcut",
        allow_cutting=True,
        allow_multi_qpu=False,
        force_cut_wide=True,
        notes="Cutting allowed, but no multi-QPU Plan C.",
    ),
}


DEFAULT_WORKLOADS: Dict[str, WorkloadPreset] = {
    "light_fit": WorkloadPreset(
        name="light_fit",
        description="Mostly small jobs that fit on one backend.",
        n_jobs=24,
        arrival_gap_choices=(0.0, 0.1, 0.2, 0.5, 1.0),
        heavy_fraction=0.10,
        force_wide_fraction=0.0,
        light_widths=(4, 6, 8),
        heavy_widths=(10, 12),
        light_depths=(4, 6, 8),
        heavy_depths=(10, 14),
        shots_light=(128, 256, 512),
        shots_heavy=(256, 512, 1024),
        families_light=("ghz", "qft", "random", "qaoa"),
        families_heavy=("qft", "random", "vqe"),
        fits_without_cutting=True,
    ),
    "mixed_fit": WorkloadPreset(
        name="mixed_fit",
        description="Mixed practical workload with some heavier jobs that still fit the largest backend.",
        n_jobs=30,
        arrival_gap_choices=(0.0, 0.1, 0.2, 0.5, 1.0),
        heavy_fraction=0.35,
        force_wide_fraction=0.0,
        light_widths=(4, 6, 8),
        heavy_widths=(10, 12, 14),
        light_depths=(4, 8, 12),
        heavy_depths=(12, 18, 24),
        shots_light=(128, 256, 512),
        shots_heavy=(256, 512, 1024),
        families_light=("ghz", "qft", "random"),
        families_heavy=("random", "qaoa", "vqe", "qv"),
        fits_without_cutting=True,
    ),
    "mixed_forcecut": WorkloadPreset(
        name="mixed_forcecut",
        description="Bursty mixed workload with heavy jobs larger than any single backend.",
        n_jobs=24,
        arrival_gap_choices=(0.0, 0.1, 0.2, 0.5, 1.0),
        heavy_fraction=0.50,
        force_wide_fraction=0.35,
        light_widths=(4, 6, 8),
        heavy_widths=(12, 16, 18),
        light_depths=(4, 8, 12),
        heavy_depths=(16, 24, 32),
        shots_light=(128, 256, 512),
        shots_heavy=(256, 512, 1024),
        families_light=("ghz", "qft", "random"),
        families_heavy=("random", "qaoa", "vqe", "qv"),
        fits_without_cutting=False,
    ),
    "bottleneck_bursty_fit": WorkloadPreset(
        name="bottleneck_bursty_fit",
        description="Bursty workload that still fits the largest backend, but creates a strong qpu_C bottleneck unless jobs are cut and distributed.",
        n_jobs=36,
        arrival_gap_choices=(0.0, 0.0, 0.0, 0.1, 0.2, 0.5),
        heavy_fraction=0.70,
        force_wide_fraction=0.60,
        light_widths=(4, 6, 8),
        heavy_widths=(10, 12, 14),
        light_depths=(4, 8, 12),
        heavy_depths=(18, 24, 32, 40),
        shots_light=(128, 256, 512),
        shots_heavy=(512, 1024, 2048),
        families_light=("ghz", "qft", "random"),
        families_heavy=("random", "qaoa", "vqe", "qv", "qft"),
        fits_without_cutting=True,
        align_wide_to_bursts=True,
        burst_starts=(8.0, 20.0, 32.0),
        congestion_targets=("qpu_B", "qpu_C"),
        congestion_duration_s=12.0,
        congestion_block_fraction=0.70,
    ),
    "heavy_forcecut": WorkloadPreset(
        name="heavy_forcecut",
        description="Very heavy bursty workload with many jobs exceeding any single backend, designed to force Plan C or fail without cutting.",
        n_jobs=28,
        arrival_gap_choices=(0.0, 0.0, 0.1, 0.2, 0.5),
        heavy_fraction=0.85,
        force_wide_fraction=0.75,
        light_widths=(6, 8),
        heavy_widths=(16, 18, 20),
        light_depths=(8, 12),
        heavy_depths=(20, 28, 36, 44),
        shots_light=(256, 512),
        shots_heavy=(512, 1024, 2048),
        families_light=("ghz", "qft"),
        families_heavy=("random", "qaoa", "vqe", "qv"),
        fits_without_cutting=False,
        align_wide_to_bursts=True,
        burst_starts=(10.0, 18.0, 26.0, 34.0),
        congestion_targets=("qpu_B", "qpu_C"),
        congestion_duration_s=14.0,
        congestion_block_fraction=0.85,
    ),
    "quality_small": WorkloadPreset(
        name="quality_small",
        description="Small family sweep for full-evaluation quality experiments.",
        n_jobs=15,
        arrival_gap_choices=(0.0, 0.1, 0.2),
        heavy_fraction=0.40,
        force_wide_fraction=0.0,
        light_widths=(4, 6),
        heavy_widths=(8, 10, 12),
        light_depths=(4, 6),
        heavy_depths=(8, 12, 16),
        shots_light=(128, 256),
        shots_heavy=(256, 512),
        families_light=("ghz", "qft", "qaoa"),
        families_heavy=("random", "vqe", "qv"),
        fits_without_cutting=True,
        align_wide_to_bursts=False,
    ),
}


SUITES: Dict[str, Dict[str, Any]] = {
        "system_value": {
        "workloads": ["light_fit", "mixed_fit", "bottleneck_bursty_fit"],
        "methods": ["scheduler_aware_fitcut", "no_cut_scheduler", "cut_single_seq_fitcut"],
        "full_eval": False,
    },
        "forcecut_value": {
        "workloads": ["bottleneck_bursty_fit", "mixed_forcecut", "heavy_forcecut"],
        "methods": ["scheduler_aware_fitcut", "no_cut_scheduler", "cut_single_seq_fitcut", "scheduler_aware_naive"],
        "full_eval": False,
    },
        "cut_method": {
        "workloads": ["mixed_fit", "bottleneck_bursty_fit", "mixed_forcecut", "heavy_forcecut"],
        "methods": ["scheduler_aware_fitcut", "scheduler_aware_addon", "scheduler_aware_naive"],
        "full_eval": False,
    },
    "quality_small": {
        "workloads": ["quality_small"],
        "methods": ["scheduler_aware_fitcut", "scheduler_aware_addon", "scheduler_aware_naive", "no_cut_scheduler"],
        "full_eval": True,
    },
    "paper_main": {
        "workloads": ["bottleneck_bursty_fit", "mixed_forcecut"],
        "methods": ["scheduler_aware_fitcut", "no_cut_scheduler", "cut_single_seq_fitcut", "scheduler_aware_naive"],
        "full_eval": False,
    },
}


# -------------------------
# Helper functions
# -------------------------

def make_line_qpu(qpu_id: str, n: int, base_queue_delay_s: float) -> QPUState:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from((i, i + 1) for i in range(n - 1))
    hp = HardwareProfile(qpu_id=qpu_id, num_qubits=n, coupling_graph=g, base_queue_delay_s=base_queue_delay_s)
    return QPUState(hp)


def inject_congestion_bursts(qpus: Dict[str, QPUState], burst_starts: Sequence[float], burst_dur: float, targets: Sequence[str], block_fraction: float = 1.0) -> None:
    for qid in targets:
        if qid not in qpus:
            continue
        k_full = int(qpus[qid].profile.num_qubits)
        k = max(1, min(k_full, int(round(k_full * block_fraction))))
        for i, t0 in enumerate(burst_starts):
            qpus[qid].reserve(f"BLOCK_{qid}_{i}", list(range(k)), start_s=float(t0), duration_s=float(burst_dur))


def build_family_circuit(family: str, n: int, depth: int, seed: int) -> QuantumCircuit:
    fam = family.strip().lower()
    if fam == "ghz":
        return ghz_circuit(n)
    if fam == "qft":
        return qft_circuit(n)
    if fam == "random":
        return random_cx_circuit(n, depth, seed)
    if fam == "qaoa":
        return qaoa_ring_circuit(n, max(1, depth // 6), seed)
    if fam == "vqe":
        return vqe_hwe_circuit(n, max(1, depth // 6), seed)
    if fam == "qv":
        return qv_circuit(n, max(3, depth // 3), seed)
    raise ValueError(f"Unsupported family: {family}")


def _planned_wide_submit_times(n_wide: int, rng: random.Random, burst_starts: Sequence[float], enabled: bool) -> List[float]:
    if not enabled or n_wide <= 0 or not burst_starts:
        return []
    jitter_choices = (-2.0, -1.0, -0.5, 0.0, 0.2, 0.5, 1.0, 2.0)
    anchors = list(burst_starts) * ((n_wide + len(burst_starts) - 1) // len(burst_starts))
    anchors = anchors[:n_wide]
    rng.shuffle(anchors)
    out = [max(0.0, float(a) + float(rng.choice(jitter_choices))) for a in anchors]
    out.sort()
    return out


def make_workload(preset: WorkloadPreset, seed: int, method: MethodPreset) -> Tuple[List[Tuple[float, Job]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    wl: List[Tuple[float, Job]] = []
    manifest_rows: List[Dict[str, Any]] = []
    t = 0.0

    n_force = int(round(preset.n_jobs * preset.force_wide_fraction))
    force_slots = set(rng.sample(range(preset.n_jobs), k=n_force)) if n_force > 0 else set()
    wide_submit_times = _planned_wide_submit_times(
        n_force, rng, burst_starts=tuple(preset.burst_starts), enabled=preset.align_wide_to_bursts
    )
    wide_idx = 0

    for i in range(preset.n_jobs):
        t += float(rng.choice(tuple(preset.arrival_gap_choices)))
        is_heavy = rng.random() < preset.heavy_fraction
        force_wide = i in force_slots

        if force_wide:
            is_heavy = True

        if is_heavy:
            width = int(rng.choice(tuple(preset.heavy_widths)))
            depth = int(rng.choice(tuple(preset.heavy_depths)))
            shots = int(rng.choice(tuple(preset.shots_heavy)))
            family = str(rng.choice(tuple(preset.families_heavy)))
        else:
            width = int(rng.choice(tuple(preset.light_widths)))
            depth = int(rng.choice(tuple(preset.light_depths)))
            shots = int(rng.choice(tuple(preset.shots_light)))
            family = str(rng.choice(tuple(preset.families_light)))

        qc = build_family_circuit(family, width, depth, seed + 997 * i)
        submit_t = float(wide_submit_times[wide_idx]) if force_wide and wide_idx < len(wide_submit_times) else float(t)
        if force_wide and wide_idx < len(wide_submit_times):
            wide_idx += 1

        jid = f"{family.upper()}_{'HEAVY' if is_heavy else 'LIGHT'}_{i:03d}"
        constraints = JobConstraints(
            allow_cutting=bool(method.allow_cutting),
            force_cutting=bool(method.force_cut_wide and force_wide),
            allow_multi_qpu=bool(method.allow_multi_qpu),
            max_cuts=3 if force_wide else 6,
            comm_overhead_s=0.02 if method.allow_multi_qpu else 0.0,
        )
        job = Job(
            job_id=jid,
            circuit=qc,
            task_type="expectation",
            observables=z_observable(width),
            shots=shots,
            submit_time_s=submit_t,
            constraints=constraints,
        )
        wl.append((submit_t, job))
        manifest_rows.append(
            {
                "job_id": jid,
                "family": family,
                "bucket": "heavy" if is_heavy else "light",
                "force_wide": bool(force_wide),
                "width": int(width),
                "depth": int(depth),
                "shots": int(shots),
                "submit_time_s": float(submit_t),
            }
        )

    wl.sort(key=lambda x: (float(x[0]), x[1].job_id))
    return wl, manifest_rows


def select_cut_strategy(name: str) -> CutStrategy:
    key = name.strip().lower()
    if key == "fitcut":
        return FitCutCutStrategy()
    if key == "addon":
        return QiskitAddonCutStrategy()
    if key == "naive":
        return NaiveChunkCutStrategy()
    raise ValueError(f"Unsupported cut strategy: {name}")


def active_reservations(qpus: Dict[str, QPUState], now_s: float) -> int:
    total = 0
    for st in qpus.values():
        try:
            active = getattr(st, "active_reservations", []) or []
            total += sum(1 for r in active if float(getattr(r, "end_s", 0.0)) > float(now_s))
        except Exception:
            continue
    return total


def export_results(sched: Scheduler, run_dir: Path, job_manifest: List[Dict[str, Any]], spec: RunSpec, sim_now_s: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    records = getattr(getattr(sched, "metrics", None), "records", []) or []
    events = getattr(sched, "task_log", []) or []

    with (run_dir / "records.csv").open("w", newline="") as f:
        fieldnames = [
            "job_id","qpu_id","plan_kind","submit_time_s",
            "t_schedule_s","t_partition_s","t_mapping_s","t_execution_s",
            "t_reconstruction_s","end_to_end_s","wall_end_to_end_s",
            "sim_queue_wait_s","sim_execution_span_s","sim_result_ready_time_s",
            "sim_first_task_start_s","sim_last_task_end_s","fidelity_proxy",
            "fidelity_estimated","details_json"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            det = getattr(r, "details", {}) or {}
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
                "end_to_end_s": det.get("sim_latency_s", getattr(r, "end_to_end_s", None)),
                "wall_end_to_end_s": det.get("wall_end_to_end_s", None),
                "sim_queue_wait_s": det.get("sim_queue_wait_s", det.get("queue_wait_s", None)),
                "sim_execution_span_s": det.get("sim_execution_span_s", None),
                "sim_result_ready_time_s": det.get("sim_result_ready_time_s", None),
                "sim_first_task_start_s": det.get("sim_first_task_start_s", None),
                "sim_last_task_end_s": det.get("sim_last_task_end_s", None),
                "fidelity_proxy": getattr(r, "fidelity_proxy", None),
                "fidelity_estimated": getattr(r, "fidelity_estimated", None),
                "details_json": json.dumps(det, default=str),
            })

    with (run_dir / "events.csv").open("w", newline="") as f:
        fieldnames = ["task_id","job_id","kind","start_s","end_s","qpu_id","qubits","label","depends_on","metadata_json"]
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

    with (run_dir / "job_manifest.csv").open("w", newline="") as f:
        fieldnames = ["job_id", "family", "bucket", "force_wide", "width", "depth", "shots", "submit_time_s"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(job_manifest)

    meta = {
        "run_name": spec.run_name,
        "suite": spec.suite,
        "seed": spec.seed,
        "full_eval": spec.full_eval,
        "method": asdict(spec.method),
        "workload": asdict(spec.workload),
        "sim_now_s": float(sim_now_s),
        "sim_completion_time_s": float(sim_now_s),
        "jobs_completed": len(records),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def run_one(spec: RunSpec, out_root: Path) -> Path:
    run_dir = out_root / spec.run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep planner fast and deterministic for experiment sweeps.
    env_defaults = {
        "QDC_CUT_FAST_PARTITION": "1",
        "QDC_PLANNER_APPROX_PARTITION": "1",
        "QDC_SKIP_PLAN_B_FOR_FORCED_WIDE": "1",
        "QDC_C_EXACT_PACK_MAX_LABELS": "8",
        "QDC_C_EXACT_PACK_MAX_STATES": "256",
        "QDC_C_MAX_SUBSETS": "2",
        "QDC_PLANNER_BUDGET_S": "0.10",
        "QDC_CUT_TIMEOUT_S": "0.03",
        "QDC_DEBUG_PLAN_B": "0",
        "QDC_DEBUG_PLAN_C": "0",
    }
    prev_env = {k: os.environ.get(k) for k in env_defaults}
    prev_env.update({k: os.environ.get(k) for k in spec.method.planner_env})
    try:
        for k, v in env_defaults.items():
            os.environ[k] = str(v)
        for k, v in spec.method.planner_env.items():
            os.environ[k] = str(v)

        qpus: Dict[str, QPUState] = {
            "qpu_A": make_line_qpu("qpu_A", 7, base_queue_delay_s=1.5),
            "qpu_B": make_line_qpu("qpu_B", 7, base_queue_delay_s=0.9),
            "qpu_C": make_line_qpu("qpu_C", 14, base_queue_delay_s=0.05),
        }
        inject_congestion_bursts(
            qpus,
            burst_starts=tuple(spec.workload.burst_starts),
            burst_dur=float(spec.workload.congestion_duration_s),
            targets=tuple(spec.workload.congestion_targets),
            block_fraction=float(spec.workload.congestion_block_fraction),
        )

        cfg = SchedulerConfig()
        strategy = select_cut_strategy(spec.method.cut_strategy)
        cfg.planner = PlannerConfig(cut_strategy=strategy)
        cfg.exec_cfg = ExecConfig(reserve_nonsim=True, timing_mode="analytic", cut_strategy=strategy)

        sched = Scheduler(qpus=qpus, quality=QualityModel(noise_models={}), cfg=cfg)
        workload, manifest_rows = make_workload(spec.workload, seed=spec.seed, method=spec.method)

        toggles = RunToggles(compute_estimated_fidelity=False, compute_expectation=bool(spec.full_eval), simulate_only=False)

        now = 0.0
        next_idx = 0
        idle_steps = 0
        stop_reason = None
        max_steps = 50000

        for step in range(max_steps):
            while next_idx < len(workload) and workload[next_idx][0] <= now:
                _t_submit, job = workload[next_idx]
                sched.submit_and_try_schedule(job, toggles)
                next_idx += 1

            t0 = time.perf_counter()
            dt = 1.0
            if next_idx < len(workload):
                dt = min(1.0, max(0.0, workload[next_idx][0] - now))
            if dt <= 0.0:
                dt = 0.1
            sched.step(dt)
            wall = time.perf_counter() - t0
            now += dt

            active_res = active_reservations(qpus, now)
            pending = sched.pending_count() if hasattr(sched, "pending_count") else len(getattr(sched, "_pending", []))
            if next_idx >= len(workload) and pending == 0 and active_res == 0:
                stop_reason = "all_jobs_completed_and_no_active_reservations"
                break

            if wall > 10.0:
                stop_reason = f"watchdog wall={wall:.2f}s at t={now:.2f}"
                break

            idle_steps = idle_steps + 1 if (next_idx >= len(workload) and pending == 0 and active_res == 0) else 0
            if idle_steps >= 5:
                stop_reason = "idle_break"
                break
        else:
            stop_reason = "max_steps reached"

        export_results(sched, run_dir, manifest_rows, spec, sim_now_s=now)
        (run_dir / "stdout_summary.json").write_text(json.dumps({
            "stop_reason": stop_reason,
            "submitted": next_idx,
            "completed": len(getattr(getattr(sched, "metrics", None), "records", []) or []),
            "pending": sched.pending_count() if hasattr(sched, "pending_count") else len(getattr(sched, "_pending", [])),
            "active_res": active_reservations(qpus, now),
        }, indent=2))
        return run_dir
    finally:
        for k, old in prev_env.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def expand_suite(suite_name: str, seeds: Sequence[int]) -> List[RunSpec]:
    if suite_name == "all":
        suite_names = list(SUITES.keys())
    else:
        suite_names = [suite_name]
    runs: List[RunSpec] = []
    for sname in suite_names:
        if sname not in SUITES:
            raise ValueError(f"Unknown suite: {sname}")
        sdef = SUITES[sname]
        for workload_name in sdef["workloads"]:
            wp = DEFAULT_WORKLOADS[workload_name]
            for method_name in sdef["methods"]:
                mp = DEFAULT_METHODS[method_name]
                if (not wp.fits_without_cutting) and mp.name == "no_cut_scheduler":
                    continue
                for seed in seeds:
                    runs.append(RunSpec(suite=sname, workload=wp, method=mp, seed=int(seed), full_eval=bool(sdef["full_eval"])))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Run scheduler experiment suites.")
    ap.add_argument("--suite", default="all", choices=["all", *SUITES.keys()])
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds.")
    ap.add_argument("--outdir", default="results/experiment_suite")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    runs = expand_suite(args.suite, seeds)
    manifest: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, spec in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {spec.run_name}")
        run_dir = run_one(spec, out_root)
        manifest.append({"run_name": spec.run_name, "run_dir": str(run_dir), "suite": spec.suite, "seed": spec.seed, "method": spec.method.name, "workload": spec.workload.name, "full_eval": spec.full_eval})
    (out_root / "suite_manifest.json").write_text(json.dumps({"runs": manifest, "elapsed_wall_s": time.perf_counter() - t0}, indent=2))
    print(f"Wrote {len(runs)} runs under {out_root}")


if __name__ == "__main__":
    main()