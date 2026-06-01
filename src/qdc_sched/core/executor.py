from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit_aer.primitives import SamplerV2, EstimatorV2
from qiskit_aer.noise import NoiseModel

from .types import Job, Plan, RunToggles, Task, TaskGraph
from .hardware import QPUState
from .quality import QualityModel
from .runtime import (
    predict_exec_time_s,
    estimate_qpu_execution_s,
    estimate_reconstruction_duration_s,
    estimate_communication_duration_s,
)
from .metrics import MetricsRecorder, JobRunRecord
from ..cutting import CutStrategy, FitCutCutStrategy, CutConstraints
from ..cutting.assignment import MinMakespanGreedyAssignment

def _timing_dict() -> dict:
    return {
        "transpile_s": 0.0,
        "execute_s": 0.0,
        "communication_s": 0.0,
        "reconstruct_s": 0.0,
        "postprocess_s": 0.0,
        "total_s": 0.0,
    }


def _attach_timing_dicts(details: dict, timing_model: dict, timing_wall: dict) -> dict:
    details = details or {}
    details["timing_model_s"] = timing_model
    details["timing_wall_s"] = timing_wall
    details.setdefault(
        "timing",
        {
            "transpile_s": float(timing_wall.get("transpile_s", 0.0)),
            "execute_s": float(timing_model.get("execute_s", 0.0)),
            "communication_s": float(timing_model.get("communication_s", 0.0)),
            "reconstruct_s": float(timing_model.get("reconstruct_s", 0.0)),
            "postprocess_s": float(timing_wall.get("postprocess_s", 0.0)),
            "total_measured_s": float(timing_wall.get("total_s", 0.0)),
        },
    )
    return details


def _effective_qpu_timing_mode(cfg) -> str:
    return str(os.getenv("QDC_QPU_TIMING_MODE", getattr(cfg, "timing_mode", "analytic")) or "analytic").lower()


class _CPUWorkerPool:
    def __init__(self, n_workers: int):
        import heapq
        self._heapq = heapq
        self.n_workers = max(1, int(n_workers or 1))
        self.available = [(0.0, i) for i in range(self.n_workers)]
        self._heapq.heapify(self.available)

    def reserve(self, ready_time: float, duration_s: float):
        avail_t, wid = self._heapq.heappop(self.available)
        start_t = max(float(ready_time), float(avail_t))
        end_t = start_t + max(0.0, float(duration_s))
        self._heapq.heappush(self.available, (end_t, wid))
        queue_delay = max(0.0, start_t - float(ready_time))
        return int(wid), float(start_t), float(end_t), float(queue_delay)


class _CommWorkerPool:
    def __init__(self, n_workers: int):
        import heapq
        self._heapq = heapq
        self.n_workers = max(1, int(n_workers or 1))
        self.available = [(0.0, i) for i in range(self.n_workers)]
        self._heapq.heapify(self.available)

    def reserve(self, ready_time: float, duration_s: float):
        avail_t, wid = self._heapq.heappop(self.available)
        start_t = max(float(ready_time), float(avail_t))
        end_t = start_t + max(0.0, float(duration_s))
        self._heapq.heappush(self.available, (end_t, wid))
        queue_delay = max(0.0, start_t - float(ready_time))
        return int(wid), float(start_t), float(end_t), float(queue_delay)


def _safe_noise_model(obj) -> Optional[NoiseModel]:
    try:
        if isinstance(obj, NoiseModel):
            return obj
    except Exception:
        pass
    return None


def _max_end_s(tasks: List[Task], default: float = 0.0) -> float:
    mx = float(default)
    for t in tasks or []:
        try:
            mx = max(mx, float(getattr(t, "end_s", mx)))
        except Exception:
            continue
    return mx


def _ensure_taskgraph(details: dict, job_id: str, tasks: List[Task]) -> dict:
    details = details or {}
    details["tasks"] = list(tasks)
    details["task_graph"] = TaskGraph(job_id=job_id, tasks=list(tasks))
    return details


def _append_comm_and_recon_tasks(
    tasks: List[Task],
    *,
    job_id: str,
    plan_kind: str,
    comm_overhead_s: float = 0.0,
    recon_s: float = 0.0,
    comm_dep_task_ids: Optional[List[str]] = None,
    recon_dep_task_ids: Optional[List[str]] = None,
) -> List[Task]:
    """
    COMM0 depends on all quantum tasks unless comm_dep_task_ids is provided.
    RECON0 depends on COMM0 if present else all quantum tasks unless recon_dep_task_ids is provided.
    """
    out = list(tasks or [])
    q_ids = [t.task_id for t in out if getattr(t, "kind", None) == "quantum" and getattr(t, "task_id", None)]
    cursor = _max_end_s(out, default=0.0)

    comm_id = None
    comm_dur = float(comm_overhead_s or 0.0)
    if comm_dur > 0.0:
        comm_id = f"{job_id}:COMM0"
        out.append(
            Task(
                task_id=comm_id,
                job_id=job_id,
                kind="communication",
                start_s=float(cursor),
                end_s=float(cursor + comm_dur),
                qpu_id=None,
                qubits=None,
                label=None,
                depends_on=list(comm_dep_task_ids) if comm_dep_task_ids is not None else list(q_ids) if q_ids else None,
                metadata={"plan": plan_kind, "comm_overhead_s": comm_dur},
            )
        )
        cursor += comm_dur

    rdur = float(recon_s or 0.0)
    if rdur > 0.0:
        dep = None
        if recon_dep_task_ids is not None:
            dep = list(recon_dep_task_ids)
        else:
            dep = [comm_id] if comm_id is not None else list(q_ids) if q_ids else None
        out.append(
            Task(
                task_id=f"{job_id}:RECON0",
                job_id=job_id,
                kind="reconstruction",
                start_s=float(cursor),
                end_s=float(cursor + rdur),
                qpu_id=None,
                qubits=None,
                label=None,
                depends_on=dep,
                metadata={"plan": plan_kind, "recon_s": rdur},
            )
        )
    return out


def _pad_observable_to_num_qubits(obs, n: int):
    """Pad observable with identities if circuit got expanded."""
    if obs is None:
        return None
    if isinstance(obs, SparsePauliOp):
        if obs.num_qubits == n:
            return obs
        if obs.num_qubits > n:
            raise ValueError(f"Observable has {obs.num_qubits} qubits but circuit has {n}.")
        k = n - obs.num_qubits
        padded = [("I" * k + p, complex(c)) for p, c in zip(obs.paulis.to_labels(), obs.coeffs)]
        return SparsePauliOp.from_list(padded)
    if isinstance(obs, PauliList):
        if obs.num_qubits == n:
            return obs
        if obs.num_qubits > n:
            raise ValueError(f"Observable has {obs.num_qubits} qubits but circuit has {n}.")
        k = n - obs.num_qubits
        return PauliList([("I" * k + p) for p in obs.to_labels()])
    if isinstance(obs, str):
        if len(obs) == n:
            return obs
        if len(obs) > n:
            raise ValueError(f"Observable string has {len(obs)} qubits but circuit has {n}.")
        return "I" * (n - len(obs)) + obs
    return obs


def _to_paulilist(obs):
    if obs is None:
        raise ValueError("Expectation task requires observables.")
    if isinstance(obs, PauliList):
        return obs
    if isinstance(obs, SparsePauliOp):
        return PauliList(obs.paulis)
    if isinstance(obs, str):
        return PauliList([obs])
    raise TypeError(f"Unsupported observable type: {type(obs)}")


# ----------------------------
# Config
# ----------------------------

@dataclass
class ExecConfig:
    optimization_level: int = 1
    shots_default: int = 1000
    reserve_nonsim: bool = False

    # Timing: "analytic" uses predict_exec_time_s; "aer" measures wall-clock with Aer primitives.
    timing_mode: str = "analytic"
    aer_timing_repeats: int = 1
    aer_timing_use_noise: bool = False
    aer_timing_include_transpile: bool = False

    # Cutting
    cut_constraints: CutConstraints = field(default_factory=CutConstraints)
    cut_strategy: CutStrategy = field(default_factory=FitCutCutStrategy)
    partition_assignment_policy: Any = field(default_factory=MinMakespanGreedyAssignment)

class Executor:
    def __init__(self, qpus: Dict[str, QPUState], quality: QualityModel, metrics: MetricsRecorder, cfg: ExecConfig):
        self.qpus = qpus
        self.quality = quality
        self.metrics = metrics
        self.cfg = cfg
        self._aer_timing_cache: Dict[tuple, float] = {}
        self.cpu_recon_pool = _CPUWorkerPool(int(os.getenv("QDC_CPU_RECON_WORKERS", "1") or 1))
        self.comm_pool = _CommWorkerPool(int(os.getenv("QDC_COMM_WORKERS", "1") or 1))

    def _hash_circuit(self, circ: QuantumCircuit) -> str:
        try:
            s = circ.qasm()
        except Exception:
            s = repr(circ)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _strip_cut_instructions(circ: QuantumCircuit) -> QuantumCircuit:
        """Return a copy of circ with qiskit-addon-cutting boundary ops removed.

        Wire-cut subcircuits contain 'In' and 'Out' channel instructions that
        Aer cannot simulate.  For timing purposes we only need the gate structure,
        so we drop any instruction whose name is not in a known-Aer-safe set.
        """
        _AER_SAFE = frozenset({
            "h", "x", "y", "z", "s", "sdg", "t", "tdg",
            "sx", "sxdg", "rx", "ry", "rz", "u", "u1", "u2", "u3",
            "cx", "cy", "cz", "ch", "swap", "iswap", "ecr",
            "ccx", "cswap", "reset", "barrier", "measure", "delay",
        })
        out = QuantumCircuit(circ.num_qubits)
        for inst in circ.data:
            if inst.operation.name.lower() in _AER_SAFE:
                out.append(inst.operation, inst.qubits)
        return out

    def _measure_exec_time_s_aer(
        self,
        circ: QuantumCircuit,
        shots: int,
        task_type: str,
        observables: Optional[Any] = None,
        use_noise: bool = False,
        repeats: int = 1,
    ) -> float:
        reps = max(1, int(repeats or 1))
        if str(task_type) == "expectation":
            prim = EstimatorV2()
        else:
            prim = SamplerV2()

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            if str(task_type) == "expectation":
                job = prim.run([(circ, observables)], shots=shots)
                _ = job.result()
            else:
                job = prim.run([circ], shots=shots)
                _ = job.result()
            times.append(time.perf_counter() - t0)
        times.sort()
        return float(times[len(times) // 2])

    def _estimate_exec_duration_s(
        self,
        qid: str,
        circ: Optional[QuantumCircuit],
        shots: int,
        task_type: str,
        observables: Optional[Any] = None,
    ) -> float:
        if circ is None:
            return 0.0
        eff_shots = int(shots or 0) or int(getattr(self.cfg, "shots_default", 1000) or 1000)

        mode = _effective_qpu_timing_mode(self.cfg)
        if mode != "aer":
            from qdc_sched.core.profiler import profile_circuit
            sp = profile_circuit(circ)
            dur, _meta = estimate_qpu_execution_s(
                self.qpus[qid].profile,
                circuit=circ,
                prof=sp,
                shots=eff_shots,
            )
            return float(dur)

        # Strip cut-boundary instructions (In/Out) that Aer cannot simulate.
        # For timing we only need the gate structure; the cut scaffolding is irrelevant.
        aer_circ = self._strip_cut_instructions(circ)
        key = (qid, self._hash_circuit(aer_circ), eff_shots, str(task_type), bool(getattr(self.cfg, "aer_timing_use_noise", False)))
        if key in self._aer_timing_cache:
            return float(self._aer_timing_cache[key])

        try:
            dur = self._measure_exec_time_s_aer(
                circ=aer_circ,
                shots=eff_shots,
                task_type=str(task_type),
                observables=observables,
                use_noise=bool(getattr(self.cfg, "aer_timing_use_noise", False)),
                repeats=int(getattr(self.cfg, "aer_timing_repeats", 1) or 1),
            )
        except Exception as _aer_exc:
            import os as _os, traceback as _tb
            if _os.getenv("QDC_AER_DEBUG", "0") == "1":
                print(f"[AER_FALLBACK] qid={qid} circ={getattr(aer_circ,'name','?')} "
                      f"n_qubits={getattr(aer_circ,'num_qubits','?')} n_clbits={getattr(aer_circ,'num_clbits','?')} "
                      f"err={_aer_exc!r}")
                _tb.print_exc()
            from qdc_sched.core.profiler import profile_circuit
            sp = profile_circuit(aer_circ)
            dur, _meta = estimate_qpu_execution_s(
                self.qpus[qid].profile,
                circuit=aer_circ,
                prof=sp,
                shots=eff_shots,
            )

        self._aer_timing_cache[key] = float(dur)
        return float(dur)

    def build_task_graph(self, job: Job, plan: Plan, now_s: float) -> Tuple[List[Task], Dict[str, Any]]:
        """Build quantum tasks for the plan. Does not reserve resources."""
        tasks: List[Task] = []
        aux: Dict[str, Any] = {"task_reservations": []}

        if plan.kind == "A_NO_CUT_SINGLE":
            qid = str(plan.qpu_id)
            qpu = self.qpus[qid]
            det = plan.details or {}
            qdelay = float(det.get("pred_queue_delay_s", 0.0) or 0.0)
            exec_s = float(det.get("pred_exec_time_s", 0.0) or 0.0)
            start_s = float(now_s) + qdelay
            dur_s = max(0.0, exec_s)
            qubits = sorted(list(plan.physical_qubits or []))
            tasks.append(Task(
                task_id=f"{job.job_id}:A0",
                job_id=job.job_id,
                kind="quantum",
                qpu_id=qid,
                qubits=qubits,
                start_s=float(start_s),
                end_s=float(start_s + dur_s),
                label=None,
                depends_on=None,
                metadata={"plan": plan.kind, "k": len(qubits), "timing_mode": _effective_qpu_timing_mode(self.cfg)},
            ))
            return tasks, aux

        if plan.kind == "B_CUT_SINGLE_SEQ":
            # Partition to learn subcircuits and label widths. Execution uses sequential schedule on chosen qpu.
            qid = str(plan.qpu_id)
            qpu = self.qpus[qid]
            base_cc = self.cfg.cut_constraints
            cc = CutConstraints(
                max_cuts=min(base_cc.max_cuts, job.constraints.max_cuts),
                allow_gate_cuts=base_cc.allow_gate_cuts,
                allow_wire_cuts=base_cc.allow_wire_cuts,
                reconstruction_target=base_cc.reconstruction_target,
                target_labels=None,
                target_labels_cap=int(getattr(base_cc, "target_labels_cap", 4) or 4),
                seed_tries=int(getattr(base_cc, "seed_tries", 4) or 4),
            )
            max_local = None
            try:
                max_local = (plan.details or {}).get("max_local_qubits", None)
            except Exception:
                max_local = None
            if max_local is None:
                max_local = qpu.max_connected_free_qubits(now_s)

            det = plan.details or {}
            preferred_subs = det.get("partition_subcircuits", None)
            if preferred_subs:
                class _StoredPartition:
                    pass
                part = _StoredPartition()
                part.subcircuits = list(preferred_subs)
                part.k_wire = det.get("k_wire", None)
                part.k_gate = det.get("k_gate", None)
                part.est_executions = det.get("sampling_overhead", None)
                part.reconstruction = None
            else:
                part = self.cfg.cut_strategy.partition(job.circuit, cc, {"max_local_qubits": max_local, "observable": job.observables, "qpu_candidates": [qid]})
            aux["partition"] = part
            aux["cut_metadata"] = {"k_wire": getattr(part, "k_wire", None), "k_gate": getattr(part, "k_gate", None), "est_executions": getattr(part, "est_executions", None)}
            qdelay = float(det.get("pred_queue_delay_s", float(getattr(qpu.profile, "base_queue_delay_s", 0.0)) or 0.0))
            cursor = float(now_s) + qdelay

            from qdc_sched.core.profiler import profile_circuit
            for i, sc in enumerate(getattr(part, "subcircuits", []) or []):
                sp = profile_circuit(sc)
                k = max(1, int(getattr(sp, "width", 1) or 1))
                dur = float(self._estimate_exec_duration_s(qid, sc, int(job.shots or 0) or self.cfg.shots_default, "sampling"))
                # Reserve-time qubits chosen later in apply_reservations (to reflect real free sets)
                tasks.append(Task(
                    task_id=f"{job.job_id}:B{i}",
                    job_id=job.job_id,
                    kind="quantum",
                    qpu_id=qid,
                    qubits=None,
                    start_s=float(cursor),
                    end_s=float(cursor + dur),
                    label=int(i),
                    depends_on=None,
                    metadata={"plan": plan.kind, "k": k, "subcircuit_index": i, "timing_mode": _effective_qpu_timing_mode(self.cfg)},
                ))
                cursor += dur
            return tasks, aux

        if plan.kind == "C_CUT_MULTI_QPU":
            qpu_ids = []
            try:
                qpu_ids = (plan.details or {}).get("qpu_candidates", []) or []
            except Exception:
                qpu_ids = []
            if not qpu_ids:
                qpu_ids = list(self.qpus.keys())

            base_cc = self.cfg.cut_constraints
            cc = CutConstraints(
                max_cuts=min(base_cc.max_cuts, job.constraints.max_cuts),
                allow_gate_cuts=base_cc.allow_gate_cuts,
                allow_wire_cuts=base_cc.allow_wire_cuts,
                reconstruction_target=base_cc.reconstruction_target,
                target_labels=getattr(base_cc, "target_labels", None),
                target_labels_cap=int(getattr(base_cc, "target_labels_cap", 4) or 4),
                seed_tries=int(getattr(base_cc, "seed_tries", 4) or 4),
            )

            max_local = min(int(self.qpus[qid].max_connected_free_qubits(now_s)) for qid in qpu_ids) if qpu_ids else 0
            if max_local <= 1:
                raise RuntimeError("No capacity for multi-QPU cutting (max_local<=1).")

            # Make sure we have enough labels to distribute
            import math
            max_local_safe = max(2, int(max_local))
            needed_labels = max(2, int(math.ceil(float(job.circuit.num_qubits) / float(max_local_safe))))
            cc.target_labels = int(max(needed_labels, int(getattr(cc, "target_labels", needed_labels) or needed_labels)))

            det = plan.details or {}
            preferred_subs = det.get("partition_subcircuits", None)
            if preferred_subs:
                class _StoredPartition:
                    pass
                part = _StoredPartition()
                part.subcircuits = list(preferred_subs)
                part.k_wire = det.get("k_wire", None)
                part.k_gate = det.get("k_gate", None)
                part.est_executions = det.get("sampling_overhead", None)
                part.reconstruction = None
            else:
                part = self.cfg.cut_strategy.partition(job.circuit, cc, {"max_local_qubits": max_local_safe, "observable": job.observables, "qpu_candidates": qpu_ids})
            aux["partition"] = part

            subs = list(getattr(part, "subcircuits", []) or [])
            if len(subs) < needed_labels:
                raise RuntimeError(f"Plan C partition produced {len(subs)} labels, need >= {needed_labels}.")

            labels = list(range(len(subs)))
            aux["labels"] = labels

            # Assignment (policy) based on predicted per-label times
            from qdc_sched.core.profiler import profile_circuit
            label_qpu_pred: Dict[int, Dict[str, float]] = {i: {} for i in labels}
            label_k: Dict[int, int] = {}
            for i, sc in enumerate(subs):
                sp = profile_circuit(sc)
                k = max(1, int(getattr(sp, "width", 1) or 1))
                label_k[i] = k
                for qid in qpu_ids:
                    label_qpu_pred[i][qid] = float(self._estimate_exec_duration_s(qid, sc, int(job.shots or 0) or self.cfg.shots_default, "sampling"))

            aux["label_k"] = label_k
            aux["label_qpu_pred_time"] = label_qpu_pred

            # If the planner provided an explicit assignment, honor it to keep scoring consistent with execution
            preferred_assignment = None
            try:
                preferred_assignment = (plan.details or {}).get("assignment", None)
            except Exception:
                preferred_assignment = None

            if preferred_assignment is not None:
                try:
                    assignment = {int(k): v for k, v in dict(preferred_assignment).items()}
                except Exception:
                    assignment = preferred_assignment
            else:
                policy = getattr(self.cfg, "partition_assignment_policy", None)
                if policy is None:
                    assignment = {i: qpu_ids[i % len(qpu_ids)] for i in labels}
                else:
                    label_costs = {i: float(label_k[i]) for i in labels}
                    qpu_caps = {qid: int(self.qpus[qid].max_connected_free_qubits(now_s)) for qid in qpu_ids}
                    qpu_quality = {qid: float(self.quality.fidelity_proxy_from_profile(qid, profile_circuit(job.circuit))) for qid in qpu_ids}
                    qpu_pred_time = {qid: float(getattr(self.qpus[qid].profile, "base_queue_delay_s", 0.0) or 0.0) for qid in qpu_ids}
                    assignment = policy.assign(
                        labels,
                        qpu_ids,
                        label_costs=label_costs,
                        
                        qpu_caps=qpu_caps,
                        qpu_quality=qpu_quality,
                        qpu_pred_time=qpu_pred_time,
                        label_qpu_pred_time=label_qpu_pred,
                    )
            aux["assignment"] = assignment
            per_qpu_wait_s = {}
            try:
                per_qpu_wait_s = dict((plan.details or {}).get("per_qpu_wait_s", {}) or {})
            except Exception:
                per_qpu_wait_s = {}
            cursors = {qid: float(now_s) + float(per_qpu_wait_s.get(qid, per_qpu_wait_s.get(str(qid), 0.0)) or 0.0) for qid in qpu_ids}

            for i, sc in enumerate(subs):
                qid = str(assignment[i])
                dur = float(label_qpu_pred[i][qid])
                tasks.append(Task(
                    task_id=f"{job.job_id}:task{i}",
                    job_id=job.job_id,
                    kind="quantum",
                    qpu_id=qid,
                    qubits=None,
                    start_s=float(cursors[qid]),
                    end_s=float(cursors[qid] + dur),
                    label=int(i),
                    depends_on=None,
                    metadata={"plan": plan.kind, "k": label_k[i], "subcircuit_index": i, "timing_mode": _effective_qpu_timing_mode(self.cfg)},
                ))
                cursors[qid] += dur
            return tasks, aux

        return tasks, aux

    def apply_reservations(self, tasks: List[Task], *, now_s: float, reserve: bool) -> Dict[str, Any]:
        """Assign connected qubit sets (if missing) and reserve them when requested."""
        out: Dict[str, Any] = {"task_reservations": [], "reservations_applied": False, "reserve_nonsim_error": None}
        if not reserve:
            return out
        try:
            for t in tasks:
                if getattr(t, "kind", None) != "quantum":
                    continue
                qid = getattr(t, "qpu_id", None)
                if qid is None:
                    continue
                qpu = self.qpus[str(qid)]
                k = max(1, int((t.metadata or {}).get("k", 1) or 1))
                start_s = float(getattr(t, "start_s", now_s))
                # find and reserve connected subgraph
                qset = qpu.find_free_connected_subgraph(k, start_s)
                if qset is None:
                    raise RuntimeError(f"reserve_nonsim: no free connected subgraph size={k} on {qid} at t={start_s}")
                dur_s = max(1e-6, float(getattr(t, "end_s", start_s) - start_s))
                qpu.reserve(t.task_id, set(qset), start_s=start_s, duration_s=dur_s)
                t.qubits = sorted(list(qset))
                out["task_reservations"].append({"qpu": str(qid), "task_id": t.task_id, "k": k, "start_s": start_s, "end_s": start_s + dur_s, "qubits": sorted(list(qset))})
            out["reservations_applied"] = True
        except Exception as e:
            out["reserve_nonsim_error"] = repr(e)
        return out

    def maybe_execute(self, job: Job, plan: Plan, aux: Dict[str, Any], toggles: RunToggles) -> Tuple[Any, Dict[str, Any]]:
        """Run Aer primitives if requested; otherwise return None while still charging model time."""
        out: Any = None
        extra: Dict[str, Any] = {}

        if plan.kind == "A_NO_CUT_SINGLE":
            return None, extra

        # Fast path: if compute_expectation is False, don't execute heavy workloads or rebuild reconstruction data.
        if not bool(getattr(toggles, "compute_expectation", True)):
            extra["execution_mode"] = "timing_only"
            return None, extra

        # Plan B/C: expectation jobs support reconstruction
        part = aux.get("partition", None)
        if part is None:
            return None, extra

        # If the stored partition came from planner details only, rebuild once here with observable metadata.
        if not isinstance(getattr(part, "reconstruction", None), dict) or "subexperiments" not in getattr(part, "reconstruction", {}):
            try:
                if plan.kind == "B_CUT_SINGLE_SEQ":
                    qid = str(plan.qpu_id)
                    max_local = (plan.details or {}).get("max_local_qubits", None)
                    if max_local is None:
                        max_local = self.qpus[qid].max_connected_free_qubits(0.0)
                    cc = CutConstraints(
                        max_cuts=min(self.cfg.cut_constraints.max_cuts, job.constraints.max_cuts),
                        allow_gate_cuts=self.cfg.cut_constraints.allow_gate_cuts,
                        allow_wire_cuts=self.cfg.cut_constraints.allow_wire_cuts,
                        reconstruction_target=self.cfg.cut_constraints.reconstruction_target,
                        target_labels=(plan.details or {}).get("target_labels", None),
                        target_labels_cap=int(getattr(self.cfg.cut_constraints, "target_labels_cap", 4) or 4),
                        seed_tries=int(getattr(self.cfg.cut_constraints, "seed_tries", 4) or 4),
                    )
                    part = self.cfg.cut_strategy.partition(job.circuit, cc, {"max_local_qubits": max_local, "observable": job.observables, "qpu_candidates": [qid]})
                elif plan.kind == "C_CUT_MULTI_QPU":
                    qpu_ids = (plan.details or {}).get("qpu_candidates", []) or list(self.qpus.keys())
                    max_local = min(int(self.qpus[qid].max_connected_free_qubits(0.0)) for qid in qpu_ids) if qpu_ids else 0
                    cc = CutConstraints(
                        max_cuts=min(self.cfg.cut_constraints.max_cuts, job.constraints.max_cuts),
                        allow_gate_cuts=self.cfg.cut_constraints.allow_gate_cuts,
                        allow_wire_cuts=self.cfg.cut_constraints.allow_wire_cuts,
                        reconstruction_target=self.cfg.cut_constraints.reconstruction_target,
                        target_labels=(plan.details or {}).get("target_labels", None),
                        target_labels_cap=int(getattr(self.cfg.cut_constraints, "target_labels_cap", 4) or 4),
                        seed_tries=int(getattr(self.cfg.cut_constraints, "seed_tries", 4) or 4),
                    )
                    part = self.cfg.cut_strategy.partition(job.circuit, cc, {"max_local_qubits": max(2, int(max_local)), "observable": job.observables, "qpu_candidates": qpu_ids})
                aux["partition"] = part
            except Exception:
                extra["cut_error"] = getattr(part, "reconstruction", None)
                return None, extra

        if not isinstance(getattr(part, "reconstruction", None), dict) or "subexperiments" not in part.reconstruction:
            extra["cut_error"] = getattr(part, "reconstruction", None)
            return None, extra

        subexperiments = part.reconstruction["subexperiments"]
        coefficients = part.reconstruction["coefficients"]
        subobservables = part.reconstruction["subobservables"]

        # Execute subexperiments with SamplerV2 (optionally per-qpu noise in C)
        t_exec0 = time.perf_counter()
        results_by_label = {}

        if plan.kind == "B_CUT_SINGLE_SEQ":
            qid = str(plan.qpu_id)
            nm = _safe_noise_model(getattr(self.quality, "noise_models", {}).get(qid))
            sampler = SamplerV2(options={"backend_options": {"noise_model": nm}}) if nm is not None else SamplerV2()
            for lab, circs in subexperiments.items():
                pubs = [(c, None, int(job.shots or 0) or self.cfg.shots_default) for c in circs]
                results_by_label[lab] = sampler.run(pubs).result()
        else:
            assignment = aux.get("assignment", {}) or {}
            per_label_wall_s: Dict[int, float] = {}
            per_label_qpu: Dict[int, str] = {}

            def _run_label(lab, circs):
                qid = str(assignment.get(int(lab), assignment.get(lab)))
                nm = _safe_noise_model(getattr(self.quality, "noise_models", {}).get(qid))
                sampler = SamplerV2(options={"backend_options": {"noise_model": nm}}) if nm is not None else SamplerV2()
                pubs = [(c, None, int(job.shots or 0) or self.cfg.shots_default) for c in circs]
                t_lab0 = time.perf_counter()
                res = sampler.run(pubs).result()
                t_lab1 = time.perf_counter()
                return lab, qid, res, float(t_lab1 - t_lab0)

            used_qpus = sorted({str(assignment.get(int(lab), assignment.get(lab))) for lab in subexperiments.keys()})
            max_workers = max(1, min(len(subexperiments), len(used_qpus)))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = [pool.submit(_run_label, lab, circs) for lab, circs in subexperiments.items()]
                for fut in as_completed(futs):
                    lab, qid, res, wall_s = fut.result()
                    results_by_label[lab] = res
                    per_label_wall_s[int(lab)] = float(wall_s)
                    per_label_qpu[int(lab)] = str(qid)

            extra["c_parallel_workers"] = int(max_workers)
            extra["c_parallel_used_qpus"] = list(used_qpus)
            extra["c_parallel_label_wall_s"] = dict(sorted(per_label_wall_s.items()))
            extra["c_parallel_label_qpu"] = dict(sorted(per_label_qpu.items()))

        t_exec1 = time.perf_counter()

        # Reconstruction
        cutting = __import__("qiskit_addon_cutting")
        reconstruct = getattr(cutting, "reconstruct_expectation_values")
        t_rec0 = time.perf_counter()
        evs = reconstruct(results_by_label, coefficients, subobservables)
        t_rec1 = time.perf_counter()

        extra["wall_exec_s"] = float(t_exec1 - t_exec0)
        extra["wall_recon_s"] = float(t_rec1 - t_rec0)
        extra["wall_total_s"] = float((t_exec1 - t_exec0) + (t_rec1 - t_rec0))

        if isinstance(evs, dict) and len(evs) > 0:
            out = evs[list(evs.keys())[0]]
        elif isinstance(evs, list) and len(evs) > 0:
            out = evs[0]
        else:
            out = evs
        return out, extra

    def assemble_record(
        self,
        job: Job,
        plan: Plan,
        *,
        now_s: float,
        t0_wall: float,
        t1_wall: float,
        tasks: List[Task],
        aux: Dict[str, Any],
        reserve_info: Dict[str, Any],
        exec_extra: Dict[str, Any],
    ) -> JobRunRecord:
        details: Dict[str, Any] = dict(plan.details or {})
        details.update(aux or {})
        details.update(reserve_info or {})
        details.update(exec_extra or {})

        # Model-facing execution / communication / reconstruction times
        t_exec_model = 0.0
        t_comm_model = 0.0
        t_recon_model = 0.0
        q_only = [t for t in tasks if getattr(t, "kind", None) == "quantum"]

        if plan.kind == "A_NO_CUT_SINGLE":
            t_exec_model = float((plan.details or {}).get("pred_exec_time_s", 0.0) or 0.0)
            comm_meta = {}
        elif plan.kind in ("B_CUT_SINGLE_SEQ", "C_CUT_MULTI_QPU"):
            if q_only:
                t_exec_model = float(
                    max(float(t.end_s) for t in q_only) - min(float(t.start_s) for t in q_only)
                )
            t_recon_model = float(details.get("recon_est_s", 0.0) or details.get("wall_recon_s", 0.0) or 0.0)

            num_subexperiments = float(
                details.get("labels_used", 0.0)
                or details.get("sampling_overhead", 0.0)
                or 0.0
            )
            num_samples = float(details.get("sampling_overhead", 0.0) or 0.0)

            if plan.kind == "C_CUT_MULTI_QPU":
                used_qpus = details.get("used_qpus", []) or []
                if not used_qpus:
                    assignment = details.get("assignment", {}) or {}
                    try:
                        used_qpus = sorted({str(v) for v in assignment.values()})
                    except Exception:
                        used_qpus = []
                if not used_qpus:
                    used_qpus = details.get("qpu_candidates", []) or []
                n_qpus_used = max(1, len(used_qpus))
            else:
                n_qpus_used = 1

            t_comm_model, comm_meta = estimate_communication_duration_s(
                num_subexperiments=float(num_subexperiments),
                num_samples=float(num_samples),
                n_qpus_used=int(max(1, n_qpus_used)),
                plan_kind=str(plan.kind),
            )
            details["comm_timing_meta"] = dict(comm_meta or {})
        else:
            comm_meta = {}

        # Append explicit COMM and CPU-hosted reconstruction tasks using charged values
        tasks2 = list(tasks or [])
        q_ids = [t.task_id for t in tasks2 if getattr(t, "kind", None) == "quantum" and getattr(t, "task_id", None)]
        cursor = _max_end_s(tasks2, default=0.0)

        comm_queue_delay_s = 0.0
        comm_busy_time_s = 0.0
        sim_comm_queue_s = 0.0
        sim_comm_service_s = 0.0
        comm_id = None

        if float(t_comm_model) > 0.0:
            ready_time = float(cursor)
            comm_worker_id, comm_start, comm_end, qdelay = self.comm_pool.reserve(
                ready_time,
                float(t_comm_model),
            )
            comm_id = f"{job.job_id}:COMM0"
            tasks2.append(
                Task(
                    task_id=comm_id,
                    job_id=job.job_id,
                    kind="communication",
                    start_s=float(comm_start),
                    end_s=float(comm_end),
                    qpu_id=f"host_comm:{comm_worker_id}",
                    qubits=None,
                    label=None,
                    depends_on=list(q_ids) if q_ids else None,
                    metadata={
                        "plan": plan.kind,
                        "comm_s": float(t_comm_model),
                        "comm_worker_id": int(comm_worker_id),
                        "comm_timing_meta": dict(comm_meta or {}),
                    },
                )
            )
            comm_queue_delay_s = float(qdelay)
            comm_busy_time_s = float(comm_end - comm_start)
            sim_comm_queue_s = float(qdelay)
            sim_comm_service_s = float(comm_end - comm_start)
            cursor = float(comm_end)

        cpu_queue_delay_s = 0.0
        cpu_busy_time_s = 0.0
        sim_recon_queue_s = 0.0
        sim_recon_service_s = 0.0
        if float(t_recon_model) > 0.0:
            ready_time = float(cursor)
            worker_id, recon_start, recon_end, qdelay = self.cpu_recon_pool.reserve(ready_time, float(t_recon_model))
            dep = [comm_id] if comm_id is not None else list(q_ids) if q_ids else None
            tasks2.append(
                Task(
                    task_id=f"{job.job_id}:RECON0",
                    job_id=job.job_id,
                    kind="reconstruction",
                    start_s=float(recon_start),
                    end_s=float(recon_end),
                    qpu_id=f"host_recon:{worker_id}",
                    qubits=None,
                    label=None,
                    depends_on=dep,
                    metadata={"plan": plan.kind, "recon_s": float(t_recon_model), "cpu_worker_id": int(worker_id)},
                )
            )
            cpu_queue_delay_s = float(qdelay)
            cpu_busy_time_s = float(recon_end - recon_start)
            sim_recon_queue_s = float(qdelay)
            sim_recon_service_s = float(recon_end - recon_start)

        details["comm_queue_delay_s"] = float(comm_queue_delay_s)
        details["comm_busy_time_s"] = float(comm_busy_time_s)
        details["sim_comm_queue_s"] = float(sim_comm_queue_s)
        details["sim_comm_service_s"] = float(sim_comm_service_s)
        details["cpu_queue_delay_s"] = float(cpu_queue_delay_s)
        details["cpu_busy_time_s"] = float(cpu_busy_time_s)
        details["sim_recon_queue_s"] = float(sim_recon_queue_s)
        details["sim_recon_service_s"] = float(sim_recon_service_s)

        details = _ensure_taskgraph(details, job.job_id, tasks2)

        # Simulation-time timing fields derived from the task graph.
        all_tasks = list(tasks2 or [])
        q_tasks = [t for t in all_tasks if getattr(t, "kind", None) == "quantum"]
        comm_tasks = [t for t in all_tasks if getattr(t, "kind", None) == "communication"]
        recon_tasks = [t for t in all_tasks if getattr(t, "kind", None) == "reconstruction"]
        sim_first_task_start_s = float(min((float(getattr(t, "start_s", now_s)) for t in all_tasks), default=float(now_s)))
        sim_last_task_end_s = float(max((float(getattr(t, "end_s", now_s)) for t in all_tasks), default=float(now_s)))
        sim_first_quantum_start_s = float(min((float(getattr(t, "start_s", now_s)) for t in q_tasks), default=sim_first_task_start_s))
        sim_last_quantum_end_s = float(max((float(getattr(t, "end_s", now_s)) for t in q_tasks), default=sim_last_task_end_s))
        sim_queue_wait_s = max(0.0, sim_first_task_start_s - float(getattr(job, "submit_time_s", 0.0) or 0.0))
        sim_execution_span_s = max(0.0, sim_last_quantum_end_s - sim_first_quantum_start_s) if q_tasks else 0.0
        sim_result_ready_time_s = sim_last_task_end_s
        sim_latency_s = max(0.0, sim_result_ready_time_s - float(getattr(job, "submit_time_s", 0.0) or 0.0))
        sim_comm_s = sum(max(0.0, float(t.end_s) - float(t.start_s)) for t in comm_tasks)
        sim_recon_s = sum(max(0.0, float(t.end_s) - float(t.start_s)) for t in recon_tasks)
        sim_submit_to_quantum_done_s = (
            max(0.0, sim_last_quantum_end_s - float(getattr(job, "submit_time_s", 0.0) or 0.0))
            if q_tasks else 0.0
        )

        details["sim_first_task_start_s"] = sim_first_task_start_s
        details["sim_last_task_end_s"] = sim_last_task_end_s
        details["sim_first_quantum_start_s"] = sim_first_quantum_start_s
        details["sim_last_quantum_end_s"] = sim_last_quantum_end_s
        details["sim_result_ready_time_s"] = sim_result_ready_time_s
        details["sim_queue_wait_s"] = sim_queue_wait_s
        details["sim_execution_span_s"] = sim_execution_span_s
        details["sim_latency_s"] = sim_latency_s
        details["sim_comm_s"] = sim_comm_s
        details["sim_recon_s"] = sim_recon_s
        details["sim_submit_to_quantum_done_s"] = sim_submit_to_quantum_done_s
        details["charged_comm_s"] = float(t_comm_model)
        details["charged_recon_s"] = float(t_recon_model)

        rec = JobRunRecord(
            job_id=job.job_id,
            qpu_id=(plan.qpu_id if plan.qpu_id is not None else ("MULTI" if plan.kind == "C_CUT_MULTI_QPU" else None)),
            plan_kind=plan.kind,
            submit_time_s=float(getattr(job, "submit_time_s", 0.0) or 0.0),
            t_schedule_s=0.0,
            t_partition_s=0.0,
            t_mapping_s=0.0,
            t_execution_s=float(t_exec_model),
            t_reconstruction_s=float(t_recon_model),
            end_to_end_s=float(sim_latency_s),
            fidelity_proxy=getattr(plan, "predicted_fidelity_proxy", None),
            fidelity_estimated=None,
            details=details,
        )

        # Timing dicts: model vs wall
        timing_model = _timing_dict()
        timing_wall = _timing_dict()
        timing_model["execute_s"] = float(rec.t_execution_s)
        timing_model["communication_s"] = float(t_comm_model)
        timing_model["reconstruct_s"] = float(rec.t_reconstruction_s)
        timing_model["total_s"] = float(rec.t_execution_s + t_comm_model + rec.t_reconstruction_s)

        timing_wall["execute_s"] = float(details.get("wall_exec_s", 0.0) or 0.0)
        timing_wall["reconstruct_s"] = float(details.get("wall_recon_s", 0.0) or 0.0)
        timing_wall["total_s"] = float(t1_wall - t0_wall)
        rem = timing_wall["total_s"] - timing_wall["execute_s"] - timing_wall["reconstruct_s"]
        timing_wall["postprocess_s"] = float(rem) if rem > 0 else 0.0

        rec.details = _attach_timing_dicts(rec.details if isinstance(rec.details, dict) else {}, timing_model, timing_wall)
        rec.details["wall_end_to_end_s"] = float(t1_wall - t0_wall)
        return rec


    def _build_tasks_from_plan_details_only(self, job: Job, plan: Plan, now_s: float) -> Tuple[List[Task], Dict[str, Any]]:
        """Cheap fallback for timing-only runs when planner already supplied durations but not subcircuits."""
        tasks: List[Task] = []
        aux: Dict[str, Any] = {"task_reservations": []}
        det = plan.details or {}

        if plan.kind == "A_NO_CUT_SINGLE":
            return self.build_task_graph(job, plan, now_s)

        if plan.kind == "B_CUT_SINGLE_SEQ":
            qid = str(plan.qpu_id)
            labels_used = max(1, int(det.get("labels_used", 1) or 1))
            qdelay = float(det.get("pred_queue_delay_s", 0.0) or 0.0)
            total_exec = float(det.get("pred_exec_time_s", 0.0) or 0.0)
            sampling = float(det.get("sampling_overhead", 1.0) or 1.0)
            approx_total = max(0.0, total_exec * sampling)
            per_label = approx_total / float(labels_used)
            cursor = float(now_s) + qdelay
            max_local = int(det.get("max_local_qubits", max(1, job.circuit.num_qubits)) or max(1, job.circuit.num_qubits))
            k_each = max(1, min(max_local, job.circuit.num_qubits))
            for i in range(labels_used):
                tasks.append(Task(
                    task_id=f"{job.job_id}:B{i}",
                    job_id=job.job_id,
                    kind="quantum",
                    qpu_id=qid,
                    qubits=None,
                    start_s=float(cursor),
                    end_s=float(cursor + per_label),
                    label=int(i),
                    depends_on=None,
                    metadata={"plan": plan.kind, "k": int(k_each), "approx_from_plan": True, "subcircuit_index": i, "timing_mode": _effective_qpu_timing_mode(self.cfg)},
                ))
                cursor += per_label
            return tasks, aux

        if plan.kind == "C_CUT_MULTI_QPU":
            qpu_ids = list((det.get("qpu_candidates", []) or [])) or list(self.qpus.keys())
            assignment = det.get("assignment", {}) or {}
            labels_used = max(1, int(det.get("labels_used", len(assignment) or 1) or 1))
            qdelay = float(det.get("pred_queue_delay_s", 0.0) or 0.0)
            makespan = float(det.get("predicted_makespan_s", det.get("pred_exec_time_s", 0.0)) or 0.0)
            per_qpu_wait_s = dict(det.get("per_qpu_wait_s", {}) or {})
            max_local = int(det.get("max_local_qubits", max(1, min((self.qpus[q].max_connected_free_qubits(now_s) for q in qpu_ids), default=job.circuit.num_qubits))) or job.circuit.num_qubits)
            if not assignment:
                # round-robin fallback
                qpu_ids2 = qpu_ids or list(self.qpus.keys())
                assignment = {i: qpu_ids2[i % len(qpu_ids2)] for i in range(labels_used)}
            counts = {}
            for _lab, qid in assignment.items():
                counts[str(qid)] = counts.get(str(qid), 0) + 1
            per_label_dur = {}
            for qid, cnt in counts.items():
                usable = max(1, int(cnt))
                per_label_dur[qid] = max(0.0, makespan) / float(usable)
            cursor_by_qpu = {str(qid): float(now_s) + float(per_qpu_wait_s.get(str(qid), qdelay)) for qid in set(map(str, assignment.values()))}
            for i in range(labels_used):
                qid = str(assignment.get(i, assignment.get(str(i), qpu_ids[i % len(qpu_ids)] if qpu_ids else None)))
                dur = per_label_dur.get(qid, max(0.0, makespan))
                start = cursor_by_qpu.get(qid, float(now_s) + qdelay)
                tasks.append(Task(
                    task_id=f"{job.job_id}:C{i}",
                    job_id=job.job_id,
                    kind="quantum",
                    qpu_id=qid,
                    qubits=None,
                    start_s=float(start),
                    end_s=float(start + dur),
                    label=int(i),
                    depends_on=None,
                    metadata={"plan": plan.kind, "k": int(max_local), "approx_from_plan": True, "subcircuit_index": i, "timing_mode": _effective_qpu_timing_mode(self.cfg)},
                ))
                cursor_by_qpu[qid] = float(start + dur)
            aux["assignment"] = assignment
            return tasks, aux

        return tasks, aux

    def run_job_plan(self, job: Job, plan: Plan, now_s: float, toggles: RunToggles) -> Any:
        if plan.kind == "D_WAIT":
            return None

        t0_wall = time.perf_counter()

        # Phase 1: build tasks (quantum tasks only)
        timing_only = not bool(getattr(toggles, "compute_expectation", True))
        has_stored_partition = bool((plan.details or {}).get("partition_subcircuits", None))
        if timing_only and plan.kind in ("B_CUT_SINGLE_SEQ", "C_CUT_MULTI_QPU") and not has_stored_partition:
            tasks, aux = self._build_tasks_from_plan_details_only(job, plan, now_s)
        else:
            tasks, aux = self.build_task_graph(job, plan, now_s)

        # Phase 2: reserve resources
        reserve = bool(getattr(toggles, "simulate_only", False) or getattr(self.cfg, "reserve_nonsim", False))
        reserve_info = self.apply_reservations(tasks, now_s=now_s, reserve=reserve)

        # Phase 3: conditionally execute asynchronously
        exec_toggles = toggles
        if plan.kind == "D_WAIT" or plan.kind == "A_NO_CUT_SINGLE" or not getattr(toggles, "compute_expectation", True):
            out, exec_extra = self.maybe_execute(job, plan, aux, toggles)
            t1_wall = time.perf_counter()
            rec = self.assemble_record(
                job, plan, now_s=now_s,
                t0_wall=t0_wall, t1_wall=t1_wall,
                tasks=tasks, aux=aux, reserve_info=reserve_info, exec_extra=exec_extra,
            )
            self.metrics.add(rec)
            return out, rec

        # For heavy simulation jobs (Plan B/C), assemble the record with predictions first
        exec_extra = {}
        t1_wall = time.perf_counter()
        rec = self.assemble_record(
            job, plan, now_s=now_s,
            t0_wall=t0_wall, t1_wall=t1_wall,
            tasks=tasks, aux=aux, reserve_info=reserve_info, exec_extra=exec_extra,
        )
        self.metrics.add(rec)

        global _async_qpu_pool
        if "_async_qpu_pool" not in globals():
            from concurrent.futures import ThreadPoolExecutor
            import os
            workers = int(os.getenv("QDC_ASYNC_EXEC_WORKERS", "4"))
            global _async_qpu_pool
            _async_qpu_pool = ThreadPoolExecutor(max_workers=workers)

        def _bg_execute():
            bg_out, bg_extra = self.maybe_execute(job, plan, aux, exec_toggles)
            # Update the record in-place once finished so logs contain real-time info
            if rec.details is None:
                rec.details = {}
            rec.details.update(bg_extra)
            if bg_out is not None:
                rec.details["async_final_out"] = bg_out
            return bg_out, bg_extra

        fut = _async_qpu_pool.submit(_bg_execute)
        if rec.details is None:
            rec.details = {}
        rec.details["async_eval_future"] = fut

        return None, rec