from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import hashlib
import os
import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit_aer.primitives import SamplerV2, EstimatorV2
from qiskit_aer.noise import NoiseModel

from .types import Job, Plan, RunToggles, Task, TaskGraph
from .hardware import QPUState
from .quality import QualityModel
from .runtime import predict_exec_time_s
from .metrics import MetricsRecorder, JobRunRecord
from ..cutting import CutStrategy, FitCutCutStrategy, CutConstraints
from ..cutting.assignment import MinMakespanGreedyAssignment

def _timing_dict() -> dict:
    return {"transpile_s": 0.0, "execute_s": 0.0, "reconstruct_s": 0.0, "postprocess_s": 0.0, "total_s": 0.0}


def _attach_timing_dicts(details: dict, timing_model: dict, timing_wall: dict) -> dict:
    details = details or {}
    details["timing_model_s"] = timing_model
    details["timing_wall_s"] = timing_wall
    details.setdefault(
        "timing",
        {
            "transpile_s": float(timing_wall.get("transpile_s", 0.0)),
            "execute_s": float(timing_model.get("execute_s", 0.0)),
            "reconstruct_s": float(timing_model.get("reconstruct_s", 0.0)),
            "postprocess_s": float(timing_wall.get("postprocess_s", 0.0)),
            "total_measured_s": float(timing_wall.get("total_s", 0.0)),
        },
    )
    return details


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
    """Append optional COMM0 and RECON0 tasks.

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
    cut_constraints: CutConstraints = CutConstraints()
    cut_strategy: CutStrategy = FitCutCutStrategy()
    partition_assignment_policy: Any = MinMakespanGreedyAssignment()

class Executor:
    def __init__(self, qpus: Dict[str, QPUState], quality: QualityModel, metrics: MetricsRecorder, cfg: ExecConfig):
        self.qpus = qpus
        self.quality = quality
        self.metrics = metrics
        self.cfg = cfg
        self._aer_timing_cache: Dict[tuple, float] = {}

    def _hash_circuit(self, circ: QuantumCircuit) -> str:
        try:
            s = circ.qasm()
        except Exception:
            s = repr(circ)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

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

        mode = str(getattr(self.cfg, "timing_mode", "analytic") or "analytic").lower()
        if mode != "aer":
            from qdc_sched.core.profiler import profile_circuit
            sp = profile_circuit(circ)
            return float(predict_exec_time_s(self.qpus[qid].profile, sp, shots=eff_shots))

        key = (qid, self._hash_circuit(circ), eff_shots, str(task_type), bool(getattr(self.cfg, "aer_timing_use_noise", False)))
        if key in self._aer_timing_cache:
            return float(self._aer_timing_cache[key])

        try:
            dur = self._measure_exec_time_s_aer(
                circ=circ,
                shots=eff_shots,
                task_type=str(task_type),
                observables=observables,
                use_noise=bool(getattr(self.cfg, "aer_timing_use_noise", False)),
                repeats=int(getattr(self.cfg, "aer_timing_repeats", 1) or 1),
            )
        except Exception:
            from qdc_sched.core.profiler import profile_circuit
            sp = profile_circuit(circ)
            dur = float(predict_exec_time_s(self.qpus[qid].profile, sp, shots=eff_shots))

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
                metadata={"plan": plan.kind, "k": len(qubits)},
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

            part = self.cfg.cut_strategy.partition(job.circuit, cc, {"max_local_qubits": max_local, "observable": job.observables, "qpu_candidates": [qid]})
            aux["partition"] = part
            aux["cut_metadata"] = {"k_wire": getattr(part, "k_wire", None), "k_gate": getattr(part, "k_gate", None), "est_executions": getattr(part, "est_executions", None)}
            det = plan.details or {}
            qdelay = float(det.get("pred_queue_delay_s", float(getattr(qpu.profile, "base_queue_delay_s", 0.0)) or 0.0))
            cursor = float(now_s) + qdelay

            from qdc_sched.core.profiler import profile_circuit
            for i, sc in enumerate(getattr(part, "subcircuits", []) or []):
                sp = profile_circuit(sc)
                k = max(1, int(getattr(sp, "width", 1) or 1))
                dur = float(predict_exec_time_s(qpu.profile, sp, shots=int(job.shots or 0) or self.cfg.shots_default))
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
                    metadata={"plan": plan.kind, "k": k, "subcircuit_index": i},
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
                    label_qpu_pred[i][qid] = float(predict_exec_time_s(self.qpus[qid].profile, sp, shots=int(job.shots or 0) or self.cfg.shots_default))

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
            det = plan.details or {}
            qdelay = float(det.get("pred_queue_delay_s", 0.0) or 0.0)
            cursors = {qid: float(now_s) + qdelay for qid in qpu_ids}

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
                    metadata={"plan": plan.kind, "k": label_k[i], "subcircuit_index": i},
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

        # Plan B/C: expectation jobs support reconstruction
        part = aux.get("partition", None)
        if part is None:
            return None, extra

        # If no reconstruction metadata, nothing to execute/reconstruct
        if not isinstance(getattr(part, "reconstruction", None), dict) or "subexperiments" not in part.reconstruction:
            extra["cut_error"] = getattr(part, "reconstruction", None)
            return None, extra

        subexperiments = part.reconstruction["subexperiments"]
        coefficients = part.reconstruction["coefficients"]
        subobservables = part.reconstruction["subobservables"]

        # Fast path: if compute_expectation is False, don't execute heavy workloads
        if not bool(getattr(toggles, "compute_expectation", True)):
            extra["execution_mode"] = "timing_only"
            return None, extra

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
            for lab, circs in subexperiments.items():
                qid = str(assignment.get(int(lab), assignment.get(lab)))
                nm = _safe_noise_model(getattr(self.quality, "noise_models", {}).get(qid))
                sampler = SamplerV2(options={"backend_options": {"noise_model": nm}}) if nm is not None else SamplerV2()
                pubs = [(c, None, int(job.shots or 0) or self.cfg.shots_default) for c in circs]
                results_by_label[lab] = sampler.run(pubs).result()

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

        # Model-facing execution/recon times
        t_exec_model = 0.0
        t_recon_model = 0.0

        if plan.kind == "A_NO_CUT_SINGLE":
            t_exec_model = float((plan.details or {}).get("pred_exec_time_s", 0.0) or 0.0)
        elif plan.kind == "B_CUT_SINGLE_SEQ":
            # sum sequential quantum durations for model execution time
            t_exec_model = float(_max_end_s([t for t in tasks if t.kind == "quantum"], default=0.0) - min((t.start_s for t in tasks if t.kind == "quantum"), default=0.0))
            t_recon_model = float(details.get("recon_est_s", 0.0) or details.get("wall_recon_s", 0.0) or 0.0)
        elif plan.kind == "C_CUT_MULTI_QPU":
            # makespan (parallel) + comm overhead for model execution
            comm = float(getattr(job.constraints, "comm_overhead_s", 0.0) or 0.0) if getattr(job.constraints, "allow_multi_qpu", False) else 0.0
            q_tasks = [t for t in tasks if t.kind == "quantum"]
            makespan = 0.0
            if q_tasks:
                makespan = float(max(t.end_s for t in q_tasks) - min(t.start_s for t in q_tasks))
            t_exec_model = makespan + comm
            t_recon_model = float(details.get("recon_est_s", 0.0) or details.get("wall_recon_s", 0.0) or 0.0)

        # Append explicit COMM/RECON tasks using charged values
        comm_over = 0.0
        if plan.kind == "C_CUT_MULTI_QPU":
            comm_over = float(getattr(job.constraints, "comm_overhead_s", 0.0) or 0.0) if getattr(job.constraints, "allow_multi_qpu", False) else 0.0
        tasks2 = _append_comm_and_recon_tasks(tasks, job_id=job.job_id, plan_kind=plan.kind, comm_overhead_s=comm_over, recon_s=t_recon_model)

        details = _ensure_taskgraph(details, job.job_id, tasks2)

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
            end_to_end_s=float(t1_wall - t0_wall),
            fidelity_proxy=getattr(plan, "predicted_fidelity_proxy", None),
            fidelity_estimated=None,
            details=details,
        )

        # Timing dicts: model vs wall
        timing_model = _timing_dict()
        timing_wall = _timing_dict()
        timing_model["execute_s"] = float(rec.t_execution_s)
        timing_model["reconstruct_s"] = float(rec.t_reconstruction_s)
        timing_model["total_s"] = float(rec.t_execution_s + rec.t_reconstruction_s)

        timing_wall["execute_s"] = float(details.get("wall_exec_s", 0.0) or 0.0)
        timing_wall["reconstruct_s"] = float(details.get("wall_recon_s", 0.0) or 0.0)
        timing_wall["total_s"] = float(rec.end_to_end_s)
        rem = timing_wall["total_s"] - timing_wall["execute_s"] - timing_wall["reconstruct_s"]
        timing_wall["postprocess_s"] = float(rem) if rem > 0 else 0.0

        rec.details = _attach_timing_dicts(rec.details if isinstance(rec.details, dict) else {}, timing_model, timing_wall)
        return rec

    def run_job_plan(self, job: Job, plan: Plan, now_s: float, toggles: RunToggles) -> Any:
        if plan.kind == "D_WAIT":
            return None

        t0_wall = time.perf_counter()

        # Phase 1: build tasks (quantum tasks only)
        tasks, aux = self.build_task_graph(job, plan, now_s)

        # Phase 2: reserve resources
        reserve = bool(getattr(toggles, "simulate_only", False) or getattr(self.cfg, "reserve_nonsim", False))
        reserve_info = self.apply_reservations(tasks, now_s=now_s, reserve=reserve)

        # Phase 3: maybe execute (Aer primitives) + reconstruction
        out, exec_extra = self.maybe_execute(job, plan, aux, toggles)

        # Phase 4: assemble record (also adds COMM/RECON tasks to graph)
        t1_wall = time.perf_counter()
        rec = self.assemble_record(
            job, plan, now_s=now_s,
            t0_wall=t0_wall, t1_wall=t1_wall,
            tasks=tasks, aux=aux, reserve_info=reserve_info, exec_extra=exec_extra,
        )

        self.metrics.add(rec)
        return out, rec