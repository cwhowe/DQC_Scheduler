from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Set
import math
import signal
import time
import os
import itertools

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .types import Job, CircuitProfile, Plan, RunToggles
from ..cutting.base import CutConstraints
from ..cutting import CutStrategy, QiskitAddonCutStrategy, FitCutCutStrategy
from .hardware import QPUState
from .quality import QualityModel
from .runtime import predict_exec_time_s
from qdc_sched.cutting.assignment import MinMakespanGreedyAssignment
from qdc_sched.core.profiler import profile_circuit

class _qdc_time_limit:
    """Raise TimeoutError if the block runs longer than 'seconds'. Uses SIGALRM."""
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self._old_handler = None

    def __enter__(self):
        if self.seconds <= 0:
            return self
        self._old_handler = signal.getsignal(signal.SIGALRM)

        def _handler(signum, frame):
            raise TimeoutError(f"cut timed out after {self.seconds}s")

        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        except Exception:
            pass
        if self._old_handler is not None:
            try:
                signal.signal(signal.SIGALRM, self._old_handler)
            except Exception:
                pass
        return False

@dataclass
class PlannerConfig:
    top_k_qpus: int = 5
    cut_constraints: CutConstraints = CutConstraints()
    cut_strategy: CutStrategy = FitCutCutStrategy()
    weight_time: float = 1.0
    weight_quality: float = 1.0
    weight_fairness: float = 0.0

class Planner:
    def __init__(self, qpus: Dict[str, QPUState], quality: QualityModel, cfg: PlannerConfig):
        self.qpus = qpus
        self.quality = quality
        self.cfg = cfg

    def choose_plan(self, job: Job, prof: CircuitProfile, now_s: float, ranked_qpus) -> Plan:
        _planner_budget_s = float(os.getenv("QDC_PLANNER_BUDGET_S", "1.0"))
        _cut_timeout_s = float(os.getenv("QDC_CUT_TIMEOUT_S", "0.5"))
        _t0_plan = time.perf_counter()

        def _budget_exceeded() -> bool:
            return (time.perf_counter() - _t0_plan) > _planner_budget_s

        candidates: List[Plan] = []
        debug_scores = {}  # kind -> dict(pred_total, fid_proxy, score, slo_ok)

        def _wait_s(qpu_id: str, need: int, at_s: float) -> float:
            """Estimate additional wait (beyond base_queue_delay_s) at a specific time.

            Important: we probe at the *predicted start time* (now_s + base_queue_delay_s)
            so that future reservations (like congestion bursts) are correctly reflected in the estimate.
            """
            st = self.qpus.get(qpu_id)
            if st is None:
                return 0.0
            fn = getattr(st, "estimate_wait_s", None)
            if callable(fn):
                try:
                    return float(fn(int(need), float(at_s)))
                except Exception:
                    return 0.0
            return 0.0

        # -----------------
        # Plan A: no cut, single QPU
        # -----------------
        if not (job.task_type == "expectation" and job.constraints.allow_cutting and job.constraints.force_cutting):
            for (qpu_id, _, pred_t, fid_proxy) in ranked_qpus[: self.cfg.top_k_qpus]:
                st = self.qpus[qpu_id]
                free = st.find_free_connected_subgraph(prof.width, now_s)
                if free is None:
                    continue
                base_delay = float(getattr(st.profile, 'base_queue_delay_s', 0.0) or 0.0)
                start_probe_s = float(now_s) + float(base_delay)
                pred_queue = base_delay + _wait_s(qpu_id, prof.width, start_probe_s)
                pred_exec  = pred_t

                plan = Plan(
                    kind="A_NO_CUT_SINGLE",
                    qpu_id=qpu_id,
                    physical_qubits=sorted(list(free)),
                    predicted_total_time_s=pred_queue + pred_exec,
                    predicted_fidelity_proxy=float(fid_proxy),
                    details={
                        "pred_queue_delay_s": pred_queue,
                        "pred_exec_time_s": pred_exec,
                        "pred_recon_s": 0.0,
                        "pred_comm_s": 0.0,
                    }
                )
                candidates.append(plan)

        cut_qpu_pool = []
        for qpu_id, st in self.qpus.items():
            cap = st.max_connected_free_qubits(now_s)
            if cap > 1:
                fid_proxy = self.quality.fidelity_proxy_from_profile(qpu_id, prof)
                pred_t = float(ranked_qpus[0][2]) if ranked_qpus else 0.05
                cut_qpu_pool.append((qpu_id, pred_t, float(fid_proxy), int(cap)))

        candidates_qpus = [q for (q, _pt, _fp, _cap) in cut_qpu_pool]
        if not candidates_qpus:
            candidates_qpus = [q for (q, *_rest) in ranked_qpus] if ranked_qpus else list(self.qpus.keys())

        allow_cut = (job.task_type == "expectation" and job.constraints.allow_cutting and
                     (job.constraints.force_cutting
                      or prof.cut_suitability in ("good", "neutral")
                      or os.getenv("QDC_ALLOW_CUT_BAD", "0") == "1"))

        # -----------------
        # Plan B: cut, single QPU sequential
        # -----------------
        if allow_cut and not _budget_exceeded():
            for (qpu_id, _pred_t, fid_proxy, max_local) in cut_qpu_pool:
                cc0 = self.cfg.cut_constraints
                tl_raw = getattr(cc0, "target_labels", None)
                tl0 = 2 if (tl_raw is None) else int(tl_raw)

                max_local_safe = max(1, int(max_local))

                needed_labels = max(
                    tl0,
                    int(math.ceil(float(prof.width) / float(max_local_safe))),
                )
                chosen_labels = needed_labels if cc0.target_labels is None else int(cc0.target_labels)
                chosen_labels = min(int(cc0.target_labels_cap), int(chosen_labels))
                cc = CutConstraints(
                    max_cuts=min(cc0.max_cuts, job.constraints.max_cuts),
                    allow_gate_cuts=cc0.allow_gate_cuts,
                    allow_wire_cuts=cc0.allow_wire_cuts,
                    reconstruction_target=cc0.reconstruction_target,
                    target_labels=int(chosen_labels),
                    target_labels_cap=cc0.target_labels_cap,
                    seed_tries=cc0.seed_tries,
                )
                # Partition once (FitCut search is lightweight with caps. this makes time/labels explicit)
                part = None
                try:
                    with _qdc_time_limit(_cut_timeout_s):
                        part = self.cfg.cut_strategy.partition(job.circuit, cc, {"max_local_qubits": max_local, "qpu_candidates": [qpu_id]})
                except TimeoutError:
                    part = None
                except Exception as e:
                    if os.getenv("QDC_DEBUG_CUT_ERRORS", "0") == "1":
                        print(f"[CUT-ERR] job={job.job_id} qpu={qpu_id} err={e}")
                    part = None
                if part is None:
                    continue
                if (part is None) or (not part.subcircuits):
                    continue
                sub_profiles = [profile_circuit(sc) for sc in part.subcircuits]
                base_time = sum(predict_exec_time_s(self.qpus[qpu_id].profile, sp, shots=job.shots) for sp in sub_profiles)
                sampling = float(part.est_executions or 1.0)
                recon_base = float(getattr(self.qpus[qpu_id].profile, 'reconstruction_base_s', 0.0) or 0.0)
                recon_per = float(getattr(self.qpus[qpu_id].profile, 'reconstruction_per_exec_s', 0.0) or 0.0)
                recon_est = recon_base + recon_per * sampling
                max_width = max(int(sp.width) for sp in sub_profiles) if sub_profiles else int(prof.width)
                base_delay = float(getattr(self.qpus[qpu_id].profile, 'base_queue_delay_s', 0.0) or 0.0)
                start_probe_s = float(now_s) + float(base_delay)
                wait_b = base_delay + _wait_s(qpu_id, max_width, start_probe_s)
                pred_total = wait_b + base_time * sampling + recon_est
                pred_queue = float(wait_b)
                pred_exec  = float(base_time) 
                pred_recon = float(recon_est)
                pred_comm  = 0.0

                pred_total = pred_queue + (pred_exec * float(sampling)) + pred_recon + pred_comm

                candidates.append(Plan(
                    kind="B_CUT_SINGLE_SEQ",
                    qpu_id=qpu_id,
                    physical_qubits=None,
                    predicted_total_time_s=float(pred_total),
                    predicted_fidelity_proxy=float(fid_proxy),
                    details={
                        "pred_queue_delay_s": pred_queue,
                        "pred_exec_time_s": pred_exec,
                        "sampling_overhead": float(sampling),
                        "pred_recon_s": pred_recon,
                        "pred_comm_s": float(pred_comm),
                        "labels_used": len(part.subcircuits),
                        "recon_est_s": pred_recon,
                        "k_wire": int(part.k_wire),
                        "k_gate": int(part.k_gate),
                        "qpu_candidates": [qpu_id],
                        "max_local_qubits": int(max_local),
                    },
                ))


        # -----------------
        # Plan C: cut, multi-QPU parallel
        # -----------------
        if allow_cut and job.constraints.allow_multi_qpu and len(cut_qpu_pool) >= 2 and not _budget_exceeded():
            candidates_qpus = [c[0] for c in cut_qpu_pool]
            # Prefer distributing across QPUs that cannot fit the whole circuit (true multi-QPU case)
            # This prevents a large QPU from collapsing Plan C into a single-label "no-cut" partition
            if job.constraints.force_cutting:
                small = [c[0] for c in cut_qpu_pool if int(c[3]) < int(prof.width)]
                if len(small) >= 2:
                    candidates_qpus = small
            max_local = min(int(self.qpus[qid].max_connected_free_qubits(now_s)) for qid in candidates_qpus)  # smallest local capacity over chosen QPUs
            cc0 = self.cfg.cut_constraints
            # Ensure we ask for enough labels to make the circuit fit into max_local qubits.
            tl_raw = getattr(cc0, 'target_labels', None)
            tl0 = 2 if (tl_raw is None) else int(tl_raw)
            needed_labels = max(tl0, int(math.ceil(float(prof.width) / float(max_local))))
            chosen_labels = min(int(cc0.target_labels_cap), int(needed_labels))
            cc = CutConstraints(
                max_cuts=min(cc0.max_cuts, job.constraints.max_cuts),
                allow_gate_cuts=cc0.allow_gate_cuts,
                allow_wire_cuts=cc0.allow_wire_cuts,
                reconstruction_target=cc0.reconstruction_target,
                target_labels=int(chosen_labels),
                target_labels_cap=cc0.target_labels_cap,
                seed_tries=cc0.seed_tries,
            )
            part = None
            try:
                with _qdc_time_limit(_cut_timeout_s):
                    part = self.cfg.cut_strategy.partition(job.circuit, cc, {"max_local_qubits": max_local, "qpu_candidates": candidates_qpus})
            except TimeoutError:
                part = None
            except Exception as e:
                    if os.getenv("QDC_DEBUG_CUT_ERRORS", "0") == "1":
                        print(f"[CUT-ERR] job={job.job_id} qpu={qpu_id} err={e}")
                    part = None
            if part is None:
                # cut timed out / failed this step (skip scoring Plan C this tick)
                part = None
            # Plan C requires a distributable partition.
            if (not getattr(part, 'subcircuits', None)) or (len(part.subcircuits) < int(needed_labels)):
                part = None
            if part is not None:
                labels = list(range(len(part.subcircuits)))
                # Per-label cost proxy
                sub_profiles = [profile_circuit(sc) for sc in part.subcircuits]
                label_costs = {i: float(sp.twoq_count + 0.2 * sp.oneq_count) for i, sp in enumerate(sub_profiles)}
                # Per-label per-QPU predicted device-time
                label_qpu_pred_time = {i: {} for i in labels}
                for i, sp in enumerate(sub_profiles):
                    for qid in candidates_qpus:
                        label_qpu_pred_time[i][qid] = predict_exec_time_s(self.qpus[qid].profile, sp, shots=job.shots)

                qpu_caps = {qid: int(self.qpus[qid].max_connected_free_qubits(now_s)) for qid in candidates_qpus}
                qpu_quality = {qid: float(self.quality.fidelity_proxy_from_profile(qid, prof)) for qid in candidates_qpus}
                qpu_pred_time = {qid: float(self.qpus[qid].profile.base_queue_delay_s or 0.0) for qid in candidates_qpus}

                # Allow packing multiple labels onto the same QPU (still a multi-QPU plan if >=2 QPUs used).
                # This makes Plan C feasible even when the partitioner produces >#QPUs labels.
                allow_pack = os.getenv('QDC_C_ALLOW_PACK_LABELS', '1') not in ('0', 'false', 'False')
                if allow_pack:
                    # brute force is fine here: labels are tiny (usually <=4) and QPUs <=3
                    best = None  # (makespan, total, assignment, used_qpus)
                    for choice in itertools.product(candidates_qpus, repeat=len(labels)):
                        used = set(choice)
                        if len(used) < 2:
                            continue
                        loads = {q: 0.0 for q in used}
                        for lab, q in zip(labels, choice):
                            loads[q] += float(label_qpu_pred_time[lab][q])
                        makespan = max(loads.values()) if loads else float('inf')
                        total = sum(loads.values()) if loads else float('inf')
                        cand = (makespan, total, dict(zip(labels, choice)), used)
                        if best is None or cand[:2] < best[:2]:
                            best = cand
                    if best is None:
                        skips.setdefault('C', []).append('insufficient_cut_qpus')
                        # fall back to a degenerate single-QPU assignment (will make C unattractive but keeps us running)
                        assignment = {labels[0]: candidates_qpus[0]}
                        makespan = float('inf')
                    else:
                        makespan, _total, assignment, _used_qpus = best
                else:
                    assigner = MinMakespanGreedyAssignment()
                    assignment = assigner.assign(labels, candidates_qpus,
                                                 label_costs=label_costs,
                                                 qpu_caps=qpu_caps,
                                                 qpu_quality=qpu_quality,
                                                 qpu_pred_time=qpu_pred_time,
                                                 label_qpu_pred_time=label_qpu_pred_time)
                    # Makespan under assignment (one label per QPU)
                    makespan = 0.0
                    for lab in labels:
                        makespan = max(makespan, float(label_qpu_pred_time[lab][assignment[lab]]))

                sampling = float(part.est_executions or 1.0)
                # global reconstruction estimate (we do it once, this can be real too)
                any_q = candidates_qpus[0]
                recon_base = float(getattr(self.qpus[any_q].profile, 'reconstruction_base_s', 0.0) or 0.0)
                recon_per = float(getattr(self.qpus[any_q].profile, 'reconstruction_per_exec_s', 0.0) or 0.0)
                recon_est = recon_base + recon_per * sampling

                # Wait is dominated by the slowest participating QPU at its assigned label width.
                waits = []
                for lab, qpu_assigned in assignment.items():
                    need_w = int(sub_profiles[lab].width) if 0 <= int(lab) < len(sub_profiles) else int(prof.width)
                    base_d = float(getattr(self.qpus[qpu_assigned].profile, 'base_queue_delay_s', 0.0) or 0.0)
                    start_probe_s = float(now_s) + float(base_d)
                    waits.append(base_d + _wait_s(qpu_assigned, need_w, start_probe_s))
                wait_c = max(waits) if waits else 0.0
                pred_total = float(wait_c) + (float(makespan) * sampling) + float(job.constraints.comm_overhead_s) + float(recon_est)
                fid_proxy = max(float(self.quality.fidelity_proxy_from_profile(qid, prof)) for qid in candidates_qpus)
                pred_queue = float(wait_c)                      # max over assigned-QPU queue+wait
                pred_exec  = float(makespan)                    # service makespan (NO queue)
                pred_recon = float(recon_est)
                pred_comm  = float(job.constraints.comm_overhead_s or 0.0)

                pred_total = pred_queue + (pred_exec * float(sampling)) + pred_comm + pred_recon

                candidates.append(Plan(
                    kind="C_CUT_MULTI_QPU",
                    qpu_id=None,
                    physical_qubits=None,
                    predicted_total_time_s=float(pred_total),
                    predicted_fidelity_proxy=float(fid_proxy),
                    details={
                        "pred_queue_delay_s": pred_queue,
                        "pred_exec_time_s": pred_exec,
                        "sampling_overhead": float(sampling),
                        "pred_recon_s": pred_recon,
                        "pred_comm_s": pred_comm,
                        "qpu_candidates": candidates_qpus,
                        "labels_used": len(labels),
                        "recon_est_s": pred_recon,
                        "predicted_makespan_s": float(makespan),
                        "wait_est_s": float(wait_c),
                        "k_wire": int(part.k_wire),
                        "k_gate": int(part.k_gate),
                        "assignment": assignment,
                        "assignment_preview": assignment,
                    },
                ))


        # -----------------
        # Score + SLO filter
        # -----------------
        slo = job.constraints.slo_s
        best: Optional[Plan] = None
        for plan in candidates:
            if slo is not None and plan.predicted_total_time_s > float(slo):
                continue
            # Optional quality penalty
            plan.score = self.cfg.weight_time * float(plan.predicted_total_time_s)
            if self.cfg.weight_quality <= 0.0:
                debug_scores[f"{plan.kind}:{getattr(plan,'qpu_id',None)}"] = {
                    'pred_total_s': float(plan.predicted_total_time_s),
                    'fid_proxy': float(plan.predicted_fidelity_proxy),
                    'score': float(plan.score),
                    'slo_ok': (slo is None) or (float(plan.predicted_total_time_s) <= float(slo)),
                }
            if self.cfg.weight_quality > 0.0:
                plan.score += self.cfg.weight_quality * (1.0 - float(plan.predicted_fidelity_proxy))
            debug_scores[f"{plan.kind}:{getattr(plan,'qpu_id',None)}"] = {
                'pred_total_s': float(plan.predicted_total_time_s),
                'fid_proxy': float(plan.predicted_fidelity_proxy),
                'score': float(plan.score),
                'slo_ok': (slo is None) or (float(plan.predicted_total_time_s) <= float(slo)),
            }
            if best is None or plan.score < best.score:
                best = plan

        if best is not None:
            if best.details is None:
                best.details = {}
            best.details.setdefault('plan_scores', debug_scores)
            # Validate: do not mutate plan.kind here; if a malformed best plan slips through, fall back to WAIT.
            inconsistent = False
            if getattr(best, "kind", "") in ("A_NO_CUT", "B_CUT_SINGLE_SEQ") and getattr(best, "qpu_id", None) is None:
                inconsistent = True
            if getattr(best, "kind", "") == "C_CUT_MULTI_QPU" and getattr(best, "qpu_id", None) is not None:
                inconsistent = True
            if inconsistent:
                wait_details = {
                    "reason": "inconsistent_best_plan",
                    "best_kind": getattr(best, "kind", None),
                    "best_qpu_id": getattr(best, "qpu_id", None),
                    "plan_scores": debug_scores,
                }
                return Plan(kind="D_WAIT", predicted_total_time_s=float("inf"), score=float("inf"), details=wait_details)
            return best

        # No feasible plan under current constraints -> WAIT (include debug / feasibility info)
        wait_details = {
            "reason": "no_feasible_plan",
            "now_s": float(now_s),
            "job_width": int(getattr(prof, "width", 0) or 0),
            "allow_cutting": bool(getattr(getattr(job, "constraints", None), "allow_cutting", False)),
            "force_cutting": bool(getattr(getattr(job, "constraints", None), "force_cutting", False)),
            "allow_multi_qpu": bool(getattr(getattr(job, "constraints", None), "allow_multi_qpu", False)),
            "qpu_candidates": candidates_qpus,
            "plan_scores": debug_scores,
        }
        # Add per-QPU free connectivity info to explain why we are waiting
        try:
            max_free = {}
            reserved = {}
            for qid in candidates_qpus:
                st = self.qpus.get(qid)
                if st is None:
                    continue
                max_free[qid] = int(st.max_connected_free_qubits(now_s))
                rq = sorted(list(st.reserved_qubits(now_s)))
                reserved[qid] = rq[:32]
            wait_details["max_connected_free_qubits"] = max_free
            wait_details["reserved_qubits_sample"] = reserved
        except Exception:
            pass
        return Plan(kind="D_WAIT", predicted_total_time_s=float("inf"), score=float("inf"), details=wait_details)

def transpile_restricted(qc: QuantumCircuit, qpu: QPUState, physical_qubits: List[int], opt_level: int = 1) -> QuantumCircuit:
    """Transpile using a default/common basis so unsupported gates (like those in QFT) are decomposed.

    Important: if we leave basis_gates=None, the pass manager may try to keep some gates and then fail
    during directional routing because it doesn't know how to flip them on directed couplings.
    We avoid that by forcing decomposition into a common 1Q+2Q basis.
    """
    coupling = CouplingMap(qpu.profile.coupling_graph.edges())
    # A common basis across IBM backends
    basis_gates = ["rz", "sx", "x", "cx", "measure", "reset"]
    pm = generate_preset_pass_manager(
        optimization_level=opt_level,
        coupling_map=coupling,
        basis_gates=basis_gates,
    )
    tqc = pm.run(qc)
    return tqc