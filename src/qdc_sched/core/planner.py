from __future__ import annotations
from dataclasses import dataclass, field
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
from .runtime import estimate_qpu_execution_s, estimate_reconstruction_duration_s
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
    cut_constraints: CutConstraints = field(default_factory=CutConstraints)
    cut_strategy: CutStrategy = field(default_factory=FitCutCutStrategy)
    weight_time: float = 1.0
    weight_quality: float = 1.0
    weight_fairness: float = 0.0

class Planner:
    def __init__(self, qpus: Dict[str, QPUState], quality: QualityModel, cfg: PlannerConfig):
        self.qpus = qpus
        self.quality = quality
        self.cfg = cfg

    def _predict_host_reconstruction_s(self, sampling: float) -> tuple[float, dict]:
        return estimate_reconstruction_duration_s(
            num_subexperiments=float(sampling),
            num_samples=float(sampling),
        )

    def choose_plan(self, job: Job, prof: CircuitProfile, now_s: float, ranked_qpus) -> Plan:
        _planner_budget_s = float(os.getenv("QDC_PLANNER_BUDGET_S", "1.0"))
        _cut_timeout_s = float(os.getenv("QDC_CUT_TIMEOUT_S", "0.5"))
        _t0_plan = time.perf_counter()

        def _budget_exceeded() -> bool:
            return (time.perf_counter() - _t0_plan) > _planner_budget_s

        candidates: List[Plan] = []
        debug_scores = {}  # kind -> dict(pred_total, fid_proxy, score, slo_ok)

        _planner_approx_partition = os.getenv("QDC_PLANNER_APPROX_PARTITION", "1") not in ("0", "false", "False")
        _skip_plan_b_for_forced_wide = os.getenv("QDC_SKIP_PLAN_B_FOR_FORCED_WIDE", "1") not in ("0", "false", "False")
        _debug_plan_b = os.getenv("QDC_DEBUG_PLAN_B", "0") == "1"
        _planner_ignore_recon = os.getenv("QDC_PLANNER_IGNORE_RECON", "1") not in ("0", "false", "False")

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
            for (qpu_id, _, _ranked_pred_t, fid_proxy) in ranked_qpus[: self.cfg.top_k_qpus]:
                st = self.qpus[qpu_id]
                free = st.find_free_connected_subgraph(prof.width, now_s)
                if free is None:
                    continue
                base_delay = float(getattr(st.profile, 'base_queue_delay_s', 0.0) or 0.0)
                start_probe_s = float(now_s) + float(base_delay)
                pred_queue = base_delay + _wait_s(qpu_id, prof.width, start_probe_s)
                pred_exec, pred_exec_meta = estimate_qpu_execution_s(st.profile, circuit=job.circuit, prof=prof, shots=job.shots)

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
                        "pred_exec_meta": pred_exec_meta,
                    }
                )
                candidates.append(plan)

        cut_qpu_pool = []
        for qpu_id, st in self.qpus.items():
            cap = st.max_connected_free_qubits(now_s)
            if cap > 1:
                fid_proxy = self.quality.fidelity_proxy_from_profile(qpu_id, prof)
                try:
                    pred_t, _pred_meta = estimate_qpu_execution_s(st.profile, circuit=job.circuit, prof=prof, shots=job.shots)
                    pred_t = float(pred_t)
                except Exception:
                    pred_t = 0.05
                cut_qpu_pool.append((qpu_id, pred_t, float(fid_proxy), int(cap)))

        candidates_qpus = [q for (q, _pt, _fp, _cap) in cut_qpu_pool]
        max_single_free = max((int(cap) for (_q, _pt, _fp, cap) in cut_qpu_pool), default=0)
        if not candidates_qpus:
            candidates_qpus = [q for (q, *_rest) in ranked_qpus] if ranked_qpus else list(self.qpus.keys())

        allow_cut = (job.task_type == "expectation" and job.constraints.allow_cutting and
                     (job.constraints.force_cutting
                      or prof.cut_suitability in ("good", "neutral")
                      or os.getenv("QDC_ALLOW_CUT_BAD", "0") == "1"))

        # -----------------
        # Plan B: cut, single QPU sequential
        # -----------------
        skip_plan_b = (
            _skip_plan_b_for_forced_wide
            and bool(getattr(job.constraints, 'force_cutting', False))
            and bool(getattr(job.constraints, 'allow_multi_qpu', False))
            and int(getattr(prof, 'width', 0) or 0) > int(max_single_free)
        )
        if allow_cut and skip_plan_b and _debug_plan_b:
            print(f"[PLAN-B-SKIP] job={job.job_id} reason=forced_wide_prefers_multi width={int(getattr(prof, 'width', 0) or 0)} max_single_free={int(max_single_free)}")

        if allow_cut and (not skip_plan_b) and not _budget_exceeded():
            for (qpu_id, _pred_t, fid_proxy, max_local) in cut_qpu_pool:
                if _budget_exceeded():
                    break
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
                if _debug_plan_b:
                    print(f"[PLAN-B-TRY] job={job.job_id} qpu={qpu_id} max_local={max_local} target_labels={chosen_labels} approx={_planner_approx_partition}")
                try:
                    with _qdc_time_limit(_cut_timeout_s):
                        part = self.cfg.cut_strategy.partition(job.circuit, cc, {"max_local_qubits": max_local, "qpu_candidates": [qpu_id], "approx_only": _planner_approx_partition, "observable": getattr(job, "observables", None)})
                except TimeoutError:
                    if _debug_plan_b:
                        print(f"[PLAN-B-SKIP] job={job.job_id} qpu={qpu_id} reason=cut_timeout timeout_s={_cut_timeout_s}")
                    part = None
                except Exception as e:
                    if os.getenv("QDC_DEBUG_CUT_ERRORS", "0") == "1":
                        print(f"[CUT-ERR] job={job.job_id} qpu={qpu_id} err={e}")
                    part = None
                if part is None:
                    continue
                if (part is None) or (not part.subcircuits):
                    continue
                sub_profiles = []
                for sc in part.subcircuits:
                    if _budget_exceeded():
                        break
                    sub_profiles.append(profile_circuit(sc))
                if len(sub_profiles) != len(part.subcircuits):
                    break
                base_time = 0.0
                sub_timing_meta = []
                for sc, sp in zip(part.subcircuits, sub_profiles):
                    sub_t, sub_meta = estimate_qpu_execution_s(self.qpus[qpu_id].profile, circuit=sc, prof=sp, shots=job.shots)
                    base_time += float(sub_t)
                    sub_timing_meta.append(sub_meta)
                sampling = float(part.est_executions or 1.0)
                recon_est, recon_meta = self._predict_host_reconstruction_s(sampling)
                max_width = max(int(sp.width) for sp in sub_profiles) if sub_profiles else int(prof.width)
                base_delay = float(getattr(self.qpus[qpu_id].profile, 'base_queue_delay_s', 0.0) or 0.0)
                start_probe_s = float(now_s) + float(base_delay)
                wait_b = base_delay + _wait_s(qpu_id, max_width, start_probe_s)
                pred_total = wait_b + base_time * sampling + recon_est
                pred_queue = float(wait_b)
                pred_exec  = float(base_time) 
                pred_recon = float(recon_est)
                pred_comm  = 0.0

                pred_total = pred_queue + (pred_exec * float(sampling)) + pred_comm
                if not _planner_ignore_recon:
                    pred_total += pred_recon

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
                        "k_wire": int(getattr(part, "k_wire", 0) or 0),
                        "k_gate": int(getattr(part, "k_gate", 0) or 0),
                        "qpu_candidates": [qpu_id],
                        "max_local_qubits": int(max_local),
                        "target_labels": int(chosen_labels),
                        "partition_subcircuits": list(getattr(part, "subcircuits", []) or []),
                        "sub_qpu_timing_meta": sub_timing_meta,
                        "recon_timing_meta": recon_meta,
                    },
                ))


        # -----------------
        # Plan C: cut, multi-QPU parallel
        # -----------------

        if allow_cut and job.constraints.allow_multi_qpu and len(cut_qpu_pool) >= 2 and not _budget_exceeded():
            all_candidates_qpus = [c[0] for c in cut_qpu_pool]
            cc0 = self.cfg.cut_constraints
            tl_raw = getattr(cc0, 'target_labels', None)
            tl0 = 2 if (tl_raw is None) else int(tl_raw)
            allow_pack = os.getenv('QDC_C_ALLOW_PACK_LABELS', '1') not in ('0', 'false', 'False')
            debug_c = os.getenv("QDC_DEBUG_PLAN_C", "0") == "1"
            max_exact_labels = int(os.getenv("QDC_C_EXACT_PACK_MAX_LABELS", "8"))
            max_exact_states = int(os.getenv("QDC_C_EXACT_PACK_MAX_STATES", "256"))
            max_c_subsets = int(os.getenv("QDC_C_MAX_SUBSETS", "8"))


            seen_subset_sig = set()
            subset_list = []
            max_c_subset_size = int(os.getenv("QDC_C_MAX_SUBSET_SIZE", "3"))
            for r in range(2, min(len(all_candidates_qpus), max_c_subset_size) + 1):
                for subset in itertools.combinations(all_candidates_qpus, r):
                    sig = tuple(sorted(subset))
                    if sig in seen_subset_sig:
                        continue
                    seen_subset_sig.add(sig)
                    subset_list.append(list(subset))
            
            # Sort subset_list descending by bottleneck capacity so we evaluate the best pairs/triplets first
            subset_list.sort(key=lambda s: min(int(self.qpus[qid].max_connected_free_qubits(now_s)) for qid in s), reverse=True)
            subset_list = subset_list[:max(1, max_c_subsets)]

            for candidates_qpus in subset_list:
                if _budget_exceeded():
                    if debug_c:
                        print(f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} reason=planner_budget_exceeded")
                    break

                max_local = min(int(self.qpus[qid].max_connected_free_qubits(now_s)) for qid in candidates_qpus)
                if max_local <= 0:
                    if debug_c:
                        print(f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} reason=no_free_capacity max_local={max_local}")
                    continue

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
                        part = self.cfg.cut_strategy.partition(
                            job.circuit,
                            cc,
                            {"max_local_qubits": max_local, "qpu_candidates": candidates_qpus, "approx_only": _planner_approx_partition, "observable": getattr(job, "observables", None)},
                        )
                except TimeoutError:
                    if debug_c:
                        print(
                            f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} "
                            f"reason=cut_timeout timeout_s={_cut_timeout_s} max_local={max_local} "
                            f"needed_labels={needed_labels}"
                        )
                    part = None
                except Exception as e:
                    if debug_c or os.getenv("QDC_DEBUG_CUT_ERRORS", "0") == "1":
                        print(f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} reason=cut_error err={e}")
                    part = None

                if part is None:
                    continue

                if not getattr(part, 'subcircuits', None):
                    if debug_c:
                        print(f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} reason=no_subcircuits")
                    continue

                if len(part.subcircuits) < int(needed_labels):
                    if debug_c:
                        print(
                            f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} "
                            f"reason=too_few_labels got={len(part.subcircuits)} needed={needed_labels}"
                        )
                    continue

                labels = list(range(len(part.subcircuits)))
                sub_profiles = []
                for sc in part.subcircuits:
                    if _budget_exceeded():
                        break
                    sub_profiles.append(profile_circuit(sc))
                if len(sub_profiles) != len(part.subcircuits):
                    if debug_c:
                        print(f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} reason=planner_budget_during_profile")
                    break

                label_costs = {i: float(sp.twoq_count + 0.2 * sp.oneq_count) for i, sp in enumerate(sub_profiles)}
                label_qpu_pred_time = {i: {} for i in labels}
                sub_timing_meta = {i: {} for i in labels}

                for i, (sc, sp) in enumerate(zip(part.subcircuits, sub_profiles)):
                    if _budget_exceeded():
                        break
                    for qid in candidates_qpus:
                        exec_s, timing_meta = estimate_qpu_execution_s(
                            self.qpus[qid].profile,
                            circuit=sc,
                            prof=sp,
                            shots=job.shots,
                        )
                        label_qpu_pred_time[i][qid] = float(exec_s)
                        sub_timing_meta[i][qid] = dict(timing_meta or {})
                        sub_timing_meta[i][qid].setdefault("pred_exec_time_s", float(exec_s))
                if any(len(v) != len(candidates_qpus) for v in label_qpu_pred_time.values()):
                    if debug_c:
                        print(f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} reason=planner_budget_during_pred_time")
                    break

                qpu_caps = {qid: int(self.qpus[qid].max_connected_free_qubits(now_s)) for qid in candidates_qpus}
                qpu_quality = {qid: float(self.quality.fidelity_proxy_from_profile(qid, prof)) for qid in candidates_qpus}
                qpu_pred_time = {qid: float(self.qpus[qid].profile.base_queue_delay_s or 0.0) for qid in candidates_qpus}

                n_labels = len(labels)
                n_qpus = len(candidates_qpus)
                search_states = n_qpus ** n_labels
                use_exact_pack = allow_pack and n_labels <= max_exact_labels and search_states <= max_exact_states

                if debug_c:
                    print(
                        f"[PLAN-C-INFO] job={job.job_id} pair={candidates_qpus} "
                        f"needed_labels={needed_labels} got_labels={len(part.subcircuits)} "
                        f"search_states={search_states} exact_pack={use_exact_pack}"
                    )

                if use_exact_pack:
                    best_assign = None  # (makespan, total, assignment, used_qpus)
                    for choice in itertools.product(candidates_qpus, repeat=n_labels):
                        if _budget_exceeded():
                            if debug_c:
                                print(f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} reason=planner_budget_during_pack")
                            break
                        used = set(choice)
                        if len(used) < 2:
                            continue
                        loads = {q: 0.0 for q in used}
                        for lab, q in zip(labels, choice):
                            loads[q] += float(label_qpu_pred_time[lab][q])
                        makespan = max(loads.values()) if loads else float('inf')
                        total = sum(loads.values()) if loads else float('inf')
                        cand = (makespan, total, dict(zip(labels, choice)), used)
                        if best_assign is None or cand[:2] < best_assign[:2]:
                            best_assign = cand
                    if best_assign is None:
                        if debug_c:
                            print(f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} reason=no_valid_exact_assignment")
                        continue
                    makespan, _total, assignment, used_qpus = best_assign
                else:
                    assigner = MinMakespanGreedyAssignment()
                    assignment = assigner.assign(
                        labels,
                        candidates_qpus,
                        label_costs=label_costs,
                        qpu_caps=qpu_caps,
                        qpu_quality=qpu_quality,
                        qpu_pred_time=qpu_pred_time,
                        label_qpu_pred_time=label_qpu_pred_time,
                    )
                    used_qpus = set(assignment.values())
                    if len(used_qpus) < 2:
                        if debug_c:
                            print(f"[PLAN-C-SKIP] job={job.job_id} pair={candidates_qpus} reason=assignment_not_multi_qpu")
                        continue
                    loads = {q: 0.0 for q in used_qpus}
                    for lab, q in assignment.items():
                        loads[q] += float(label_qpu_pred_time[lab][q])
                    makespan = max(loads.values()) if loads else float('inf')
                sampling = float(part.est_executions or 1.0)
                per_qpu_wait_s = {}
                for qid in used_qpus:
                    assigned_labs = [lab for lab, q in assignment.items() if q == qid]
                    need_w = max(int(sub_profiles[lab].width) for lab in assigned_labs) if assigned_labs else int(prof.width)
                    base_d = float(getattr(self.qpus[qid].profile, 'base_queue_delay_s', 0.0) or 0.0)
                    start_probe_s = float(now_s) + float(base_d)
                    per_qpu_wait_s[qid] = base_d + _wait_s(qid, need_w, start_probe_s)

                pred_finish = 0.0
                for qid in used_qpus:
                    q_load = sum(float(label_qpu_pred_time[lab][qid]) for lab, q in assignment.items() if q == qid)
                    pred_finish = max(pred_finish, float(per_qpu_wait_s.get(qid, 0.0)) + (q_load * sampling))

                recon_est, recon_meta = self._predict_host_reconstruction_s(sampling)

                used_qpus_list = sorted(list(used_qpus))
                pred_comm = float(job.constraints.comm_overhead_s or 0.0) if len(used_qpus_list) >= 2 else 0.0
                pred_queue = max(float(per_qpu_wait_s.get(qid, 0.0)) for qid in used_qpus_list) if used_qpus_list else 0.0
                pred_exec = max(
                    sum(float(label_qpu_pred_time[lab][qid]) for lab, q in assignment.items() if q == qid)
                    for qid in used_qpus_list
                ) if used_qpus_list else float('inf')
                pred_recon = float(recon_est)
                pred_total = float(pred_finish) + pred_comm
                if not _planner_ignore_recon:
                    pred_total += pred_recon
                fid_proxy = max(float(self.quality.fidelity_proxy_from_profile(qid, prof)) for qid in used_qpus_list)

                if debug_c:
                    print(
                        f"[PLAN-C-CAND] job={job.job_id} pair={candidates_qpus} used={used_qpus_list} "
                        f"labels={len(labels)} sampling={sampling:.3f} "
                        f"pred_queue={pred_queue:.6f} pred_exec={pred_exec:.6f} "
                        f"pred_comm={pred_comm:.6f} pred_recon={pred_recon:.6f} total={pred_total:.6f}"
                    )

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
                        "qpu_candidates": list(candidates_qpus),
                        "used_qpus": used_qpus_list,
                        "subset_signature": "+".join(sorted(used_qpus_list)),
                        "labels_used": len(labels),
                        "recon_est_s": pred_recon,
                        "predicted_makespan_s": float(pred_exec),
                        "wait_est_s": float(pred_queue),
                        "per_qpu_wait_s": dict(per_qpu_wait_s),
                        "k_wire": int(getattr(part, "k_wire", 0) or 0),
                        "k_gate": int(getattr(part, "k_gate", 0) or 0),
                        "assignment": assignment,
                        "assignment_preview": assignment,
                        "max_local_qubits": int(max_local),
                        "target_labels": int(chosen_labels),
                        "partition_subcircuits": list(getattr(part, "subcircuits", []) or []),
                        "sub_qpu_timing_meta": sub_timing_meta,
                        "recon_timing_meta": recon_meta,
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

            plan.score = self.cfg.weight_time * float(plan.predicted_total_time_s)
            if self.cfg.weight_quality > 0.0:
                plan.score += self.cfg.weight_quality * (1.0 - float(plan.predicted_fidelity_proxy))

            if getattr(plan, "kind", "") == "C_CUT_MULTI_QPU":
                used = plan.details.get("used_qpus", []) if plan.details else []
                cand = plan.details.get("qpu_candidates", []) if plan.details else []
                dbg_key = f"{plan.kind}:{'+'.join(map(str, used)) or 'none'}|cand={'+'.join(map(str, cand)) or 'none'}"
            else:
                dbg_key = f"{plan.kind}:{getattr(plan, 'qpu_id', None)}"

            debug_scores[dbg_key] = {
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