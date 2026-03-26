from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList, SparsePauliOp

from .base import CutAnalysis, CutConstraints, CutStrategy, PartitionPlan


_QDC_CUT_FAST_PARTITION = os.environ.get("QDC_CUT_FAST_PARTITION", "0") not in ("0", "", "false", "False")
_QDC_CUT_FALLBACK_PARTITION = os.environ.get("QDC_CUT_FALLBACK_PARTITION", "1") not in ("0", "", "false", "False")


def _naive_chunk_partition(circuit: QuantumCircuit, max_subq: int):
    """Split logical qubits into contiguous chunks of size <= max_subq.

    This is intentionally approximate and is meant for planner/demo timing only.
    Cross-chunk operations are dropped so width/depth/2q-count stay cheap to estimate.
    """
    from qiskit.circuit import QuantumCircuit as Qc

    n = circuit.num_qubits
    if max_subq <= 0 or max_subq >= n:
        return [(0, circuit)]

    chunks = []
    label = 0
    for start in range(0, n, max_subq):
        qs = list(range(start, min(n, start + max_subq)))
        sub = Qc(len(qs), name=f"sub_{label}")
        idx_map = {q: i for i, q in enumerate(qs)}
        for inst, qargs, cargs in circuit.data:
            qids = [circuit.find_bit(q).index for q in qargs]
            if all(q in idx_map for q in qids):
                sub.append(inst, [idx_map[q] for q in qids], [])
        chunks.append((label, sub))
        label += 1
    return chunks


def _to_paulilist(obs: Any) -> PauliList:
    if isinstance(obs, PauliList):
        return obs
    if isinstance(obs, SparsePauliOp):
        return PauliList(obs.paulis)
    if isinstance(obs, str):
        return PauliList([obs])
    raise TypeError(f"Unsupported observable type for cutting: {type(obs)}")


def _coerce_observables(obs: Any, n_qubits: int) -> PauliList:
    if obs is None:
        return PauliList(["Z" + "I" * (n_qubits - 1)])
    if isinstance(obs, (list, tuple)):
        pls = []
        for o in obs:
            pls.extend(list(_to_paulilist(o)))
        return PauliList(pls)
    return _to_paulilist(obs)


def _strip_classical_bits(circ: QuantumCircuit) -> QuantumCircuit:
    if circ.num_clbits == 0:
        return circ
    new = QuantumCircuit(circ.num_qubits)
    for inst, qargs, cargs in circ.data:
        if inst.name == "measure":
            continue
        if len(cargs) > 0:
            continue
        new.append(inst, qargs)
    return new


def _fast_partition_plan(
    circ: QuantumCircuit,
    max_local: int,
    *,
    num_samples: int,
    reason: str,
    target_labels: Optional[int] = None,
) -> PartitionPlan:
    n = int(getattr(circ, "num_qubits", 0) or 0)
    if target_labels is not None and int(target_labels) >= 2 and n >= 2:
        n_chunks = max(2, min(int(target_labels), n))
        chunk_size = max(1, math.ceil(n / n_chunks))
        naive = _naive_chunk_partition(circ, chunk_size)
        if len(naive) < 2:
            naive = _naive_chunk_partition(circ, max(1, n // 2))
    else:
        naive = _naive_chunk_partition(circ, max_local)
    subcircuits = {lbl: sc for (lbl, sc) in naive}
    subobservables = {lbl: [] for (lbl, _sc) in naive}
    est_exec = max(1, len(subcircuits)) * max(1, int(num_samples or 1))
    return PartitionPlan(
        kind="qiskit_addon",
        subcircuits=list(subcircuits.values()),
        reconstruction={
            "subexperiments": {},
            "coefficients": {},
            "subobservables": subobservables,
            "meta": {"approx_only": True, "fallback": reason},
        },
        est_executions=int(est_exec),
        k_wire=0,
        k_gate=0,
    )


class QiskitAddonCutStrategy(CutStrategy):
    name = "qiskit_addon"

    def __init__(
        self,
        num_samples: int = 200,
        seed: Optional[int] = None,
        max_gamma: float = 1024.0,
        max_backjumps: Optional[int] = 10000,
    ):
        self.num_samples = int(num_samples)
        self.seed = seed
        self.max_gamma = float(max_gamma)
        self.max_backjumps = max_backjumps

    def _import_addon(self):
        import qiskit_addon_cutting as cutting
        return cutting

    def analyze(self, circuit: QuantumCircuit, constraints: CutConstraints, context: Dict[str, Any]) -> CutAnalysis:
        t0 = time.perf_counter()
        circ = _strip_classical_bits(circuit)
        max_local = int((context or {}).get("max_local_qubits", 0) or 0)

        if max_local and circ.num_qubits <= max_local:
            return CutAnalysis(
                feasible=True,
                reason="fits_without_cutting",
                est_executions=1,
                est_quality_delta=0.0,
                est_search_time_s=time.perf_counter() - t0,
            )

        if constraints.max_cuts <= 0:
            return CutAnalysis(feasible=False, reason="max_cuts=0", est_search_time_s=time.perf_counter() - t0)

        # In analytic/planner mode, do not call find_cuts at all
        if (_QDC_CUT_FAST_PARTITION or (context or {}).get("approx_only", False)) and max_local > 0:
            target = getattr(constraints, "target_labels", None)
            if target is not None and int(target) >= 2:
                n_parts = max(2, min(int(target), int(circ.num_qubits)))
                est_exec = max(1, n_parts * max(1, int(self.num_samples or 1)))
                return CutAnalysis(
                    feasible=True,
                    reason="approx_target_labels_feasible",
                    est_executions=int(est_exec),
                    est_quality_delta=None,
                    est_search_time_s=time.perf_counter() - t0,
                )
            if circ.num_qubits > max_local:
                n_parts = math.ceil(circ.num_qubits / max_local)
                est_exec = max(1, n_parts * max(1, int(self.num_samples or 1)))
                return CutAnalysis(
                    feasible=True,
                    reason="approx_chunk_feasible",
                    est_executions=int(est_exec),
                    est_quality_delta=None,
                    est_search_time_s=time.perf_counter() - t0,
                )
            return CutAnalysis(
                feasible=True,
                reason="approx_no_cut_needed",
                est_executions=1,
                est_quality_delta=0.0,
                est_search_time_s=time.perf_counter() - t0,
            )

        try:
            cutting = self._import_addon()
            OptimizationParameters = cutting.OptimizationParameters
            DeviceConstraints = cutting.DeviceConstraints
            find_cuts = cutting.find_cuts

            modes = []
            if constraints.allow_gate_cuts or constraints.allow_wire_cuts:
                modes.append(("both", True, True))
            if constraints.allow_gate_cuts:
                modes.append(("gate_only", True, False))
            if constraints.allow_wire_cuts:
                modes.append(("wire_only", False, True))

            best = None
            best_meta = None
            target = getattr(constraints, "target_labels", None)
            seed_tries = max(1, int(getattr(constraints, "seed_tries", 1) or 1))

            observables = (context or {}).get("observable", None)
            if observables is not None:
                observables = [_to_paulilist(o) for o in observables]
            if observables is None:
                observables = [PauliList(["Z" + "I" * (circ.num_qubits - 1)])]

            for _name, gate_lo, wire_lo in modes:
                for k in range(seed_tries):
                    opt = OptimizationParameters(
                        seed=(None if self.seed is None else int(self.seed) + k),
                        max_gamma=self.max_gamma,
                        max_backjumps=self.max_backjumps,
                        gate_lo=gate_lo,
                        wire_lo=wire_lo,
                    )
                    dev = DeviceConstraints(qubits_per_subcircuit=int(max_local) if max_local else int(circ.num_qubits))
                    cut_circ, meta = find_cuts(circ, opt, dev)
                    cuts = meta.get("cuts", [])
                    if len(cuts) > constraints.max_cuts:
                        continue
                    overhead = float(meta.get("sampling_overhead", math.inf))

                    num_labels = None
                    if target is not None:
                        try:
                            tmp_final = cutting.cut_wires(cut_circ)
                            tmp_obs = cutting.expand_observables(observables, cut_circ, tmp_final)
                            tmp_prob = cutting.partition_problem(tmp_final, partition_labels=None, observables=tmp_obs)
                            num_labels = len(tmp_prob.subcircuits)
                        except Exception:
                            num_labels = None

                    score = (abs(int(num_labels) - int(target)), overhead) if (target is not None and num_labels is not None) else (0, overhead)
                    if best is None or score < best:
                        best = score
                        best_meta = meta

            if best_meta is None:
                return CutAnalysis(feasible=False, reason="no_cut_solution_within_constraints", est_search_time_s=time.perf_counter() - t0)

            overhead = float(best_meta.get("sampling_overhead", math.inf))
            est_exec = int(max(1, round(overhead))) if math.isfinite(overhead) else None
            return CutAnalysis(
                feasible=True,
                reason="cut_feasible",
                est_executions=est_exec,
                est_quality_delta=None,
                est_search_time_s=time.perf_counter() - t0,
            )
        except Exception as e:
            return CutAnalysis(feasible=False, reason=f"analyze_error:{repr(e)}", est_search_time_s=time.perf_counter() - t0)

    def partition(self, circuit: QuantumCircuit, constraints: CutConstraints, context: Dict[str, Any]) -> PartitionPlan:
        circ = _strip_classical_bits(circuit)
        ctx = context or {}
        max_local = int(ctx.get("max_local_qubits", circ.num_qubits) or circ.num_qubits)
        observables = _coerce_observables(ctx.get("observable", None), circ.num_qubits)

        # fast/approximate planning must bypass find_cuts entirely
        if (_QDC_CUT_FAST_PARTITION or ctx.get("approx_only", False)) and _QDC_CUT_FALLBACK_PARTITION and max_local > 0:
            target = getattr(constraints, "target_labels", None)
            if (target is not None and int(target) >= 2) or circ.num_qubits > max_local:
                reason = "target_labels" if (target is not None and int(target) >= 2) else "naive_chunk"
                return _fast_partition_plan(
                    circ,
                    max_local,
                    num_samples=self.num_samples,
                    reason=reason,
                    target_labels=target,
                )
            return PartitionPlan(
                kind="qiskit_addon",
                subcircuits=[circ],
                reconstruction={
                    "subexperiments": {},
                    "coefficients": {},
                    "subobservables": {0: []},
                    "meta": {"approx_only": True, "fallback": "single_chunk"},
                },
                est_executions=max(1, int(self.num_samples or 1)),
                k_wire=0,
                k_gate=0,
            )

        try:
            cutting = self._import_addon()
            OptimizationParameters = cutting.OptimizationParameters
            DeviceConstraints = cutting.DeviceConstraints
            find_cuts = cutting.find_cuts

            modes = []
            if constraints.allow_gate_cuts or constraints.allow_wire_cuts:
                modes.append(("both", True, True))
            if constraints.allow_gate_cuts:
                modes.append(("gate_only", True, False))
            if constraints.allow_wire_cuts:
                modes.append(("wire_only", False, True))

            best = None
            best_circ = None
            best_meta = None
            target = getattr(constraints, "target_labels", None)
            seed_tries = max(1, int(getattr(constraints, "seed_tries", 1) or 1))

            for _name, gate_lo, wire_lo in modes:
                for k in range(seed_tries):
                    opt = OptimizationParameters(
                        seed=(None if self.seed is None else int(self.seed) + k),
                        max_gamma=self.max_gamma,
                        max_backjumps=self.max_backjumps,
                        gate_lo=gate_lo,
                        wire_lo=wire_lo,
                    )
                    dev = DeviceConstraints(qubits_per_subcircuit=int(max_local))
                    cut_circ, meta = find_cuts(circ, opt, dev)
                    cuts = meta.get("cuts", [])
                    if len(cuts) > constraints.max_cuts:
                        continue
                    overhead = float(meta.get("sampling_overhead", math.inf))
                    num_labels = None
                    if target is not None:
                        try:
                            tmp_final = cutting.cut_wires(cut_circ)
                            tmp_obs = cutting.expand_observables(observables, cut_circ, tmp_final)
                            tmp_prob = cutting.partition_problem(tmp_final, partition_labels=None, observables=tmp_obs)
                            num_labels = len(tmp_prob.subcircuits)
                        except Exception:
                            num_labels = None

                    score = (abs(int(num_labels) - int(target)), overhead) if (target is not None and num_labels is not None) else (0, overhead)
                    if best is None or score < best:
                        best = score
                        best_circ = cut_circ
                        best_meta = meta

            if best_circ is None:
                raise ValueError("No cutting solution found within constraints (max_cuts / allowed cut types).")

            final_circ = cutting.cut_wires(best_circ)
            final_obs = cutting.expand_observables(observables, best_circ, final_circ)

            prob = cutting.partition_problem(final_circ, partition_labels=None, observables=final_obs)
            subcircuits = prob.subcircuits
            subobservables = prob.subobservables

            if (not subcircuits) and _QDC_CUT_FALLBACK_PARTITION and max_local > 0 and getattr(final_circ, "num_qubits", 0) > max_local:
                naive = _naive_chunk_partition(final_circ, max_local)
                subcircuits = {lbl: sc for (lbl, sc) in naive}
                subobservables = {lbl: [] for (lbl, _sc) in naive}

            subexperiments, coefficients = cutting.generate_cutting_experiments(
                subcircuits, subobservables, num_samples=self.num_samples
            )
            if isinstance(subexperiments, dict):
                est_exec = sum(len(v) for v in subexperiments.values())
            else:
                est_exec = len(subexperiments)

            k_wire = 0
            k_gate = 0
            for ctype, _cid in (best_meta.get("cuts", []) if isinstance(best_meta, dict) else []):
                ctype_s = str(ctype).lower()
                if "wire" in ctype_s:
                    k_wire += 1
                elif "gate" in ctype_s:
                    k_gate += 1

            return PartitionPlan(
                kind="qiskit_addon",
                subcircuits=list(subcircuits.values()),
                reconstruction={
                    "subexperiments": subexperiments,
                    "coefficients": coefficients,
                    "subobservables": subobservables,
                    "meta": best_meta,
                },
                est_executions=int(est_exec),
                k_wire=int(k_wire),
                k_gate=int(k_gate),
            )
        except Exception as e:
            return PartitionPlan(
                kind="qiskit_addon",
                subcircuits=[],
                reconstruction={"error": repr(e)},
                est_executions=0,
                k_wire=0,
                k_gate=0,
            )