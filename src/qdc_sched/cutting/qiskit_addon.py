from __future__ import annotations
import os
from typing import Any, Dict, Optional, Tuple, List
import math
import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, PauliList

from .base import CutStrategy, CutConstraints, PartitionPlan, CutAnalysis


# -------------------------
# Fast/approx partition toggles (demo / planning)
# -------------------------
_QDC_CUT_FAST_PARTITION = os.environ.get("QDC_CUT_FAST_PARTITION", "0") not in ("0", "", "false", "False")
_QDC_CUT_FALLBACK_PARTITION = os.environ.get("QDC_CUT_FALLBACK_PARTITION", "1") not in ("0", "", "false", "False")

def _naive_chunk_partition(circuit, max_subq: int):
    """Fallback: split logical qubits into contiguous chunks of size <= max_subq.
    This is **only** intended for analytic planning/demo when full QPD decomposition is too expensive.
    Returns list of (label, subcircuit) for PartitionPlan.subcircuits.
    """
    from qiskit.circuit import QuantumCircuit
    n = circuit.num_qubits
    if max_subq <= 0 or max_subq >= n:
        return [(0, circuit)]
    chunks = []
    label = 0
    for start in range(0, n, max_subq):
        qs = list(range(start, min(n, start + max_subq)))
        sub = QuantumCircuit(len(qs), name=f"sub_{label}")
        # Rebuild a *very* rough circuit: copy over 1q/2q gates that stay within the chunk, ignore cross-chunk ops.
        # For analytic timing, width/depth/2q-count approximations are what matter most.
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


def _coerce_observables(obs, n_qubits: int) -> PauliList:
    """Coerce various observable container forms into a single PauliList.

    qiskit_addon_cutting expects Pauli/PauliList-like objects (with .num_qubits).
    We standardize to PauliList, where each entry is one Pauli string.
    """
    if obs is None:
        return PauliList(["Z" + "I" * (n_qubits - 1)])
    if isinstance(obs, list) or isinstance(obs, tuple):
        pls = []
        for o in obs:
            pl = _to_paulilist(o)
            pls.extend(list(pl))
        return PauliList(pls)
    return _to_paulilist(obs)
def _strip_classical_bits(circ: QuantumCircuit) -> QuantumCircuit:
    """Cutting add-on requires no classical bits; expectation jobs should not contain measurements."""
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

class QiskitAddonCutStrategy(CutStrategy):
    """Baseline strategy using qiskit-addon-cutting (add-on v0.10.x).

    Core API used:
      - find_cuts(circuit, OptimizationParameters, DeviceConstraints)
      - cut_wires, expand_observables
      - partition_problem
      - generate_cutting_experiments
      - reconstruct_expectation_values
    """

    name = "qiskit_addon"

    def __init__(self, num_samples: int = 200, seed: Optional[int] = None, max_gamma: float = 1024.0, max_backjumps: Optional[int] = 10000):
        self.num_samples = int(num_samples)
        self.seed = seed
        self.max_gamma = float(max_gamma)
        self.max_backjumps = max_backjumps

    def _import_addon(self):
        import qiskit_addon_cutting as cutting
        return cutting


    def analyze(self, circuit: QuantumCircuit, constraints: CutConstraints, context: Dict[str, Any]) -> CutAnalysis:
        """Fast feasibility + overhead analysis for planner.

        Returns a CutAnalysis. Does *not* generate full subexperiments
        """
        t0 = time.perf_counter()
        circ = _strip_classical_bits(circuit)
        max_local = context.get("max_local_qubits", None)
        if max_local is not None and circ.num_qubits <= int(max_local):
            return CutAnalysis(feasible=True, reason="fits_without_cutting", est_executions=1, est_quality_delta=0.0, est_search_time_s=0.0)

        if constraints.max_cuts <= 0:
            return CutAnalysis(feasible=False, reason="max_cuts=0", est_search_time_s=0.0)

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
            max_local = int(context.get("max_local_qubits", 0) or 0)

            # minimal observable placeholder is fine for label count estimate; planner only needs feasibility + overhead
            observables = context.get("observable", None)
            if observables is not None:
                observables = [_to_paulilist(o) for o in observables]
            if observables is None:
                # Z gate on first qubit for compatibility
                observables = [PauliList(["Z" + "I" * (circ.num_qubits - 1)])]

            observables = [_to_paulilist(o) for o in observables]
            # If caller provided a list of SparsePauliOp/strings, coerce them for add-on APIs.

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

                    # Optional: estimate label count to match target_labels
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
            # Map overhead to a rough execution multiplier: higher overhead => more subexperiments
            est_exec = int(max(1, round(overhead))) if math.isfinite(overhead) else None
            return CutAnalysis(feasible=True, reason="cut_feasible", est_executions=est_exec, est_quality_delta=None, est_search_time_s=time.perf_counter() - t0)

        except Exception as e:
            return CutAnalysis(feasible=False, reason=f"analyze_error:{repr(e)}", est_search_time_s=time.perf_counter() - t0)

    def partition(self, circuit: QuantumCircuit, constraints: CutConstraints, context: Dict[str, Any]) -> PartitionPlan:
        """Build the full partition plan + subexperiments + reconstruction payload."""
        circ = _strip_classical_bits(circuit)
        max_local = int(context.get("max_local_qubits", circ.num_qubits) or circ.num_qubits)
        observables = _coerce_observables(context.get("observable", None), circ.num_qubits)

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

            # Fallback (fast/approx only): if the cutting addon couldn't produce any subcircuits,
            # create a simple contiguous-chunk partition so planning can still proceed.
            if (not subcircuits) and (_QDC_CUT_FAST_PARTITION or context.get("approx_only", False)) and _QDC_CUT_FALLBACK_PARTITION:
                max_subq = int(context.get("max_subcircuit_qubits", 0) or context.get("max_local_qubits", 0) or 0)
                if max_subq > 0 and getattr(final_circ, "num_qubits", 0) > max_subq:
                    naive = _naive_chunk_partition(final_circ, max_subq)
                    subcircuits = {lbl: sc for (lbl, sc) in naive}
                    subobservables = {lbl: [] for (lbl, _) in naive}


            if _QDC_CUT_FAST_PARTITION or context.get("approx_only", False):
                subexperiments, coefficients = {}, {}
            else:
                subexperiments, coefficients = cutting.generate_cutting_experiments(
                    subcircuits, subobservables, num_samples=self.num_samples
                )
            if isinstance(subexperiments, dict):
                est_exec = sum(len(v) for v in subexperiments.values())
                if est_exec == 0 and (_QDC_CUT_FAST_PARTITION or context.get("approx_only", False)):
                    # rough proxy: each subcircuit gets `num_samples` QPD samples (or 1 if unset)
                    ns = int(getattr(self, "num_samples", 1) or 1)
                    est_exec = max(1, len(subcircuits)) * ns
            else:
                est_exec = len(subexperiments)



            k_wire = 0
            k_gate = 0
            for ctype, _cid in (best_meta.get("cuts", []) if isinstance(best_meta, dict) else []):
                if "wire" in str(ctype).lower():
                    k_wire += 1
                elif "gate" in str(ctype).lower():
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