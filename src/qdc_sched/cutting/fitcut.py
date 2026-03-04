
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

from qiskit import QuantumCircuit
from qiskit import QuantumCircuit

import math


def _dummy_partition(circuit: QuantumCircuit, max_local_qubits: int, target_labels: int):
    """Lightweight placeholder partition for analytic-only experiments.

    Enable with QDC_C_FALLBACK_DUMMY=1 to exercise Plan C logic when the
    real cutter fails or times out.
    """
    w = int(getattr(circuit, 'num_qubits', 0) or 0)
    if w <= 0:
        return []
    max_local_qubits = max(1, int(max_local_qubits))
    k = max(int(target_labels), int(math.ceil(w / float(max_local_qubits))))
    subcircs = []
    remaining = w
    for i in range(k):
        n = min(max_local_qubits, remaining)
        if n <= 0:
            break
        subcircs.append(QuantumCircuit(n, name=f"dummy_part_{i}"))
        remaining -= n
    if remaining > 0:
        subcircs.append(QuantumCircuit(min(max_local_qubits, remaining), name=f"dummy_part_{len(subcircs)}"))
    return subcircs


from .base import CutStrategy, CutConstraints, CutAnalysis, PartitionPlan
from .qiskit_addon import QiskitAddonCutStrategy

@dataclass
class FitCutSearchConfig:
    """Heuristic search controls for FitCut-style selection.

    This does NOT implement the full FitCut paper yet (graph partitioner),
    but it does implement the key scheduler-facing behavior:
    - explore multiple cutting solutions (wire/gate/both)
    - explore multiple target label counts and random seeds
    - choose the solution that minimizes an estimated makespan proxy
    """
    max_candidates: int = 12
    # penalty strength for additional cuts
    cut_penalty: float = 0.05
    sampling_weight: float = 0.02
    labels_weight: float = 0.02
    recon_base_s: float = 0.02
    recon_per_exec_s: float = 0.001

class FitCutCutStrategy(CutStrategy):
    name: str = "fitcut_v1"

    def __init__(self, base: Optional[CutStrategy] = None, cfg: Optional[FitCutSearchConfig] = None):
        # We use the Qiskit add-on strategy as the engine to produce cutting plans.
        self.base = base if base is not None else QiskitAddonCutStrategy()
        self.cfg = cfg if cfg is not None else FitCutSearchConfig()

    def _score_plan(self, plan: PartitionPlan, constraints: CutConstraints, context: Dict[str, Any]) -> float:
        n_labels = None
        try:
            if isinstance(plan.reconstruction, dict) and "subexperiments" in plan.reconstruction:
                n_labels = len(plan.reconstruction["subexperiments"])
        except Exception:
            n_labels = None
        if n_labels is None:
            n_labels = max(1, len(plan.subcircuits))

        # Parallelism available: number of candidate QPUs if provided
        qpu_candidates = context.get("qpu_candidates", None)
        n_qpus = len(qpu_candidates) if isinstance(qpu_candidates, list) else n_labels
        parallel = max(1, min(n_labels, n_qpus))

        sampling = float(plan.est_executions or 1.0)
        makespan_proxy = sampling / float(parallel)

        cuts = float(plan.k_wire + plan.k_gate)
        penalty = self.cfg.cut_penalty * cuts

        target = constraints.target_labels
        if target is not None:
            penalty += 0.02 * abs(float(n_labels) - float(target))

        # Add sampling and labels penalties (FitCut-style tradeoff knobs)
        penalty += self.cfg.sampling_weight * sampling
        penalty += self.cfg.labels_weight * float(n_labels)

        # Estimated reconstruction time (seconds) - proxy
        recon_s = float(self.cfg.recon_base_s) + float(self.cfg.recon_per_exec_s) * sampling
        penalty += recon_s

        return float(makespan_proxy + penalty)

    def analyze(self, circuit: QuantumCircuit, constraints: CutConstraints, context: Dict[str, Any]) -> CutAnalysis:
        # Explore multiple target label counts / seeds by delegating to base strategy.
        import time
        t0 = time.perf_counter()
        best: Optional[Tuple[float, CutAnalysis]] = None

        # Determine candidate targets
        cap = int(constraints.target_labels_cap or 4)
        targets: List[Optional[int]] = []
        if constraints.target_labels is not None:
            targets.append(int(constraints.target_labels))
        else:
            targets.extend(list(range(2, max(2, cap) + 1)))
            
        tries = max(1, int(constraints.seed_tries or 1))
        modes = []
        if constraints.allow_wire_cuts:
            modes.append(("wire", True, False))
        if constraints.allow_gate_cuts:
            modes.append(("gate", False, True))
        if constraints.allow_wire_cuts and constraints.allow_gate_cuts:
            modes.append(("both", True, True))
        if not modes:
            modes = [("none", False, False)]

        n_examined = 0
        for seed in range(tries):
            for (_mname, aw, ag) in modes:
                for tgt in targets:
                    cc = CutConstraints(
                        max_cuts=constraints.max_cuts,
                        allow_wire_cuts=aw,
                        allow_gate_cuts=ag,
                        reconstruction_target=constraints.reconstruction_target,
                        target_labels=tgt,
                        target_labels_cap=constraints.target_labels_cap,
                        seed_tries=1,
                    )
                    analysis = self.base.analyze(circuit, cc, context)
                    n_examined += 1
                    if not analysis.feasible:
                        continue
                    score = float(analysis.est_executions or 1) + 0.05 * (0 if cc.target_labels is None else abs((analysis.est_executions or 1) - (cc.target_labels or 1)))
                    if best is None or score < best[0]:
                        best = (score, analysis)
                    if n_examined >= self.cfg.max_candidates:
                        break
                if n_examined >= self.cfg.max_candidates:
                    break
            if n_examined >= self.cfg.max_candidates:
                break

        if best is None:
            return CutAnalysis(feasible=False, reason="No feasible cutting plan found in FitCut search.", est_search_time_s=time.perf_counter() - t0)

        out = best[1]
        out.est_search_time_s = time.perf_counter() - t0
        out.reason = (out.reason or "") + f" [fitcut_search examined={n_examined}]"
        return out

    def partition(self, circuit: QuantumCircuit, constraints: CutConstraints, context: Dict[str, Any]) -> PartitionPlan:
        # Enumerate candidates by calling base.partition with different constraints, pick the best by our proxy score.
        import time
        t0 = time.perf_counter()
        best_plan: Optional[PartitionPlan] = None
        best_score: Optional[float] = None
        best_meta: Dict[str, Any] = {}

        cap = int(constraints.target_labels_cap or 4)
        targets: List[Optional[int]] = []
        if constraints.target_labels is not None:
            targets.append(int(constraints.target_labels))
        else:
            targets.extend(list(range(2, max(2, cap) + 1)))

        tries = max(1, int(constraints.seed_tries or 1))
        modes = []
        if constraints.allow_wire_cuts:
            modes.append(("wire", True, False))
        if constraints.allow_gate_cuts:
            modes.append(("gate", False, True))
        if constraints.allow_wire_cuts and constraints.allow_gate_cuts:
            modes.append(("both", True, True))
        if not modes:
            modes = [("none", False, False)]

        n_examined = 0
        last_err: Optional[str] = None
        for seed in range(tries):
            for (mname, aw, ag) in modes:
                for tgt in targets:
                    cc = CutConstraints(
                        max_cuts=constraints.max_cuts,
                        allow_wire_cuts=aw,
                        allow_gate_cuts=ag,
                        reconstruction_target=constraints.reconstruction_target,
                        target_labels=tgt,
                        target_labels_cap=constraints.target_labels_cap,
                        seed_tries=1,
                    )
                    try:
                        plan = self.base.partition(circuit, cc, context)
                    except Exception as e:
                        last_err = repr(e)
                        n_examined += 1
                        if n_examined >= self.cfg.max_candidates:
                            break
                        continue
                    n_examined += 1
                    sc = self._score_plan(plan, cc, context)
                    if best_score is None or sc < best_score:
                        best_score = sc
                        best_plan = plan
                        best_meta = {"mode": mname, "seed": seed, "target_labels": tgt, "score": sc}
                    if n_examined >= self.cfg.max_candidates:
                        break
                if n_examined >= self.cfg.max_candidates:
                    break
            if n_examined >= self.cfg.max_candidates:
                break

        if best_plan is None:

            # Optional: fallback to a lightweight dummy partition in analytic experiments.

            # Enable with QDC_C_FALLBACK_DUMMY=1.

            if os.getenv("QDC_C_FALLBACK_DUMMY", "0") == "1":

                try:

                    ctx_max_local = int((context or {}).get("max_local_qubits", 0) or 0)

                except Exception:

                    ctx_max_local = 0

                if ctx_max_local > 0:

                    subcircs = _dummy_partition(circuit, ctx_max_local, int(getattr(cc, 'target_labels', 2) or 2))

                    if subcircs:

                        return PartitionPlan(

                            kind="DUMMY_PARTITION",

                            subcircuits=subcircs,

                            est_executions=1.0,

                            meta={"fallback": True, "reason": "fitcut_failed", "max_local_qubits": ctx_max_local},

                        )
            return PartitionPlan(
                kind="FITCUT_FAILED",
                subcircuits=[],
                reconstruction={"error": f"FitCut search failed (examined={n_examined}). last_err={last_err}"},
                est_executions=10**9,
                k_wire=0,
                k_gate=0,
            )

        if isinstance(best_plan.reconstruction, dict):
            meta = best_plan.reconstruction.get("meta", {})
            if not isinstance(meta, dict):
                meta = {"base_meta": meta}
            meta.update({"fitcut": {"examined": n_examined, "best": best_meta, "search_time_s": time.perf_counter() - t0}})
            best_plan.reconstruction["meta"] = meta

        return best_plan