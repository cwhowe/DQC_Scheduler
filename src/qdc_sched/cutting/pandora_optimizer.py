from __future__ import annotations

from typing import Any, Dict, Optional

from qiskit import QuantumCircuit

from .base import CutAnalysis, CutConstraints, CutStrategy, PartitionPlan
from .fitcut import FitCutCutStrategy
from .pandora_bridge import PandoraBridge


class PandoraOptimizedCutStrategy(CutStrategy):
    """Pandora gate-cancellation + FitCut circuit partitioning.

    Strategy:
    1. FitCut the ORIGINAL circuit to establish the plan type (A/B/C) and
       number of subcircuits — so the planner sees the real interaction graph.
    2. Pandora-optimize the FULL circuit before re-cutting it.
    3. If the Pandora-optimized circuit produces at least as many subcircuits
       as the original, use the optimized plan (gate savings + correct plan type).
    4. If optimization collapsed the graph (fewer subcircuits), fall back to
       per-subcircuit optimization of the original plan — preserving plan type
       at the cost of gate savings that span cut boundaries.
    """

    name = "pandora_optimized"

    def __init__(
        self,
        bridge: PandoraBridge,
        fallback: Optional[CutStrategy] = None,
    ):
        self.bridge = bridge
        self.fallback = fallback if fallback is not None else FitCutCutStrategy()
        # Each entry: (gates_before, gates_after, depth_before, depth_after)
        self.gate_count_log: list = []

    def analyze(
        self,
        circuit: QuantumCircuit,
        constraints: CutConstraints,
        context: Dict[str, Any],
    ) -> CutAnalysis:
        return self.fallback.analyze(circuit, constraints, context)

    def partition(
        self,
        circuit: QuantumCircuit,
        constraints: CutConstraints,
        context: Dict[str, Any],
    ) -> PartitionPlan:
        # Step 1: reference plan on the original circuit (determines plan type).
        plan_orig = self.fallback.partition(circuit, constraints, context)
        n_orig = len(plan_orig.subcircuits)

        # Step 2: Pandora-optimize the full circuit.
        opt = self.bridge.optimize(circuit)
        opt_circuit = opt if opt is not None else circuit

        # Step 3: Re-cut the Pandora-optimized circuit.
        plan_opt = self.fallback.partition(opt_circuit, constraints, context)
        n_opt = len(plan_opt.subcircuits)

        if n_opt >= n_orig:
            # Optimization didn't shrink the partition — use the optimized plan.
            for orig_sub, opt_sub in zip(plan_orig.subcircuits, plan_opt.subcircuits):
                self.gate_count_log.append((
                    orig_sub.size(), opt_sub.size(),
                    orig_sub.depth(), opt_sub.depth(),
                ))
            return plan_opt

        # Step 4: Optimization collapsed the graph; fall back to per-subcircuit
        # Pandora so plan type is preserved even though cross-cut pairs are lost.
        optimized_subs = []
        for sub in plan_orig.subcircuits:
            gates_before = sub.size()
            depth_before = sub.depth()
            sub_opt = self.bridge.optimize(sub)
            result = sub_opt if sub_opt is not None else sub
            self.gate_count_log.append((
                gates_before, result.size(),
                depth_before, result.depth(),
            ))
            optimized_subs.append(result)
        plan_orig.subcircuits = optimized_subs
        return plan_orig
