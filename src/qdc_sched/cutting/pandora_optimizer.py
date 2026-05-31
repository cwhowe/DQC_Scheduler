from __future__ import annotations

from typing import Any, Dict, Optional

from qiskit import QuantumCircuit

from .base import CutAnalysis, CutConstraints, CutStrategy, PartitionPlan
from .fitcut import FitCutCutStrategy
from .pandora_bridge import PandoraBridge


class PandoraOptimizedCutStrategy(CutStrategy):
    """Pre-optimize a circuit through Pandora SQL rewrites, then delegate to FitCutCutStrategy.

    If Pandora is unavailable or optimization fails, the original circuit is passed
    unchanged to the fallback strategy.
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

    def _optimized(self, circuit: QuantumCircuit) -> QuantumCircuit:
        gates_before = circuit.size()
        depth_before = circuit.depth()
        opt = self.bridge.optimize(circuit)
        result = opt if opt is not None else circuit
        self.gate_count_log.append((
            gates_before, result.size(),
            depth_before, result.depth(),
        ))
        return result

    def analyze(
        self,
        circuit: QuantumCircuit,
        constraints: CutConstraints,
        context: Dict[str, Any],
    ) -> CutAnalysis:
        return self.fallback.analyze(self._optimized(circuit), constraints, context)

    def partition(
        self,
        circuit: QuantumCircuit,
        constraints: CutConstraints,
        context: Dict[str, Any],
    ) -> PartitionPlan:
        return self.fallback.partition(self._optimized(circuit), constraints, context)
