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

    def _optimized(self, circuit: QuantumCircuit) -> QuantumCircuit:
        opt = self.bridge.optimize(circuit)
        return opt if opt is not None else circuit

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
