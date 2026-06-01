from __future__ import annotations

from typing import Any, Dict, Optional

from qiskit import QuantumCircuit

from .base import CutAnalysis, CutConstraints, CutStrategy, PartitionPlan
from .fitcut import FitCutCutStrategy
from .pandora_bridge import PandoraBridge


class PandoraWidgetizerStrategy(CutStrategy):
    """Partition a circuit into disjoint subcircuits using Pandora's widgetize() generator.

    Each widget is a connected subgraph of the circuit DAG bounded by max_t T-gates
    and max_d depth. Falls back to FitCutCutStrategy when Pandora is unavailable or
    widgetization produces no subcircuits.
    """

    name = "pandora_widgetizer"

    def __init__(
        self,
        bridge: PandoraBridge,
        fallback: Optional[CutStrategy] = None,
        max_t: int = 10,
        max_d: int = 10,
        batch_size: int = 1000,
        add_gin_per_widget: bool = True,
    ):
        self.bridge = bridge
        self.fallback = fallback if fallback is not None else FitCutCutStrategy()
        self.max_t = max_t
        self.max_d = max_d
        self.batch_size = batch_size
        self.add_gin_per_widget = add_gin_per_widget

    def _get_subcircuits(self, circuit: QuantumCircuit):
        return self.bridge.widgetize(
            circuit,
            max_t=self.max_t,
            max_d=self.max_d,
            batch_size=self.batch_size,
            add_gin_per_widget=self.add_gin_per_widget,
        )

    def analyze(
        self,
        circuit: QuantumCircuit,
        constraints: CutConstraints,
        context: Dict[str, Any],
    ) -> CutAnalysis:
        if not self.bridge.available:
            return self.fallback.analyze(circuit, constraints, context)
        subcircuits = self._get_subcircuits(circuit)
        if not subcircuits:
            return self.fallback.analyze(circuit, constraints, context)
        return CutAnalysis(
            feasible=True,
            reason="pandora_widgetize",
            est_executions=len(subcircuits),
            est_quality_delta=None,
            est_search_time_s=0.0,
        )

    def partition(
        self,
        circuit: QuantumCircuit,
        constraints: CutConstraints,
        context: Dict[str, Any],
    ) -> PartitionPlan:
        if not self.bridge.available:
            return self.fallback.partition(circuit, constraints, context)
        subcircuits = self._get_subcircuits(circuit)
        if not subcircuits:
            import os
            if os.getenv("QDC_AER_DEBUG", "0") == "1":
                print(f"[WIDGETIZER_FALLBACK] widgetize() returned [] for "
                      f"{circuit.num_qubits}q circuit, falling back to FitCut")
            return self.fallback.partition(circuit, constraints, context)
        return PartitionPlan(
            kind="pandora_widgetizer",
            subcircuits=subcircuits,
            reconstruction={
                "meta": {
                    "pandora_widgetizer": True,
                    "n_widgets": len(subcircuits),
                    "max_t": self.max_t,
                    "max_d": self.max_d,
                }
            },
            est_executions=len(subcircuits),
            k_wire=0,
            k_gate=0,
        )
