from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from qiskit import QuantumCircuit

@dataclass
class CutConstraints:
    max_cuts: int = 3
    allow_wire_cuts: bool = True
    allow_gate_cuts: bool = True
    # Default: expectation-only reconstruction for cutting plans
    reconstruction_target: str = "expectation"
    # When set, cutting will prefer producing target_labels partitions/labels.
    target_labels: Optional[int] = None
    # If target_labels is None, executor may set it to min(num_candidate_qpus, target_labels_cap)
    target_labels_cap: int = 4
    # Explore multiple seeds when searching for cuts (to find alternative label counts)
    seed_tries: int = 4

@dataclass
class PartitionPlan:
    kind: str
    subcircuits: List[QuantumCircuit]
    reconstruction: Dict[str, Any]
    est_executions: int
    k_wire: int = 0
    k_gate: int = 0

@dataclass
class CutAnalysis:
    feasible: bool
    reason: Optional[str] = None
    est_executions: Optional[int] = None
    est_quality_delta: Optional[float] = None
    est_search_time_s: Optional[float] = None

class CutStrategy:
    name: str = "base"

    def analyze(self, circuit: QuantumCircuit, constraints: CutConstraints, context: Dict[str, Any]) -> CutAnalysis:
        raise NotImplementedError

    def partition(self, circuit: QuantumCircuit, constraints: CutConstraints, context: Dict[str, Any]) -> PartitionPlan:
        raise NotImplementedError

    def estimate_cost(self, plan: PartitionPlan, assignment: Dict[str, Any]) -> Dict[str, Any]:
        return {}

def _max_subcircuit_width(partition) -> int:
    """Return max qubit count among partition.subcircuits (0 if missing)."""
    subs = getattr(partition, "subcircuits", None) or []
    try:
        return max(int(getattr(sc, "num_qubits", 0)) for sc in subs) if subs else 0
    except Exception:
        return 0
