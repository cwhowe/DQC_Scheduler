from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Literal

TaskType = Literal["counts", "expectation"]

@dataclass(frozen=True)
class JobConstraints:
    slo_s: Optional[float] = None
    min_fidelity: Optional[float] = None
    allow_cutting: bool = True
    force_cutting: bool = False
    allow_multi_qpu: bool = True
    max_cuts: int = 3
    # Communication overhead model, needs updating.
    comm_overhead_s: float = 0.25

@dataclass
class Job:
    job_id: str
    circuit: Any  # qiskit.QuantumCircuit
    task_type: TaskType = "counts"
    observables: Optional[Any] = None
    shots: int = 2000
    submit_time_s: float = 0.0
    priority: int = 0
    constraints: JobConstraints = field(default_factory=JobConstraints)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CircuitProfile:
    width: int
    depth: int
    twoq_count: int
    oneq_count: int
    meas_count: int
    interaction_density: float
    # ranked candidates: list of (qpu_id, score, predicted_latency_s, fidelity_proxy)
    ranked_qpus: List[Tuple[str, float, float, float]] = field(default_factory=list)
    # classification: "bad" | "neutral" | "good"
    cut_suitability: str = "neutral"
    # any extra features for ML later, still unsure what we want to add.
    features: Dict[str, float] = field(default_factory=dict)

@dataclass
class Plan:
    kind: Literal["A_NO_CUT_SINGLE", "B_CUT_SINGLE_SEQ", "C_CUT_MULTI_QPU", "D_WAIT"]
    qpu_id: Optional[str] = None
    physical_qubits: Optional[List[int]] = None
    predicted_total_time_s: float = float("inf")
    predicted_fidelity_proxy: float = 0.0
    score: float = float("inf")
    details: Dict[str, Any] = field(default_factory=dict)
    cut_metadata: Optional[Dict[str, Any]] = None
    partitions: Optional[List[Any]] = None

@dataclass
class RunToggles:
    compute_estimated_fidelity: bool = False
    compute_expectation: bool = False
    simulate_only: bool = False


# -------------------------
# Task abstraction
# -------------------------
TaskKind = Literal["quantum", "communication", "reconstruction"]

@dataclass
class Task:
    task_id: str
    job_id: str
    kind: TaskKind
    start_s: float
    end_s: float
    qpu_id: Optional[str] = None          # None => classical/host task
    qubits: Optional[List[int]] = None    # physical qubits (sorted)
    label: Optional[int] = None           # subcircuit label (cutting)
    depends_on: Optional[List[str]] = None   # optional DAG edges (task_ids)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskGraph:
    job_id: str
    tasks: List[Task] = field(default_factory=list)

def task_to_dict(t: Task) -> Dict[str, Any]:
    return {
        "task_id": t.task_id,
        "job_id": t.job_id,
        "kind": t.kind,
        "start_s": t.start_s,
        "end_s": t.end_s,
        "qpu_id": t.qpu_id,
        "qubits": t.qubits,
        "label": t.label,
        "depends_on": t.depends_on,
        "metadata": t.metadata or {},
    }


def taskgraph_to_dict(g: TaskGraph) -> Dict[str, Any]:
    return {"job_id": g.job_id, "tasks": [task_to_dict(t) for t in (g.tasks or [])]}