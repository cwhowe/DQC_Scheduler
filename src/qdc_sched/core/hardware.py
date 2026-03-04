from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import networkx as nx

@dataclass(frozen=True)
class HardwareProfile:
    qpu_id: str
    num_qubits: int
    coupling_graph: nx.Graph
    # Per-edge 2Q error
    twoq_error: Dict[Tuple[int,int], float] = field(default_factory=dict)
    # Per-qubit readout error
    ro_error: Dict[int, float] = field(default_factory=dict)
    # Timing model (seconds)
    t_1q: float = 35e-9
    t_2q: float = 250e-9
    t_meas: float = 1.0e-6
    t_reset: float = 1.0e-6
    # Queue delay model (seconds)
    base_queue_delay_s: float = 0.0
    # Simple timing model (used for predicted runtime; seconds)
    oneq_gate_time_s: float = 35e-9
    twoq_gate_time_s: float = 300e-9
    meas_time_s: float = 1_000e-9
    shot_overhead_s: float = 0.0
    # Optional scalar error rates (used for quality proxy)
    oneq_error: float = 1e-3
    twoq_error: float = 2e-2
    readout_error: float = 2e-2

@dataclass
class Reservation:
    job_id: str
    qubits: Set[int]
    start_s: float
    end_s: float

class QPUState:
    """Tracks spatial concurrency on disjoint qubit sets via reservations."""
    def __init__(self, profile: HardwareProfile):
        self.profile = profile
        self._reservations: List[Reservation] = []

    # Can probably remove this: old reservation defenition.
    @property
    def reservations(self) -> List[Reservation]:
        return self._reservations

    def release_completed(self, now_s: float) -> None:
        now_s = float(now_s)
        self._reservations = [r for r in self._reservations if float(r.end_s) > now_s]

    def reserved_qubits(self, now_s: float) -> Set[int]:
        now_s = float(now_s)
        used: Set[int] = set()
        for r in self._reservations:
            if float(r.start_s) <= now_s < float(r.end_s):
                used |= set(r.qubits)
        return used

    def is_free_set(self, qubits: Set[int], now_s: float) -> bool:
        return len(qubits & self.reserved_qubits(now_s)) == 0

    def reserve(self, job_id: str, qubits: Set[int], start_s: float, duration_s: float) -> Reservation:
        start_s = float(start_s)
        duration_s = float(duration_s)
        end_s = start_s + duration_s
        r = Reservation(job_id=job_id, qubits=set(qubits), start_s=start_s, end_s=end_s)
        self._reservations.append(r)
        return r

    def find_free_connected_subgraph(self, k: int, now_s: float) -> Optional[Set[int]]:
        """Greedy search for a connected set of k free qubits in coupling graph."""
        k = int(k)
        now_s = float(now_s)
        used = self.reserved_qubits(now_s)
        free_nodes = [n for n in self.profile.coupling_graph.nodes if n not in used]
        if len(free_nodes) < k:
            return None

        for seed in sorted(free_nodes):
            if seed in used:
                continue
            sub = set([seed])
            frontier = [seed]
            while frontier and len(sub) < k:
                u = frontier.pop()
                for v in self.profile.coupling_graph.neighbors(u):
                    if v in used or v in sub:
                        continue
                    sub.add(v)
                    frontier.append(v)
                    if len(sub) >= k:
                        break
            if len(sub) == k:
                return sub
        return None

    def max_connected_free_qubits(self, now_s: float) -> int:
        """Return the size of the largest connected component of currently-free qubits."""
        now_s = float(now_s)
        used = self.reserved_qubits(now_s)
        G = self.profile.coupling_graph
        free_nodes = [n for n in G.nodes if n not in used]
        if not free_nodes:
            return 0
        H = G.subgraph(free_nodes)
        import networkx as nx
        return max((len(c) for c in nx.connected_components(H)), default=0)

    def estimate_wait_s(self, needed_qubits: int, now_s: float) -> float:
        """Coarse wait-time estimate until a connected free component of size >= needed_qubits exists."""
        needed = int(needed_qubits)
        now_s = float(now_s)

        if needed <= 0:
            return 0.0

        # Always clear completed reservations before planning.
        self.release_completed(now_s)

        if self.max_connected_free_qubits(now_s) >= needed:
            return 0.0

        ends: List[float] = []
        for r in self._reservations:
            try:
                t = float(r.end_s)
                if t >= now_s:
                    ends.append(t)
            except Exception:
                continue

        for t in sorted(set(ends)):
            # simulate time t after some reservations end
            if self.max_connected_free_qubits(t) >= needed:
                return max(0.0, t - now_s)

        return float("inf")

