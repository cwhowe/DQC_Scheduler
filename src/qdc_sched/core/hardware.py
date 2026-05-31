from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import networkx as nx

@dataclass(frozen=True)
class HardwareProfile:
    """Hardware timing and error model for a single QPU.

    Timing constants are calibrated to IBM superconducting (transmon) devices,
    specifically the Falcon r5 / Eagle r3 / Heron r1 families (2022-2024).
    All values represent typical/median calibration data; per-qubit variation
    of ±20-50% is common on real devices.

    Gate times (seconds):
        t_1q / oneq_gate_time_s : SX, RZ, X  → 30-50 ns   (default 35 ns)
        t_2q / twoq_gate_time_s : CX, ECR    → 200-500 ns (default 300 ns)
        t_meas / meas_time_s    : readout     → 500-1500 ns (default 1000 ns)
        t_reset                 : active reset → ~1 µs      (default 1 µs)
        shot_overhead_s         : classical control reset + state prep + data
                                  movement between shots → 50-500 µs
                                  (default 250 µs, mid-range for IBM Falcon/Eagle)

    Error rates (dimensionless):
        oneq_error   : depolarizing error per 1Q gate → ~1e-3 (default)
        twoq_error   : depolarizing error per 2Q gate → ~2e-2 (default)
        readout_error: SPAM error per qubit           → ~2e-2 (default)

    References:
        Krantz et al., Rev. Mod. Phys. 91, 025005 (2019) — §III gate times
        IBM Quantum backend calibration (ibm_kyiv, ibm_brisbane, Jan 2024)
        Jurcevic et al., Quantum Sci. Technol. 6, 025020 (2021) — Eagle specs
    """
    qpu_id: str
    num_qubits: int
    coupling_graph: nx.Graph

    # error maps (optional; per-edge/per-qubit overrides scalar fallbacks)
    twoq_error_map: Dict[Tuple[int, int], float] = field(default_factory=dict)
    ro_error_map: Dict[int, float] = field(default_factory=dict)

    # Timing model (seconds) — used in profiler.rank_qpus scoring
    t_1q: float = 35e-9
    t_2q: float = 300e-9
    t_meas: float = 1.0e-6
    t_reset: float = 1.0e-6

    # Queue delay model (seconds)
    base_queue_delay_s: float = 0.0

    # Gate times for predict_exec_time_s (seconds)
    # Separate from t_1q/t_2q so the ranking model and timing model can be
    # tuned independently without breaking the profiler scoring.
    oneq_gate_time_s: float = 35e-9
    twoq_gate_time_s: float = 300e-9
    meas_time_s: float = 1_000e-9

    # Shot overhead: classical control reset between shots.
    # 250 µs = conservative mid-range for IBM Falcon/Eagle (50-500 µs range).
    # Set to 0.0 for ideal/noiseless simulation comparisons.
    # Override per-QPU or via env QDC_SHOT_OVERHEAD_S.
    shot_overhead_s: float = 250e-6

    # T-gate time multiplier relative to other 1Q gates.
    # Default 1.0 = T costs the same as SX/RZ (physical qubit regime).
    # Set to ~50.0 to model fault-tolerant magic-state distillation overhead,
    # where T gates dominate circuit cost vs Clifford gates.
    # Override via env QDC_T_GATE_OVERHEAD or per-QPU in RunToggles.
    t_gate_overhead: float = 1.0

    # Scalar fallback error rates (used for quality proxy when per-qubit maps absent)
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
            sub = {seed}
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
        return max((len(c) for c in nx.connected_components(H)), default=0)

    def estimate_wait_s(self, needed_qubits: int, now_s: float) -> float:
        """Coarse wait-time estimate until a connected free component of size >= needed_qubits exists."""
        needed = int(needed_qubits)
        now_s = float(now_s)

        if needed <= 0:
            return 0.0

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
            if self.max_connected_free_qubits(t) >= needed:
                return max(0.0, t - now_s)

        return float("inf")