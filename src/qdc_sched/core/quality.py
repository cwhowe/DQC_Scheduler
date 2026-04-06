from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .types import CircuitProfile

@dataclass
class QualityModel:
    """Provides a fast fidelity proxy and an optional expensive estimated fidelity (ideal vs noisy).

    The fidelity proxy uses an exponential decay model calibrated to typical
    IBM superconducting device error rates:

        F ≈ exp(-ε_2q * n_2q - ε_1q * n_1q - ε_ro * n_meas)

    where ε_Xq are the per-operation depolarizing error rates. This is a standard
    approximation for the total circuit fidelity under independent error channels
    (see Emerson et al., Science 317, 1893, 2007; Magesan et al., PRL 106, 2011).

    When a QPU's HardwareProfile is available, per-QPU error rates are used.
    Default fallback values match IBM Falcon/Eagle median calibration data (2024):
        ε_2q = 0.010  (1% per CX/ECR gate)
        ε_1q = 0.001  (0.1% per SX/RZ gate)
        ε_ro = 0.020  (2% per qubit readout, not included by default — SPAM-separated)
    """
    noise_models: Dict[str, NoiseModel]
    # Optional: QPU state registry for per-QPU error rate lookup
    qpu_profiles: Dict[str, Any] = field(default_factory=dict)

    def fidelity_proxy_from_profile(self, qpu_id: str, prof: CircuitProfile) -> float:
        """Fast fidelity proxy using exponential decay over gate counts.

        Uses per-QPU error rates from HardwareProfile when available, otherwise
        falls back to IBM Falcon/Eagle median values.
        """
        # Try to get per-QPU error rates from the registered profile
        qpu_prof = self.qpu_profiles.get(qpu_id)
        if qpu_prof is not None:
            p2 = float(getattr(qpu_prof, "twoq_error", 0.010))
            p1 = float(getattr(qpu_prof, "oneq_error", 0.001))
        else:
            # IBM Falcon/Eagle median calibration (Jan 2024):
            # CX/ECR: ~0.8-1.5% → use 1.0% as median
            # SX/RZ:  ~0.05-0.15% → use 0.1% as median
            p2 = 0.010
            p1 = 0.001

        fid = math.exp(-p2 * prof.twoq_count - p1 * prof.oneq_count)
        return max(0.0, min(1.0, fid))

    def estimated_fidelity_counts(self, qc: QuantumCircuit, qpu_id: str, shots: int = 2000) -> float:
        """Expensive: compare ideal vs noisy output distributions via Hellinger fidelity."""
        nm = self.noise_models[qpu_id]
        ideal_sim = AerSimulator()
        noisy_sim = AerSimulator(noise_model=nm)
        
        if qc.num_clbits == 0:
            qc2 = qc.copy()
            qc2.measure_all()
        else:
            qc2 = qc

        ideal = ideal_sim.run(qc2, shots=shots).result().get_counts()
        noisy = noisy_sim.run(qc2, shots=shots).result().get_counts()

        # Hellinger fidelity: (sum sqrt(p_i q_i))^2
        keys = set(ideal) | set(noisy)
        s = 0.0
        for k in keys:
            p = ideal.get(k, 0) / shots
            q = noisy.get(k, 0) / shots
            s += math.sqrt(p*q)
        return max(0.0, min(1.0, s*s))