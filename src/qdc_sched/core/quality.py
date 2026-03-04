from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .types import CircuitProfile

@dataclass
class QualityModel:
    """Provides a fast fidelity proxy and an optional expensive estimated fidelity (ideal vs noisy)"""
    noise_models: Dict[str, NoiseModel]

    def fidelity_proxy_from_profile(self, qpu_id: str, prof: CircuitProfile) -> float:
        # For now proxy: exponential decay with 2Q and 1Q counts (2/06/2026)
        # Replace with calibration / learned proxy later.
        # defaults:
        p2 = 0.010
        p1 = 0.001
        fid = math.exp(-p2*prof.twoq_count - p1*prof.oneq_count)
        # clamp
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