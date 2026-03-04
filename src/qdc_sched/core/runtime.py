from __future__ import annotations

from .types import CircuitProfile
from .hardware import HardwareProfile


def predict_exec_time_s(qpu: HardwareProfile, prof: CircuitProfile, *, shots: int) -> float:
    """Predict execution time (seconds) using a simple device-time proxy.

    Goal: keep planner 'pred_total_s' consistent with executor accounting.

    Model:
      total = base_queue_delay_s + shots * (gate_time + meas_time + shot_overhead_s)

    Where:
      gate_time = oneq_count * oneq_gate_time_s + twoq_count * twoq_gate_time_s
      meas_time = meas_count * meas_time_s   (or 1 * meas_time_s if meas_count missing/0)
    """
    # Operation counts
    n1 = int(getattr(prof, "oneq_count", 0) or 0)
    n2 = int(getattr(prof, "twoq_count", 0) or 0)
    m = int(getattr(prof, "meas_count", 0) or 0)

    # Hardware timing params (seconds)
    t1 = float(getattr(qpu, "oneq_gate_time_s", 35e-9) or 35e-9)
    t2 = float(getattr(qpu, "twoq_gate_time_s", 300e-9) or 300e-9)
    tm = float(getattr(qpu, "meas_time_s", 1_000e-9) or 1_000e-9)
    shot_ov = float(getattr(qpu, "shot_overhead_s", 0.0) or 0.0)
    #base_q = float(getattr(qpu, "base_queue_delay_s", 0.0) or 0.0)

    # Per-shot device time
    t_gate = n1 * t1 + n2 * t2
    t_meas = (m if m > 0 else 1) * tm
    t_shot = t_gate + t_meas + shot_ov

    return float(shots) * t_shot



# -------------------------
# Optional Aer wall-clock timing
# -------------------------
from dataclasses import dataclass
from typing import Any, Optional, Literal

try:
    from qiskit import QuantumCircuit
    from qiskit_aer.primitives import SamplerV2, EstimatorV2
    from qiskit_aer.noise import NoiseModel
except Exception:
    QuantumCircuit = Any
    SamplerV2 = None
    EstimatorV2 = None
    NoiseModel = Any

import time as _time


@dataclass(frozen=True)
class AerTimingResult:
    """Wall-clock timing (seconds) measured locally with Qiskit Aer primitives."""
    exec_s: float
    repeats: int = 1


def measure_exec_time_s_aer(
    circuit: Any,
    *,
    shots: int,
    task_type: Literal["counts", "expectation"] = "counts",
    observables: Optional[Any] = None,
    noise_model: Optional[Any] = None,
    repeats: int = 1,
) -> AerTimingResult:
    """Measure local wall-clock primitive runtime using Qiskit Aer.

    Notes / intent:
      - This measures *local CPU time*, not real hardware service time.
      - Useful as a more realistic proxy than the analytic timing model because it
        captures transpilation artifacts indirectly and simulator scaling with width/depth/shots.
    """
    if SamplerV2 is None or EstimatorV2 is None:
        raise RuntimeError("Qiskit Aer primitives not available; install qiskit-aer.")

    reps = max(1, int(repeats))
    total = 0.0
    nm = noise_model
    try:
        if nm is not None and NoiseModel is not Any and not isinstance(nm, NoiseModel):
            nm = None
    except Exception:
        nm = None

    if task_type == "counts":
        try:
            if getattr(circuit, "num_clbits", 0) == 0:
                qc2 = circuit.copy()
                qc2.measure_all()
            else:
                qc2 = circuit
        except Exception:
            qc2 = circuit

        sampler = SamplerV2(options={"backend_options": {"noise_model": nm}}) if nm is not None else SamplerV2()
        for _ in range(reps):
            t0 = _time.perf_counter()
            _ = sampler.run([qc2], shots=shots).result()
            total += (_time.perf_counter() - t0)
        return AerTimingResult(exec_s=total / reps, repeats=reps)

    if observables is None:
        raise ValueError("Expectation timing requires observables.")
    estimator = EstimatorV2(options={"backend_options": {"noise_model": nm}}) if nm is not None else EstimatorV2()
    for _ in range(reps):
        t0 = _time.perf_counter()
        _ = estimator.run([(circuit, observables)]).result()
        total += (_time.perf_counter() - t0)
    return AerTimingResult(exec_s=total / reps, repeats=reps)