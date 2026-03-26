from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Literal

from .types import CircuitProfile
from .hardware import HardwareProfile


def predict_exec_time_s(qpu: HardwareProfile, prof: CircuitProfile, *, shots: int) -> float:
    """Pure analytic execution-time proxy in seconds."""
    n1 = int(getattr(prof, "oneq_count", 0) or 0)
    n2 = int(getattr(prof, "twoq_count", 0) or 0)
    m = int(getattr(prof, "meas_count", 0) or 0)

    t1 = float(getattr(qpu, "oneq_gate_time_s", 35e-9) or 35e-9)
    t2 = float(getattr(qpu, "twoq_gate_time_s", 300e-9) or 300e-9)
    tm = float(getattr(qpu, "meas_time_s", 1_000e-9) or 1_000e-9)
    shot_ov = float(getattr(qpu, "shot_overhead_s", 0.0) or 0.0)

    t_gate = n1 * t1 + n2 * t2
    t_meas = (m if m > 0 else 1) * tm
    t_shot = t_gate + t_meas + shot_ov
    return float(shots) * t_shot


def estimate_qpu_execution_s(
    qpu: HardwareProfile,
    *,
    circuit: Optional[Any] = None,
    prof: Optional[CircuitProfile] = None,
    shots: int,
    timing_mode: Optional[str] = None,
) -> tuple[float, dict]:
    """Estimate QPU execution time using analytic or backend-informed timing."""
    mode = str(timing_mode or os.getenv("QDC_QPU_TIMING_MODE", "analytic") or "analytic").strip().lower()

    if mode in ("backend_profile", "backend", "backend_timing"):
        backend = getattr(qpu, "backend_obj", None)
        if backend is not None and circuit is not None:
            try:
                from qdc_sched.ibm.profile_builder import BackendProfileBuilder
            except Exception:
                try:
                    from profile_builder import BackendProfileBuilder
                except Exception:
                    BackendProfileBuilder = None
            if BackendProfileBuilder is not None:
                try:
                    built = BackendProfileBuilder().build_profile(circuit, backend, shots=int(shots or 0))
                    return float(built.total_duration_s), {
                        "timing_mode": mode,
                        "duration_source": built.duration_source,
                        "backend_name": built.backend_name,
                        "per_shot_duration_s": float(built.per_shot_duration_s),
                        "total_duration_s": float(built.total_duration_s),
                        **(built.metadata or {}),
                    }
                except Exception:
                    pass

    if prof is None and circuit is not None:
        from qdc_sched.core.profiler import profile_circuit
        prof = profile_circuit(circuit)

    if prof is None:
        return 0.0, {"timing_mode": mode, "duration_source": "missing_profile"}

    total = predict_exec_time_s(qpu, prof, shots=int(shots or 0))
    return float(total), {
        "timing_mode": "analytic",
        "duration_source": "analytic_gate_time",
        "backend_name": getattr(qpu, "backend_name", None),
        "total_duration_s": float(total),
    }


def estimate_reconstruction_duration_s(
    *,
    num_subexperiments: float,
    num_samples: float,
) -> tuple[float, dict]:
    base = float(os.getenv("QDC_HOST_RECON_BASE_S", "0.0") or 0.0)
    per_exec = float(os.getenv("QDC_HOST_RECON_PER_EXEC_S", "0.0") or 0.0)
    per_sample = float(os.getenv("QDC_HOST_RECON_PER_SAMPLE_S", "0.0") or 0.0)
    n_exec = float(num_subexperiments or 0.0)
    n_samp = float(num_samples or 0.0)
    total = base + per_exec * n_exec + per_sample * n_samp
    return float(total), {
        "base_s": float(base),
        "per_exec_s": float(per_exec),
        "per_sample_s": float(per_sample),
        "num_subexperiments": float(n_exec),
        "num_samples": float(n_samp),
        "total_s": float(total),
    }


def estimate_communication_duration_s(
    *,
    num_subexperiments: float,
    num_samples: float,
    n_qpus_used: float = 1.0,
    plan_kind: Optional[str] = None,
) -> tuple[float, dict]:
    base = float(os.getenv("QDC_COMM_BASE_S", "0.0") or 0.0)
    per_exec = float(os.getenv("QDC_COMM_PER_EXEC_S", "0.0") or 0.0)
    per_sample = float(os.getenv("QDC_COMM_PER_SAMPLE_S", "0.0") or 0.0)
    coord_per_extra_qpu = float(os.getenv("QDC_COMM_COORD_PER_EXTRA_QPU_S", "0.0") or 0.0)

    n_exec = float(num_subexperiments or 0.0)
    n_samp = float(num_samples or 0.0)
    n_qpus = max(1.0, float(n_qpus_used or 1.0))

    total = base + per_exec * n_exec + per_sample * n_samp

    coord_s = 0.0
    if str(plan_kind or "") == "C_CUT_MULTI_QPU":
        coord_s = coord_per_extra_qpu * max(0.0, n_qpus - 1.0)
        total += coord_s

    return float(total), {
        "base_s": float(base),
        "per_exec_s": float(per_exec),
        "per_sample_s": float(per_sample),
        "coord_per_extra_qpu_s": float(coord_per_extra_qpu),
        "num_subexperiments": float(n_exec),
        "num_samples": float(n_samp),
        "n_qpus_used": float(n_qpus),
        "coord_s": float(coord_s),
        "total_s": float(total),
    }


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