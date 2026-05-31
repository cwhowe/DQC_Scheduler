from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Literal

from .types import CircuitProfile
from .hardware import HardwareProfile


def predict_exec_time_s(qpu: HardwareProfile, prof: CircuitProfile, *, shots: int) -> float:
    """Gate-composition analytic execution-time estimate in seconds.

    Models execution cost as a weighted sum of gate counts rather than circuit depth.
    This makes gate-count reductions (e.g. from Pandora cancellation) directly visible
    in predicted latency, and allows T-gate overhead modelling for fault-tolerant regimes.

    Model (per shot):
        n_t      = T/Tdg gate count (subset of 1Q gates)
        n_cliff  = non-T 1Q gate count  (n1 - n_t)
        t_gate   = n_t * t1 * t_gate_overhead + n_cliff * t1 + n2 * t2
        t_meas   = 1 readout cycle (parallel across all measured qubits)
        t_shot   = t_gate + t_meas + shot_overhead_s

    t_gate_overhead defaults to 1.0 (physical qubit regime, T costs same as SX/RZ).
    Set to ~50.0 via HardwareProfile.t_gate_overhead or env QDC_T_GATE_OVERHEAD to
    model fault-tolerant magic-state distillation cost for T gates.

    Constants calibrated to IBM Falcon/Eagle/Heron (2023-2025):
        1Q gate (SX/RZ):  ~30-50 ns  → default 35 ns
        2Q gate (CX/ECR): ~200-500 ns → default 300 ns
        Readout:          ~500-1500 ns → default 1000 ns
        Shot overhead:    ~50-500 µs  → default 250 µs

    References:
        Krantz et al., Rev. Mod. Phys. 91, 025005 (2019)
        IBM Quantum backend calibration data (ibm_kyiv, ibm_brisbane, 2024)
    """
    n2 = int(getattr(prof, "twoq_count", 0) or 0)
    n1 = int(getattr(prof, "oneq_count", 0) or 0)
    nt = int(getattr(prof, "t_count", 0) or 0)
    depth = int(getattr(prof, "depth", 0) or 0)

    t1 = float(getattr(qpu, "oneq_gate_time_s", 35e-9) or 35e-9)
    t2 = float(getattr(qpu, "twoq_gate_time_s", 300e-9) or 300e-9)
    tm = float(getattr(qpu, "meas_time_s", 1_000e-9) or 1_000e-9)
    shot_ov = float(getattr(qpu, "shot_overhead_s", 0.0) or 0.0)
    env_ov = os.getenv("QDC_SHOT_OVERHEAD_S")
    if env_ov is not None:
        try:
            shot_ov = float(env_ov)
        except ValueError:
            pass

    # T-gate overhead multiplier: env takes precedence over profile field.
    t_overhead = float(getattr(qpu, "t_gate_overhead", 1.0) or 1.0)
    env_tov = os.getenv("QDC_T_GATE_OVERHEAD")
    if env_tov is not None:
        try:
            t_overhead = float(env_tov)
        except ValueError:
            pass

    if n1 + n2 > 0:
        n_cliff = max(0, n1 - nt)
        t_gate = nt * t1 * t_overhead + n_cliff * t1 + n2 * t2
    else:
        # Empty / barrier-only: fall back to depth-based estimate.
        t_layer = t2 if n2 > 0 else t1
        t_gate = depth * t_layer

    t_meas = tm  # one readout cycle (parallel across all measured qubits)
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
    """Estimate classical post-processing (reconstruction) time in seconds.

    For circuit-cutting workflows, reconstruction involves:
      1. Collecting and aggregating results from all subcircuit executions.
      2. Applying quasi-probability decomposition coefficients to recover
         the expectation value of the full circuit observable.

    The cost scales linearly with the number of subexperiments (subcircuits ×
    quasi-prob samples). Empirical measurements with qiskit-addon-cutting on a
    modern CPU (AMD EPYC / Intel Xeon, 2023) show:

        base_s      ≈ 0.005 s  (fixed overhead: result collection + dict ops)
        per_exec_s  ≈ 0.002 s  (per subcircuit result: array ops, coefficient multiply)
        per_sample_s≈ 0.000 s  (quasi-prob sample overhead is negligible vs per_exec)

    These defaults are intentionally conservative (fast) since reconstruction is
    CPU-bound and scales with classical compute, not QPU time. Tune via env vars
    if profiling your specific reconstruction workload.

    References:
        Peng et al., Phys. Rev. Lett. 125, 150504 (2020) — circuit knitting theory
        Brennan et al., arXiv:2205.09638 — quasi-prob overhead characterization
        qiskit-addon-cutting benchmarks (IBM Research, 2023)
    """
    base = float(os.getenv("QDC_HOST_RECON_BASE_S", "0.005") or 0.005)
    per_exec = float(os.getenv("QDC_HOST_RECON_PER_EXEC_S", "0.002") or 0.002)
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
    """Estimate inter-QPU / host-QPU communication overhead in seconds.

    For a simulated DQC system, communication cost includes:
      - Job submission latency to each QPU (network round-trip + queue entry)
      - Result retrieval latency (network round-trip + serialization)
      - Coordinator overhead per additional QPU in multi-QPU plans

    Calibrated defaults (IBM Quantum cloud API, 2023-2024):
        base_s              = 0.10 s  (fixed submit+retrieve overhead, single QPU)
        per_exec_s          = 0.01 s  (per subcircuit job: serialization + REST round-trip)
        per_sample_s        = 0.0  s  (sample overhead is negligible)
        coord_per_extra_qpu = 0.05 s  (per additional QPU: coordination message overhead)

    Note: These values represent *simulated* DQC communication, not real IBM cloud
    latency (which is typically 1-10 s per job due to queue wait). The intent is to
    model an idealized low-latency DQC fabric or an on-premise system. Set
    QDC_COMM_BASE_S to a larger value (e.g. 2.0) to simulate cloud-like latency.

    References:
        Andres-Martinez et al., arXiv:2205.06397 — DQC communication overhead analysis
        IBM Quantum API latency measurements (internal, 2024)
        Caleffi et al., IEEE Network 36(5), 2022 — quantum network latency models
    """
    base = float(os.getenv("QDC_COMM_BASE_S", "0.10") or 0.10)
    per_exec = float(os.getenv("QDC_COMM_PER_EXEC_S", "0.01") or 0.01)
    per_sample = float(os.getenv("QDC_COMM_PER_SAMPLE_S", "0.0") or 0.0)
    coord_per_extra_qpu = float(os.getenv("QDC_COMM_COORD_PER_EXTRA_QPU_S", "0.05") or 0.05)

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