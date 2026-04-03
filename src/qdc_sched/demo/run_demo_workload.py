from __future__ import annotations

import os
import random
import time
import csv
import json
from collections import Counter
from typing import List, Tuple, Dict, Callable, Set

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qdc_sched.core.hardware import HardwareProfile, QPUState
from qdc_sched.core.quality import QualityModel
from qdc_sched.core.scheduler import Scheduler, SchedulerConfig
from qdc_sched.core.executor import ExecConfig
from qdc_sched.core.types import Job, RunToggles, JobConstraints
from qdc_sched.ibm.fake_loader import make_ibm_fake_qpu_set

def ghz(n: int, measure: bool = True) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    if measure:
        qc.measure_all()
    return qc


def random_cx(
    n: int,
    depth: int,
    measure: bool = True,
    rng: random.Random | None = None,
) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    rng = rng or random.Random(1234 + 17 * n + depth)
    for _ in range(depth):
        a = rng.randrange(n)
        b = rng.randrange(n)
        while b == a:
            b = rng.randrange(n)
        qc.cx(a, b)
        qc.h(a)
    if measure:
        qc.measure_all()
    return qc


def z_observable(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_list([("Z" * n, 1.0)])


def make_line_qpu(qpu_id: str, n: int, base_queue_delay_s: float) -> QPUState:
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(i, i + 1) for i in range(n - 1)])
    hp = HardwareProfile(
        qpu_id=qpu_id,
        num_qubits=n,
        coupling_graph=g,
        base_queue_delay_s=base_queue_delay_s,
    )
    object.__setattr__(hp, "reconstruction_per_exec_s", float(os.getenv("QDC_DEMO_RECON_PER_EXEC_S", "0.0")))
    return QPUState(hp)


def _parse_int_list(env_name: str, default_csv: str) -> List[int]:
    raw = os.getenv(env_name, default_csv)
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out



def _parse_float_list(env_name: str, default_csv: str) -> List[float]:
    raw = os.getenv(env_name, default_csv)
    out: List[float] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out




def _parse_str_list(env_name: str, default_csv: str = "") -> List[str]:
    raw = os.getenv(env_name, default_csv)
    out: List[str] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(x)
    return out


def _logical_qpu_slot_names(n: int) -> List[str]:
    """
    Generate logical scheduler-facing slot names:
      qpu_A, qpu_B, ..., qpu_Z, qpu_27, qpu_28, ...
    """
    names: List[str] = []
    for i in range(n):
        if i < 26:
            names.append(f"qpu_{chr(ord('A') + i)}")
        else:
            names.append(f"qpu_{i + 1}")
    return names


def _apply_demo_base_delay_overrides(qpus: Dict[str, QPUState]) -> None:
    """
    Apply optional demo-level base-delay overrides to any logical slot.

    Supported forms:
      - legacy per-slot envs: QDC_QPU_A_BASE_DELAY, QDC_QPU_B_BASE_DELAY, ...
      - generic env per logical slot: QDC_QPU_qpu_A_BASE_DELAY, etc.
      - CSV list in slot order: QDC_QPU_BASE_DELAYS=1.5,0.2,0.3,0.3
    """
    csv_raw = os.getenv("QDC_QPU_BASE_DELAYS", "").strip()
    csv_delays = [float(x.strip()) for x in csv_raw.split(",") if x.strip()] if csv_raw else []
    slot_names = sorted(qpus.keys())

    for idx, slot in enumerate(slot_names):
        override = None

        generic_env = f"QDC_QPU_{slot}_BASE_DELAY"
        if os.getenv(generic_env) is not None:
            try:
                override = float(os.getenv(generic_env, "0.0"))
            except Exception:
                override = None

        if override is None and slot.startswith("qpu_"):
            suffix = slot.split("_", 1)[1]
            legacy_env = f"QDC_QPU_{suffix}_BASE_DELAY"
            if os.getenv(legacy_env) is not None:
                try:
                    override = float(os.getenv(legacy_env, "0.0"))
                except Exception:
                    override = None

        if override is None and idx < len(csv_delays):
            override = float(csv_delays[idx])

        if override is not None:
            try:
                qpus[slot].profile.base_queue_delay_s = override
            except Exception:
                pass


def _remap_qpu_slots(
    qpus_by_backend: Dict[str, QPUState],
    noise_models_by_backend: Dict[str, object],
) -> Tuple[Dict[str, QPUState], Dict[str, object]]:
    """
    Remap backend-id keyed IBM fake QPUs onto logical demo slots qpu_A, qpu_B, ...

    Important:
    - dict keys become logical slot names for scheduler/demo logic
    - profile.qpu_id remains the underlying backend id (e.g., fake_casablanca)
      so exported records still reflect the real backend used
    """
    items = list(qpus_by_backend.items())
    slot_names = _logical_qpu_slot_names(len(items))

    qpus: Dict[str, QPUState] = {}
    noise_models: Dict[str, object] = {}

    for slot_name, (backend_id, qpu_state) in zip(slot_names, items):
        qpus[slot_name] = qpu_state
        if backend_id in noise_models_by_backend:
            noise_models[slot_name] = noise_models_by_backend[backend_id]

    return qpus, noise_models

def build_expectation_circuit(n: int, rng: random.Random, *, family: str, depth: int | None = None) -> QuantumCircuit:
    fam = (family or "ghz").strip().lower()
    if fam == "ghz":
        return ghz(n, measure=False)
    if fam in ("random_cx", "randcx", "cx"):
        d = int(depth if depth is not None else max(4, 2 * n))
        return random_cx(n, d, measure=False, rng=rng)
    if fam in ("ghz_then_mix", "hybrid"):
        qc = ghz(n, measure=False)
        d = int(depth if depth is not None else max(4, n))
        mix = random_cx(n, d, measure=False, rng=rng)
        qc.compose(mix, inplace=True)
        return qc
    raise ValueError(f"Unsupported QDC_WIDE_FAMILY={family!r}")


def _planned_wide_submit_times(n_wide: int, rng: random.Random) -> List[float]:
    if n_wide <= 0:
        return []
    align = os.getenv("QDC_WIDE_ALIGN_TO_BURSTS", "1") == "1"
    if not align:
        return []
    burst_starts = _parse_float_list("QDC_BURST_STARTS_S", "15,40")
    jitter_choices = _parse_float_list("QDC_WIDE_BURST_JITTER_S", "-2.0,-1.0,-0.5,0.0,0.2,0.5,1.0,2.0")
    times: List[float] = []
    if not burst_starts:
        return times
    reps = (n_wide + len(burst_starts) - 1) // len(burst_starts)
    anchors = (burst_starts * reps)[:n_wide]
    rng.shuffle(anchors)
    for a in anchors:
        times.append(max(0.0, float(a) + float(rng.choice(jitter_choices))))
    times.sort()
    return times


def make_workload(
    n_jobs: int | None = None,
    pct_expect: float | None = None,
    pct_wide: float | None = None,
) -> Tuple[List[Tuple[float, Job]], Set[str]]:
    if n_jobs is None:
        n_jobs = int(os.getenv("QDC_N_JOBS", "160"))
    if pct_expect is None:
        pct_expect = float(os.getenv("QDC_PCT_EXPECT", "0.40"))
    if pct_wide is None:
        pct_wide = float(os.getenv("QDC_PCT_WIDE", "0.15"))

    disable_cutting = os.getenv("QDC_DISABLE_CUTTING", "0") == "1"
    disable_multi_qpu = os.getenv("QDC_DISABLE_MULTI_QPU", "0") == "1"
    allow_cutting = not disable_cutting
    allow_multi_qpu = not disable_multi_qpu

    wl: List[Tuple[float, Job]] = []
    wide_ids: Set[str] = set()
    t = 0.0
    seed = int(os.getenv("QDC_SEED", "2026"))
    rng = random.Random(seed)

    n_wide = int(round(n_jobs * pct_wide))
    wide_slots = set(rng.sample(range(n_jobs), k=n_wide)) if n_wide > 0 else set()

    wide_widths = _parse_int_list("QDC_WIDE_WIDTHS", "10,12,14,16")
    med_widths = _parse_int_list("QDC_MED_EXPECT_WIDTHS", "4,6,7")
    wide_shots = _parse_int_list("QDC_WIDE_SHOTS", "500,1000,2000")
    wide_depths = _parse_int_list("QDC_WIDE_DEPTHS", "16,24,32")
    wide_family = os.getenv("QDC_WIDE_FAMILY", "random_cx")
    wide_submit_times = _planned_wide_submit_times(n_wide, rng)
    wide_submit_idx = 0

    for i in range(n_jobs):
        t += rng.choice([0.0, 0.1, 0.2, 0.5, 1.0])
        jid = f"J{i:03d}"

        if i in wide_slots:
            width = rng.choice(wide_widths)
            shots = rng.choice(wide_shots)
            depth = rng.choice(wide_depths)
            qc = build_expectation_circuit(width, rng, family=wide_family, depth=depth)
            submit_t = t
            if wide_submit_idx < len(wide_submit_times):
                submit_t = float(wide_submit_times[wide_submit_idx])
                wide_submit_idx += 1
            job = Job(
                job_id=jid,
                circuit=qc,
                task_type="expectation",
                observables=z_observable(width),
                shots=shots,
                submit_time_s=submit_t,
                constraints=JobConstraints(
                    allow_cutting=allow_cutting,
                    force_cutting=(os.getenv("QDC_FORCE_CUT_WIDE", "1") == "1"),
                    allow_multi_qpu=allow_multi_qpu,
                    max_cuts=int(os.getenv("QDC_WIDE_MAX_CUTS", "2")),
                    comm_overhead_s=float(os.getenv("QDC_WIDE_COMM_OVERHEAD_S", "0.08")),
                ),
            )
            wl.append((submit_t, job))
            wide_ids.add(jid)
            continue

        if rng.random() < (1.0 - pct_expect):
            width = rng.choice([3, 5, 7])
            depth = rng.choice([3, 6, 9])
            shots = rng.choice([500, 1000, 2000])
            qc = random_cx(width, depth, measure=True, rng=rng)
            job = Job(job_id=jid, circuit=qc, task_type="counts", shots=shots, submit_time_s=t)
            wl.append((t, job))
            continue

        width = rng.choice(med_widths)
        shots = rng.choice([100, 200, 500])
        med_family = os.getenv("QDC_MED_EXPECT_FAMILY", "ghz")
        med_depths = _parse_int_list("QDC_MED_EXPECT_DEPTHS", "8,12,16")
        qc = build_expectation_circuit(width, rng, family=med_family, depth=rng.choice(med_depths))

        force_cut = (os.getenv("QDC_FORCE_CUT_MED", "0") == "1")
        if width > int(os.getenv("QDC_FORCE_CUT_IF_GT", "9999")):
            force_cut = True

        job = Job(
            job_id=jid,
            circuit=qc,
            task_type="expectation",
            observables=z_observable(width),
            shots=shots,
            submit_time_s=t,
            constraints=JobConstraints(
                allow_cutting=allow_cutting,
                force_cutting=force_cut,
                allow_multi_qpu=allow_multi_qpu,
                max_cuts=int(os.getenv("QDC_MAX_CUTS", "6")),
                comm_overhead_s=float(os.getenv("QDC_COMM_OVERHEAD_S", "0.02")),
            ),
        )
        wl.append((t, job))

    wl.sort(key=lambda x: (float(x[0]), x[1].job_id))
    return wl, wide_ids


def _install_reserve_counters(qpus: Dict[str, QPUState]):
    reserve_counts = Counter()
    for qid, st in qpus.items():
        orig_reserve = getattr(st, "reserve", None)
        if orig_reserve is None or not callable(orig_reserve):
            continue

        def make_wrapped(qid_local: str, orig_fn: Callable):
            def wrapped(job_id, qubits, start_s, duration_s, *args, **kwargs):
                reserve_counts[qid_local] += 1
                return orig_fn(job_id, qubits, start_s, duration_s, *args, **kwargs)
            return wrapped

        st.reserve = make_wrapped(qid, orig_reserve)
    return reserve_counts


def inject_congestion_bursts(qpus: Dict[str, QPUState]) -> None:
    burst_dur = float(os.getenv("QDC_CONGEST_BURST_S", "15.0"))
    burst_starts = _parse_float_list("QDC_BURST_STARTS_S", "15,40")
    block_targets = [x.strip() for x in os.getenv("QDC_BURST_TARGETS", "qpu_B").split(",") if x.strip()]
    block_fraction = float(os.getenv("QDC_BURST_BLOCK_FRACTION", "1.0"))
    for qid in block_targets:
        if qid not in qpus:
            continue
        k_full = int(qpus[qid].profile.num_qubits)
        k = max(1, min(k_full, int(round(k_full * block_fraction))))
        for i, t0 in enumerate(burst_starts):
            qpus[qid].reserve(f"BLOCK_{qid}_{i}", list(range(k)), start_s=float(t0), duration_s=burst_dur)


def _install_plan_debug(sched: Scheduler, wide_job_ids: Set[str]) -> Counter:
    ctr: Counter = Counter()
    exec_obj = getattr(sched, "executor", None)
    if exec_obj is None:
        return ctr

    orig = exec_obj.run_job_plan
    wide_debug_budget = {"n": int(os.getenv("QDC_WIDE_DEBUG_N", "10"))}

    def wrapped(job, plan, now_s, toggles):
        kind = str(getattr(plan, "kind", "UNKNOWN"))
        ctr[kind] += 1
        if getattr(job, "job_id", None) in wide_job_ids and wide_debug_budget["n"] > 0:
            wide_debug_budget["n"] -= 1
            details = getattr(plan, "details", None) or {}
            ps = {}
            if isinstance(details, dict) and isinstance(details.get("plan_scores"), dict):
                for k, v in details["plan_scores"].items():
                    if isinstance(v, dict) and "score" in v:
                        ps[k] = v.get("score")
                    else:
                        ps[k] = None
            psk = details.get("plan_skips") if isinstance(details, dict) else None
            if isinstance(psk, dict):
                psk_summary = {bk: [x.get("reason") for x in vals][-3:] for bk, vals in psk.items() if vals}
            else:
                psk_summary = None
            print(f"[WIDE-DBG] job={job.job_id} kind={kind} qpu_id={getattr(plan,'qpu_id',None)} scores={ps} skips={psk_summary}")
        return orig(job, plan, now_s, toggles)

    exec_obj.run_job_plan = wrapped
    return ctr


def _print_histogram(ctr: Counter) -> None:
    total = sum(ctr.values())
    print("\n================ Plan histogram ================")
    for kind, n in sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = 100.0 * n / total if total else 0.0
        print(f"{kind:20s}  {n:4d}  ({pct:5.1f}%)")
    print(f"{'TOTAL':20s}  {total:4d}  (100.0%)")


def _export_results(sched: Scheduler, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    records = []
    try:
        records = getattr(getattr(sched, "metrics", None), "records", []) or []
    except Exception:
        records = []

    rec_path = os.path.join(outdir, "records.csv")
    with open(rec_path, "w", newline="") as f:
        fieldnames = [
            "job_id","qpu_id","plan_kind","submit_time_s",
            "t_schedule_s","t_partition_s","t_mapping_s","t_execution_s",
            "t_reconstruction_s","end_to_end_s","fidelity_proxy",
            "comm_queue_delay_s","comm_busy_time_s",
            "cpu_queue_delay_s","cpu_busy_time_s",
            "sim_comm_queue_s","sim_comm_service_s","sim_comm_s",
            "sim_recon_queue_s","sim_recon_service_s","sim_recon_s",
            "pred_comm_queue_s","pred_comm_service_s","pred_comm_s",
            "fidelity_estimated","details_json"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({
                "job_id": getattr(r, "job_id", None),
                "qpu_id": getattr(r, "qpu_id", None),
                "plan_kind": getattr(r, "plan_kind", None),
                "submit_time_s": getattr(r, "submit_time_s", None),
                "t_schedule_s": getattr(r, "t_schedule_s", None),
                "t_partition_s": getattr(r, "t_partition_s", None),
                "t_mapping_s": getattr(r, "t_mapping_s", None),
                "t_execution_s": getattr(r, "t_execution_s", None),
                "t_reconstruction_s": getattr(r, "t_reconstruction_s", None),
                "end_to_end_s": getattr(r, "end_to_end_s", None),
                "fidelity_proxy": getattr(r, "fidelity_proxy", None),
                "comm_queue_delay_s": ((getattr(r, "details", {}) or {}).get("comm_queue_delay_s") if getattr(r, "details", None) is not None else None),
                "comm_busy_time_s": ((getattr(r, "details", {}) or {}).get("comm_busy_time_s") if getattr(r, "details", None) is not None else None),
                "cpu_queue_delay_s": ((getattr(r, "details", {}) or {}).get("cpu_queue_delay_s") if getattr(r, "details", None) is not None else None),
                "cpu_busy_time_s": ((getattr(r, "details", {}) or {}).get("cpu_busy_time_s") if getattr(r, "details", None) is not None else None),
                "sim_comm_queue_s": ((getattr(r, "details", {}) or {}).get("sim_comm_queue_s") if getattr(r, "details", None) is not None else None),
                "sim_comm_service_s": ((getattr(r, "details", {}) or {}).get("sim_comm_service_s") if getattr(r, "details", None) is not None else None),
                "sim_comm_s": ((getattr(r, "details", {}) or {}).get("sim_comm_s") if getattr(r, "details", None) is not None else None),
                "sim_recon_queue_s": ((getattr(r, "details", {}) or {}).get("sim_recon_queue_s") if getattr(r, "details", None) is not None else None),
                "sim_recon_service_s": ((getattr(r, "details", {}) or {}).get("sim_recon_service_s") if getattr(r, "details", None) is not None else None),
                "sim_recon_s": ((getattr(r, "details", {}) or {}).get("sim_recon_s") if getattr(r, "details", None) is not None else None),
                "pred_comm_queue_s": ((getattr(r, "details", {}) or {}).get("pred_comm_queue_s") if getattr(r, "details", None) is not None else None),
                "pred_comm_service_s": ((getattr(r, "details", {}) or {}).get("pred_comm_service_s") if getattr(r, "details", None) is not None else None),
                "pred_comm_s": ((getattr(r, "details", {}) or {}).get("pred_comm_s") if getattr(r, "details", None) is not None else None),
                "fidelity_estimated": getattr(r, "fidelity_estimated", None),
                "details_json": json.dumps(getattr(r, "details", {}) or {}, default=str),
            })

    events = []
    try:
        events = getattr(sched, "task_log", []) or []
    except Exception:
        events = []

    evt_path = os.path.join(outdir, "events.csv")
    with open(evt_path, "w", newline="") as f:
        fieldnames = [
            "task_id","job_id","kind","start_s","end_s","qpu_id","qubits","label","depends_on","metadata_json"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in events:
            w.writerow({
                "task_id": getattr(t, "task_id", None),
                "job_id": getattr(t, "job_id", None),
                "kind": str(getattr(t, "kind", None)),
                "start_s": getattr(t, "start_s", None),
                "end_s": getattr(t, "end_s", None),
                "qpu_id": getattr(t, "qpu_id", None),
                "qubits": json.dumps(getattr(t, "qubits", None), default=str),
                "label": getattr(t, "label", None),
                "depends_on": json.dumps(getattr(t, "depends_on", None), default=str),
                "metadata_json": json.dumps(getattr(t, "metadata", {}) or {}, default=str),
            })

    print(f"[EXPORT] wrote {rec_path} and {evt_path}")


def _active_reservations(qpus: Dict[str, QPUState], now_s: float) -> int:
    total = 0
    for st in qpus.values():
        try:
            active = getattr(st, "active_reservations", []) or []
            total += sum(1 for r in active if float(getattr(r, "end_s", 0.0)) > float(now_s))
        except Exception:
            continue
    return total


def run_workload(full_eval: bool = False) -> None:
    exclude_c = os.getenv("QDC_EXCLUDE_QPU_C", "1") == "1"
    disable_cutting = os.getenv("QDC_DISABLE_CUTTING", "0") == "1"
    disable_multi_qpu = os.getenv("QDC_DISABLE_MULTI_QPU", "0") == "1"

    qpu_source = os.getenv("QDC_QPU_SOURCE", "line").strip().lower()

    if qpu_source == "ibm_fake":
        selected_backends = _parse_str_list("QDC_IBM_FAKE_BACKENDS", "")
        max_backends = max(1, len(selected_backends)) if selected_backends else 3

        raw_qpus, raw_noise_models = make_ibm_fake_qpu_set(
            max_backends=max_backends,
            prefer=selected_backends if selected_backends else None,
        )

        qpus, _noise_models = _remap_qpu_slots(raw_qpus, raw_noise_models)

        if exclude_c and "qpu_C" in qpus:
            del qpus["qpu_C"]
            _noise_models.pop("qpu_C", None)

        _apply_demo_base_delay_overrides(qpus)
    else:
        qpu_source = os.getenv("QDC_QPU_SOURCE", "line").strip().lower()

        if qpu_source == "ibm_fake":
            qpus, _noise_models = make_ibm_fake_qpu_set()

            if exclude_c and "qpu_C" in qpus:
                del qpus["qpu_C"]

            _apply_demo_base_delay_overrides(qpus)
        else:
            line_sizes = _parse_int_list("QDC_LINE_QPU_SIZES", "7,7,14")
            slot_names = _logical_qpu_slot_names(len(line_sizes))
            qpus: Dict[str, QPUState] = {}
            for slot, nqubits in zip(slot_names, line_sizes):
                if exclude_c and slot == "qpu_C":
                    continue
                qpus[slot] = make_line_qpu(slot, int(nqubits), base_queue_delay_s=0.3)

            _apply_demo_base_delay_overrides(qpus)

    reserve_counts = _install_reserve_counters(qpus)
    inject_congestion_bursts(qpus)

    cfg = SchedulerConfig()
    timing_mode_env = os.getenv("QDC_TIMING_MODE", os.getenv("QDC_QPU_TIMING_MODE", "analytic")).strip().lower()
    if timing_mode_env in ("aer", "aer_timing"):
        timing_mode = "aer"
    elif timing_mode_env in ("backend_profile", "backend", "backend_timing"):
        timing_mode = "backend_profile"
    else:
        timing_mode = "analytic"
    cfg.exec_cfg = ExecConfig(
        reserve_nonsim=True,
        timing_mode=timing_mode,
        aer_timing_repeats=int(os.getenv("QDC_AER_TIMING_REPEATS", "1")),
        aer_timing_use_noise=(os.getenv("QDC_AER_TIMING_USE_NOISE", "0") == "1"),
        aer_timing_include_transpile=(os.getenv("QDC_AER_TIMING_INCLUDE_TRANSPILE", "0") == "1"),
    )

    sched = Scheduler(qpus=qpus, quality=QualityModel(noise_models={}), cfg=cfg)

    wl, wide_ids = make_workload()
    plan_ctr = _install_plan_debug(sched, wide_ids)

    print("[DEMO] reserve_nonsim =", bool(getattr(sched.executor.cfg, "reserve_nonsim", False)))
    print("[DEMO] QDC_EXCLUDE_QPU_C =", exclude_c)
    print("[DEMO] QDC_DISABLE_CUTTING =", disable_cutting)
    print("[DEMO] QDC_DISABLE_MULTI_QPU =", disable_multi_qpu)
    try:
        chosen_ids = {slot: qpus[slot].profile.qpu_id for slot in sorted(qpus.keys())}
        print("[DEMO] active backend pool =", chosen_ids)
    except Exception:
        pass

    print("\n================ REAL WORKLOAD RUN ================")
    print("full_eval:", full_eval)
    print("timing_mode:", getattr(sched.executor.cfg, "timing_mode", None))
    print("aer_repeats:", os.getenv("QDC_AER_TIMING_REPEATS", "1"))
    print("jobs:", len(wl))
    print("seed:", int(os.getenv("QDC_SEED", "2026")))
    print("congestion_burst_s:", float(os.getenv("QDC_CONGEST_BURST_S", "15.0")))
    print("burst_starts_s:", _parse_float_list("QDC_BURST_STARTS_S", "15,40"))
    print("wide_jobs:", len(wide_ids))
    print("wide_family:", os.getenv("QDC_WIDE_FAMILY", "random_cx"))
    print("wide_depths:", _parse_int_list("QDC_WIDE_DEPTHS", "16,24,32"))
    print("wide_shots:", _parse_int_list("QDC_WIDE_SHOTS", "500,1000,2000"))
    print("wide_align_to_bursts:", os.getenv("QDC_WIDE_ALIGN_TO_BURSTS", "1") == "1")

    toggles = RunToggles(
        compute_estimated_fidelity=False,
        compute_expectation=bool(full_eval),
        simulate_only=False,
    )

    now = 0.0
    next_idx = 0
    idle_steps = 0
    max_steps = int(os.getenv("QDC_MAX_STEPS", "50000"))
    stop_reason = None
    max_to_schedule = int(os.getenv("QDC_MAX_TO_SCHEDULE", "2"))

    step_wall_cap_s = float(os.getenv("QDC_STEP_WALL_CAP_S", "15.0"))
    idle_break_steps = int(os.getenv("QDC_IDLE_BREAK_STEPS", "5"))

    for step in range(max_steps):
        while next_idx < len(wl) and wl[next_idx][0] <= now + 1e-12:
            _, job = wl[next_idx]
            sched.submit(job, toggles)
            next_idx += 1

        t0 = time.time()
        max_to_schedule = int(os.getenv("QDC_MAX_TO_SCHEDULE", str(max_to_schedule)))
        started = sched.step(now_s=now, max_to_schedule=max_to_schedule)
        wall = time.time() - t0
        if wall > step_wall_cap_s:
            print(f"[WATCHDOG] sched.step took {wall:.2f}s at t={now:.2f} (cap={step_wall_cap_s:.2f}s).")
            action = os.getenv("QDC_WATCHDOG_ACTION", "adapt").strip().lower()
            if action == "break":
                print("If using Aer timing, try: QDC_AER_TIMING_REPEATS=1 or lower shots.")
                stop_reason = f"watchdog wall={wall:.2f}s at t={now:.2f}"
                break
            max_to_schedule = max(1, max_to_schedule // 2)
            os.environ["QDC_MAX_TO_SCHEDULE"] = str(max_to_schedule)
            print(f"[WATCHDOG] adapting: setting QDC_MAX_TO_SCHEDULE={max_to_schedule} and continuing.")
            sched.tick(0.5)
            now += 0.5
            continue

        if started:
            idle_steps = 0
        else:
            idle_steps += 1

        pending = sched.pending_count()
        active_res = _active_reservations(qpus, now)

        if next_idx >= len(wl) and pending == 0 and active_res == 0:
            stop_reason = "all_jobs_completed_and_no_active_reservations"
            break

        if next_idx >= len(wl) and pending == 0 and idle_steps >= idle_break_steps:
            stop_reason = f"idle_break_after_completion({idle_break_steps})"
            break

        dt = 0.5 if started else (max(0.5, float(wl[next_idx][0] - now)) if next_idx < len(wl) else 1.0)
        sched.tick(dt)
        now += dt

        if step % 5 == 0:
            print(
                f"[t={now:7.2f}] reserve_calls={dict(reserve_counts)} | "
                f"pending={pending} active_res={active_res}"
            )

    else:
        if stop_reason is None:
            stop_reason = "max_steps reached"

    completed = len(getattr(sched.metrics, "records", [])) if hasattr(sched, "metrics") else 0
    pend = sched.pending_count()
    active_res = _active_reservations(qpus, now)
    print(f"[DEMO] completed={completed} / total_jobs={len(wl)} submitted={next_idx} pending={pend} active_res={active_res}")
    if stop_reason:
        print("[DEMO] stop_reason:", stop_reason)

    try:
        import qdc_sched.core.executor as executor_mod
        if hasattr(executor_mod, "_async_qpu_pool"):
            print("\n[DEMO] Scheduler finished in real-time! Waiting for background `SamplerV2` quantum simulations to complete mathematically...")
            executor_mod._async_qpu_pool.shutdown(wait=True)
    except Exception:
        pass

    _print_histogram(plan_ctr)

    outdir = os.getenv("QDC_OUTDIR", "results/demo_workload")
    _export_results(sched, outdir=outdir)
    print("\nDone.")


def main():
    full_eval = os.getenv("QDC_FULL_EVAL", "0") == "1"
    run_workload(full_eval=full_eval)


if __name__ == "__main__":
    main()