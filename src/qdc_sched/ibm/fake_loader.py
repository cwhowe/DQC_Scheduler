from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import warnings
import importlib
import inspect
import pkgutil

from qiskit_aer.noise import NoiseModel

def _backend_id(b) -> str:
    # Qiskit fake backends expose name() / backend_name
    if hasattr(b, "name"):
        try:
            return str(b.name())
        except Exception:
            pass
    if hasattr(b, "backend_name"):
        try:
            return str(b.backend_name)
        except Exception:
            pass
    return b.__class__.__name__

def _discover_fake_backends() -> List:
    """Discover Fake backends from qiskit_ibm_runtime.fake_provider across versions.

    We avoid relying on FakeProvider (which is missing in some qiskit-ibm-runtime builds)
    by walking modules under qiskit_ibm_runtime.fake_provider and instantiating classes
    named Fake* that look like Qiskit backends.
    """
    try:
        import qiskit_ibm_runtime.fake_provider as fp
    except Exception as e:
        raise ImportError("qiskit_ibm_runtime.fake_provider not available") from e

    BackendBase = None
    try:
        from qiskit.providers.backend import BackendV2 as BackendBase
    except Exception:
        try:
            from qiskit.providers.backend import Backend as BackendBase
        except Exception:
            BackendBase = None

    found: List = []
    for m in pkgutil.walk_packages(fp.__path__, fp.__name__ + "."):
        modname = m.name
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        for attr in dir(mod):
            if not attr.startswith("Fake"):
                continue
            obj = getattr(mod, attr, None)
            if not inspect.isclass(obj):
                continue
            try:
                inst = obj()
            except Exception:
                continue

            if BackendBase is None:
                if hasattr(inst, "configuration") and hasattr(inst, "properties"):
                    found.append(inst)
            else:
                try:
                    if isinstance(inst, BackendBase):
                        found.append(inst)
                except Exception:
                    pass

    # Deduplicate by backend id
    uniq: Dict[str, object] = {}
    for b in found:
        uniq[_backend_id(b)] = b
    return list(uniq.values())


def load_fake_backends(max_backends: int = 3, prefer: Optional[List[str]] = None):
    """Load a set of IBM fake backends from qiskit-ibm-runtime.

    If 'prefer' is provided, it is a list of substrings; backends whose names contain
    those substrings are ranked earlier (case-insensitive).
    """
    backends = _discover_fake_backends()
    if not backends:
        raise ImportError(
            "Could not discover any Fake backends in qiskit_ibm_runtime.fake_provider. "
            "The installed qiskit-ibm-runtime may not include fakes."
        )

    if prefer:
        prefer_l = [p.lower() for p in prefer]

        def key(b):
            name = _backend_id(b).lower()
            hits = [i for i, s in enumerate(prefer_l) if s in name]
            best = min(hits) if hits else 10**9
            return (best, name)

        backends.sort(key=key)
    else:
        # If no preference, pick larger devices first to get diversity.
        def numq(b) -> int:
            try:
                return int(b.configuration().num_qubits)
            except Exception:
                return 0
        backends.sort(key=lambda b: (-numq(b), _backend_id(b)))

    return backends[: int(max_backends)]

def build_aer_noise_models(fake_backends) -> Dict[str, NoiseModel]:
    from qiskit_aer.noise import NoiseModel
    models = {}
    for b in fake_backends:
        try:
            nm = NoiseModel.from_backend(b)
        except Exception as e:
            warnings.warn(f"Could not build Aer noise model for {b}: {e}")
            continue
        models[_backend_id(b)] = nm
    return models


# ---------------------------
# Fallback local QPU set
# ---------------------------
from dataclasses import dataclass
import networkx as nx

def _make_line_graph(n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n-1):
        G.add_edge(i, i+1)
    return G

def _make_grid_graph(r: int, c: int) -> nx.Graph:
    G = nx.grid_2d_graph(r, c)
    # relabel to 0..N-1
    mapping = {node: i for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping)


def make_default_qpu_set(*args, base_queue_delay_s: float = 0.0, **kwargs) -> tuple[dict, dict]:
    """Return (qpus, noise_models) for a small, deterministic local testbed.

    Provides multiple QPUs so Plan C can be exercised even if IBM fakes are unavailable.
    Accepts extra args/kwargs for compatibility with callers
    """
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.errors import depolarizing_error
    from qdc_sched.core.hardware import HardwareProfile, QPUState

    qpus = {}
    noise_models = {}

    def _mk_line(n: int):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n - 1):
            G.add_edge(i, i + 1)
        return G

    def _mk_grid(r: int, c: int):
        import networkx as nx
        G = nx.grid_2d_graph(r, c)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        return nx.relabel_nodes(G, mapping)

    configs = [
        # (id, graph, oneq_err, twoq_err, readout_err, t1q, t2q, tmeas)
        ('qpu_A', _mk_line(7), 1.2e-3, 2.4e-2, 2.5e-2, 40e-9, 320e-9, 1_100e-9),
        ('qpu_B', _mk_grid(3, 3), 1.0e-3, 2.0e-2, 2.0e-2, 35e-9, 280e-9, 1_000e-9),
        ('qpu_C', _mk_line(11), 0.9e-3, 1.6e-2, 1.8e-2, 30e-9, 240e-9, 900e-9),
    ]

    for qid, G, oneq_err, twoq_err_scalar, ro_err, t1q, t2q, tmeas in configs:
        prof = HardwareProfile(
            qpu_id=qid,
            num_qubits=len(G.nodes),
            coupling_graph=G,
            base_queue_delay_s=float(base_queue_delay_s),
            oneq_gate_time_s=float(t1q),
            twoq_gate_time_s=float(t2q),
            meas_time_s=float(tmeas),
            oneq_error=float(oneq_err),
            twoq_error=float(twoq_err_scalar),
            readout_error=float(ro_err),
        )
        qpus[qid] = QPUState(profile=prof)

        nm = NoiseModel()
        e1 = depolarizing_error(float(oneq_err), 1)
        e2 = depolarizing_error(float(twoq_err_scalar), 2)
        nm.add_all_qubit_quantum_error(e1, ["rz", "sx", "x"])
        nm.add_all_qubit_quantum_error(e2, ["cx"])
        noise_models[qid] = nm

    return qpus, noise_models


# ---------------------------
# IBM Fake backends that maka up QPU set
# ---------------------------
import networkx as nx

def _coupling_graph_from_backend(b) -> nx.Graph:
    cfg = b.configuration()
    n = int(cfg.num_qubits)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    cmap = getattr(cfg, "coupling_map", None) or []
    for e in cmap:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            G.add_edge(int(e[0]), int(e[1]))
    return G

def _avg_gate_metrics_from_properties(b) -> Tuple[float, float, float, float, float]:
    """Return (oneq_err, twoq_err, readout_err, oneq_time_s, twoq_time_s)."""
    oneq_err = 1.2e-3
    twoq_err = 2.0e-2
    readout_err = 2.0e-2
    oneq_time_s = 40e-9
    twoq_time_s = 300e-9

    try:
        props = b.properties()
        if props is None:
            return oneq_err, twoq_err, readout_err, oneq_time_s, twoq_time_s

        ro = []
        for q in range(len(props.qubits)):
            for item in props.qubits[q]:
                if getattr(item, "name", "") == "readout_error":
                    ro.append(float(item.value))
        if ro:
            readout_err = float(sum(ro) / len(ro))

        e1, t1, e2, t2 = [], [], [], []
        for g in props.gates:
            name = getattr(g, "gate", None) or getattr(g, "name", "")
            params = {p.name: p.value for p in getattr(g, "parameters", [])}

            if "gate_error" in params:
                if name in ("rz", "sx", "x", "u", "u1", "u2", "u3"):
                    e1.append(float(params["gate_error"]))
                elif name in ("cx", "ecr", "cz"):
                    e2.append(float(params["gate_error"]))

            if "gate_length" in params:
                if name in ("rz", "sx", "x", "u", "u1", "u2", "u3"):
                    t1.append(float(params["gate_length"]))
                elif name in ("cx", "ecr", "cz"):
                    t2.append(float(params["gate_length"]))

        if e1:
            oneq_err = float(sum(e1) / len(e1))
        if e2:
            twoq_err = float(sum(e2) / len(e2))
        if t1:
            oneq_time_s = float(sum(t1) / len(t1))
        if t2:
            twoq_time_s = float(sum(t2) / len(t2))

    except Exception as e:
        warnings.warn(f"Could not read backend properties for metrics: {e}")

    return oneq_err, twoq_err, readout_err, oneq_time_s, twoq_time_s


def make_ibm_fake_qpu_set(
    *,
    max_backends: int = 3,
    prefer: Optional[List[str]] = None,
    base_queue_delay_s: float = 0.0,
) -> Tuple[Dict[str, 'QPUState'], Dict[str, NoiseModel]]:
    """Return (qpus, noise_models) from qiskit-ibm-runtime FakeProvider backends."""
    from qdc_sched.core.hardware import HardwareProfile, QPUState

    fakes = load_fake_backends(max_backends=max_backends, prefer=prefer)
    noise_models = build_aer_noise_models(fakes)

    qpus: Dict[str, QPUState] = {}
    for b in fakes:
        bid = _backend_id(b)
        G = _coupling_graph_from_backend(b)
        n = len(G.nodes)

        oneq_err, twoq_err, ro_err, oneq_t, twoq_t = _avg_gate_metrics_from_properties(b)

        prof = HardwareProfile(
            qpu_id=bid,
            num_qubits=int(n),
            coupling_graph=G,
            base_queue_delay_s=float(base_queue_delay_s),
            oneq_gate_time_s=float(oneq_t),
            twoq_gate_time_s=float(twoq_t),
            meas_time_s=1_000e-9,
            oneq_error=float(oneq_err),
            twoq_error=float(twoq_err),
            readout_error=float(ro_err),
        )
        qpus[bid] = QPUState(profile=prof)

        if bid not in noise_models:
            noise_models[bid] = NoiseModel()

    return qpus, noise_models