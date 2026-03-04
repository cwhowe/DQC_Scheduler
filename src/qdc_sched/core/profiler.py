from __future__ import annotations
from typing import Dict, List, Tuple
import math
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from .types import CircuitProfile
from .hardware import QPUState

def _interaction_graph(qc: QuantumCircuit) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(qc.num_qubits))
    for inst, qargs, _ in qc.data:
        if len(qargs) == 2:
            a = qc.find_bit(qargs[0]).index
            b = qc.find_bit(qargs[1]).index
            if g.has_edge(a,b):
                g[a][b]['w'] += 1
            else:
                g.add_edge(a,b,w=1)
    return g

def _density(g: nx.Graph) -> float:
    n = g.number_of_nodes()
    if n <= 1:
        return 0.0
    return 2.0 * g.number_of_edges() / (n*(n-1))

def profile_circuit(qc: QuantumCircuit) -> CircuitProfile:
    dag = circuit_to_dag(qc)
    depth = qc.depth() or 0
    oneq = 0
    twoq = 0
    meas = 0
    for node in dag.op_nodes():
        nq = node.op.num_qubits
        if node.op.name == "measure":
            meas += 1
        elif nq == 1:
            oneq += 1
        elif nq == 2:
            twoq += 1
    ig = _interaction_graph(qc)
    dens = _density(ig)

    # Heuristic cut suitability: refine later / ML later
    # - very dense interaction tends to be worse for cutting; sparse/modular tends to be better
    if qc.num_qubits <= 3:
        suit = "bad"  # small circuits: cutting rarely helps
    elif dens > 0.6 and twoq > 30:
        suit = "bad"
    elif dens < 0.25 and twoq > 10:
        suit = "good"
    else:
        suit = "neutral"

    features = {
        "width": float(qc.num_qubits),
        "depth": float(depth),
        "twoq": float(twoq),
        "dens": float(dens),
    }
    return CircuitProfile(
        width=qc.num_qubits,
        depth=depth,
        twoq_count=twoq,
        oneq_count=oneq,
        meas_count=meas,
        interaction_density=dens,
        cut_suitability=suit,
        features=features,
    )

def rank_qpus(profile: CircuitProfile, qpus: Dict[str, QPUState], quality_proxy_fn) -> List[Tuple[str, float, float, float]]:
    ranked = []
    for qpu_id, st in qpus.items():
        if profile.width > st.profile.num_qubits:
            continue
        t = st.profile.base_queue_delay_s + profile.oneq_count*st.profile.t_1q + profile.twoq_count*st.profile.t_2q + profile.meas_count*st.profile.t_meas
        fid = quality_proxy_fn(qpu_id, profile)
        # lower score is better
        score = t + (1.0 - fid) * 1.0
        ranked.append((qpu_id, score, t, fid))
    ranked.sort(key=lambda x: x[1])
    return ranked
