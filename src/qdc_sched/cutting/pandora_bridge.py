from __future__ import annotations

import asyncio
import os
import sys
from typing import List, Optional

from qiskit import QuantumCircuit, transpile

# Pandora works with this primitive gate set. Circuits are decomposed to this
# basis before being loaded into the DB so all gate types are recognised.
_PANDORA_BASIS = ["h", "t", "tdg", "cx", "x", "s", "sdg", "rz", "rx"]


def _to_pandora_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    """Decompose to Pandora's supported gate set, stripping measurements."""
    stripped = QuantumCircuit(circuit.num_qubits)
    for inst, qargs, cargs in circuit.data:
        if inst.name == "measure":
            continue
        if cargs:
            continue
        stripped.append(inst, qargs)
    return transpile(stripped, basis_gates=_PANDORA_BASIS, optimization_level=0)


def _ensure_pandora_on_path() -> None:
    """Add PANDORA_SRC_PATH to sys.path and stub heavy optional submodules.

    Pandora has no pyproject.toml so it can't be pip-installed from git.
    Clone the repo and set PANDORA_SRC_PATH=/path/to/pandora/src instead.

    Several Pandora submodules import pyLIQTR/qualtran benchmarking code at
    module level. We stub those submodules before importing the ones we need
    (db.core, db.repository, db.service, translation.*).
    """
    import types

    src = os.environ.get("PANDORA_SRC_PATH", "").strip()
    if src and src not in sys.path:
        sys.path.insert(0, src)

    if not src:
        return

    def _stub(name: str, **attrs):
        if name not in sys.modules:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m

    # Top-level package — skip __init__.py which imports pyLIQTR
    if "pandora" not in sys.modules:
        top = types.ModuleType("pandora")
        top.__path__ = [os.path.join(src, "pandora")]
        top.__package__ = "pandora"
        sys.modules["pandora"] = top

    # parallel_decompose pulls in pyLIQTR; we never call it via the bridge
    _stub("pandora.multithreading")
    _stub("pandora.multithreading.parallel_decompose", worker_entry=None)

    # pandora.multithreading pulls in pyLIQTR; we never call it via the bridge
    # No cirq/qualtran stubs — those are real Pandora deps that must be installed.


def _pandora_importable() -> bool:
    """Check only the submodules the bridge actually uses."""
    _ensure_pandora_on_path()
    try:
        import importlib
        importlib.import_module("pandora.db.core")
        importlib.import_module("pandora.db.repository")
        importlib.import_module("pandora.db.service")
        return True
    except ImportError:
        return False


class PandoraBridge:
    """Synchronous wrapper around Pandora's async PostgreSQL-backed circuit optimizer.

    Pandora has no pip package — clone the repo and point PANDORA_SRC_PATH at
    its src/ directory. Transparent no-op when Pandora is not importable or the
    DB is unreachable; callers never need to guard against Pandora errors.
    """

    def __init__(self, config_path: str, nproc: int = 1, timeout_s: float = 30.0):
        self.config_path = config_path
        self.nproc = int(nproc)
        self.timeout_s = float(timeout_s)
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = bool(self.config_path) and _pandora_importable()
        return self._available

    def _run(self, coro):
        """Run a coroutine synchronously, returning None on any error."""
        try:
            return asyncio.run(asyncio.wait_for(coro, timeout=self.timeout_s))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Optimize: load → Pandora rewrites → export
    # ------------------------------------------------------------------

    async def _run_rewrite_passes(self, pool, pass_count: int = 5, timeout_s: int = 10) -> None:
        """Call the SQL cancellation procedures for adjacent inverse gate pairs.

        Gate type IDs from PandoraGateTranslator:
          T=29, Tdg=30, H=8, S=27, Sdg=28, X=9, CX=18

        Params are queried from the DB to avoid hardcoding Pandora's internal
        encoding (half-turns vs radians etc.).
        """
        # Adjacent-pair cancellations: (type_a, type_b, procedure)
        cancel_pairs = [
            (29, 30, "cancel_single_qubit"),   # T / Tdg
            (30, 29, "cancel_single_qubit"),   # Tdg / T
            (27, 28, "cancel_single_qubit"),   # S / Sdg
            (28, 27, "cancel_single_qubit"),   # Sdg / S
            (8,  8,  "cancel_single_qubit"),   # H / H
            (9,  9,  "cancel_single_qubit"),   # X / X
            (18, 18, "cancel_two_qubit"),      # CX / CX
        ]
        async with pool.acquire() as conn:
            for type_a, type_b, proc in cancel_pairs:
                row_a = await conn.fetchrow(
                    "SELECT param FROM linked_circuit WHERE type=$1 LIMIT 1", type_a
                )
                row_b = await conn.fetchrow(
                    "SELECT param FROM linked_circuit WHERE type=$1 LIMIT 1", type_b
                )
                if row_a is None or row_b is None:
                    continue
                await conn.execute(
                    f"CALL {proc}($1, $2, $3, $4, $5, $6)",
                    type_a, type_b,
                    float(row_a["param"]), float(row_b["param"]),
                    pass_count, timeout_s,
                )

    async def _optimize_async(self, circuit: QuantumCircuit) -> Optional[QuantumCircuit]:
        from pandora.db.core import PandoraDB
        from pandora.db.repository import GateRepository
        from pandora.db.service import PandoraService

        db = PandoraDB(self.config_path)
        await db.connect()
        try:
            repo = GateRepository(db)
            service = PandoraService(db=db, repo=repo)
            await service.build_circuit(_to_pandora_basis(circuit))
            await self._run_rewrite_passes(db.pool)
            return await service.load_circuit("qiskit")
        finally:
            await db.close()

    def optimize(self, circuit: QuantumCircuit) -> Optional[QuantumCircuit]:
        """Return a gate-reduced circuit via Pandora SQL rewrites, or None on failure."""
        if not self.available:
            return None
        result = self._run(self._optimize_async(circuit))
        return result if isinstance(result, QuantumCircuit) else None

    # ------------------------------------------------------------------
    # Widgetize: load → widgetize async generator → list of subcircuits
    # ------------------------------------------------------------------

    async def _widgetize_async(
        self,
        circuit: QuantumCircuit,
        max_t: int,
        max_d: int,
        batch_size: int,
        add_gin_per_widget: bool,
    ) -> List[QuantumCircuit]:
        from pandora.db.core import PandoraDB
        from pandora.db.repository import GateRepository
        from pandora.db.service import PandoraService
        from pandora.translation.dag_to_circuit import pandora_to_circuit

        db = PandoraDB(self.config_path)
        await db.connect()
        try:
            repo = GateRepository(db)
            service = PandoraService(db=db, repo=repo)
            await service.build_circuit(_to_pandora_basis(circuit))

            subcircuits: List[QuantumCircuit] = []
            async for widget_gates in service.widgetize(max_t, max_d, batch_size, add_gin_per_widget):
                try:
                    qc = pandora_to_circuit(widget_gates, "qiskit")
                    if isinstance(qc, QuantumCircuit) and qc.num_qubits > 0:
                        subcircuits.append(qc)
                except Exception:
                    continue
            return subcircuits
        finally:
            await db.close()

    def widgetize(
        self,
        circuit: QuantumCircuit,
        max_t: int = 10,
        max_d: int = 10,
        batch_size: int = 1000,
        add_gin_per_widget: bool = True,
    ) -> List[QuantumCircuit]:
        """Return disjoint widget subcircuits via Pandora, or [] on failure."""
        if not self.available:
            return []
        result = self._run(
            self._widgetize_async(circuit, max_t, max_d, batch_size, add_gin_per_widget)
        )
        return result if isinstance(result, list) else []
