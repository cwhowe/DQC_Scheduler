from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from qiskit import transpile


@dataclass(frozen=True)
class BackendTimingProfile:
    backend_name: str
    shots: int
    scheduling_method: Optional[str]
    transpiled_depth: int
    count_ops: Dict[str, int]
    duration_source: str
    per_shot_duration_s: float
    total_duration_s: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ProfileBuildConfig:
    optimization_level: int = 1
    scheduling_method: Optional[str] = "asap"
    layout_method: Optional[str] = None
    routing_method: Optional[str] = None


class BackendProfileBuilder:
    def __init__(self, config: Optional[ProfileBuildConfig] = None):
        self.config = config or ProfileBuildConfig()

    def build_profile(self, circuit: Any, backend: Any, *, shots: int) -> BackendTimingProfile:
        tqc = transpile(
            circuit,
            backend=backend,
            optimization_level=int(self.config.optimization_level),
            scheduling_method=self.config.scheduling_method,
            layout_method=self.config.layout_method,
            routing_method=self.config.routing_method,
        )

        per_shot_s, meta = self._estimate_per_shot_duration_s(tqc, backend)
        bname = self._backend_name(backend)
        return BackendTimingProfile(
            backend_name=bname,
            shots=int(shots or 0),
            scheduling_method=self.config.scheduling_method,
            transpiled_depth=int(getattr(tqc, "depth", lambda: 0)() or 0),
            count_ops=dict(getattr(tqc, "count_ops", lambda: {})() or {}),
            duration_source=str(meta.get("duration_source", "unknown")),
            per_shot_duration_s=float(per_shot_s),
            total_duration_s=float(max(0, int(shots or 0)) * per_shot_s),
            metadata=dict(meta),
        )

    def _backend_name(self, backend: Any) -> str:
        name_attr = getattr(backend, "name", None)
        if callable(name_attr):
            try:
                return str(name_attr())
            except Exception:
                pass
        if isinstance(name_attr, str):
            return name_attr
        return backend.__class__.__name__

    def _estimate_per_shot_duration_s(self, tqc: Any, backend: Any) -> tuple[float, Dict[str, Any]]:
        target = getattr(backend, "target", None)
        dt = getattr(target, "dt", None)

        circ_duration = getattr(tqc, "duration", None)
        if circ_duration is not None and dt is not None:
            return float(circ_duration) * float(dt), {
                "duration_source": "scheduled_circuit_duration",
                "dt": float(dt),
            }

        total_s = 0.0
        used_fallback = False
        for inst, qargs, _ in getattr(tqc, "data", []):
            inst_name = getattr(inst, "name", "")
            qidx = tuple(int(getattr(q, "_index", getattr(q, "index", 0))) for q in (qargs or []))
            dur_s = None

            if target is not None:
                try:
                    op_entry = target[inst_name]
                    props = None
                    if hasattr(op_entry, "get"):
                        props = op_entry.get(qidx)
                        if props is None and len(qidx) == 1:
                            props = op_entry.get((qidx[0],))
                    if props is not None:
                        dur = getattr(props, "duration", None)
                        unit = getattr(props, "unit", None)
                        if dur is not None:
                            if unit == "s":
                                dur_s = float(dur)
                            elif dt is not None:
                                dur_s = float(dur) * float(dt)
                except Exception:
                    pass

            if dur_s is None:
                used_fallback = True
                dur_s = 0.0
            total_s += float(dur_s)

        return float(total_s), {
            "duration_source": "sum_instruction_durations",
            "dt": (float(dt) if dt is not None else None),
            "used_zero_fallback_for_missing_instr": bool(used_fallback),
        }