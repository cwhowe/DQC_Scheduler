from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

Label = str


class PartitionAssignmentPolicy:
    """Map cut-partition labels to QPU ids."""

    def assign(
        self,
        labels: List[Label],
        qpu_ids: List[str],
        *,
        label_costs: Dict[Label, float],
        qpu_caps: Dict[str, int],
        qpu_quality: Dict[str, float],
        qpu_pred_time: Dict[str, float],
    ) -> Dict[Label, str]:
        raise NotImplementedError


@dataclass(frozen=True)
class RoundRobinAssignment(PartitionAssignmentPolicy):
    def assign(
        self,
        labels: List[Label],
        qpu_ids: List[str],
        *,
        label_costs: Dict[Label, float],
        qpu_caps: Dict[str, int],
        qpu_quality: Dict[str, float],
        qpu_pred_time: Dict[str, float],
    ) -> Dict[Label, str]:
        if not qpu_ids:
            raise ValueError("qpu_ids is empty")
        return {lab: qpu_ids[i % len(qpu_ids)] for i, lab in enumerate(labels)}


@dataclass(frozen=True)
class FitCutGreedyAssignment(PartitionAssignmentPolicy):
    """Greedy heuristic using (capacity, quality, time) scoring."""

    w_cap: float = 0.5
    w_qual: float = 0.4
    w_time: float = 0.1

    def assign(
        self,
        labels: List[Label],
        qpu_ids: List[str],
        *,
        label_costs: Dict[Label, float],
        qpu_caps: Dict[str, int],
        qpu_quality: Dict[str, float],
        qpu_pred_time: Dict[str, float],
    ) -> Dict[Label, str]:
        if not qpu_ids:
            raise ValueError("qpu_ids is empty")

        caps = [max(0, int(qpu_caps.get(q, 0))) for q in qpu_ids]
        cap_max = max(caps) if caps else 1

        times = [float(qpu_pred_time.get(q, 0.0)) for q in qpu_ids]
        t_max = max(times) if times else 1.0

        def score(q: str) -> float:
            cap = max(0, int(qpu_caps.get(q, 0)))
            qual = float(qpu_quality.get(q, 0.0))
            t = float(qpu_pred_time.get(q, 0.0))
            cap_norm = (cap / cap_max) if cap_max > 0 else 0.0
            t_norm = (t / t_max) if t_max > 0 else 0.0
            return self.w_cap * cap_norm + self.w_qual * qual - self.w_time * t_norm

        labs = sorted(labels, key=lambda l: float(label_costs.get(l, 1.0)), reverse=True)
        q_sorted = sorted(qpu_ids, key=score, reverse=True)

        # Load-aware greedy: after assigning a label to a QPU, reduce its effective capacity by 1
        eff_caps = {q: int(qpu_caps.get(q, 0)) for q in qpu_ids}
        assignment: Dict[Label, str] = {}
        for lab in labs:
            # re-rank QPUs each step using updated eff_caps
            def score_eff(q: str) -> float:
                cap = max(0, int(eff_caps.get(q, 0)))
                qual = float(qpu_quality.get(q, 0.0))
                t = float(qpu_pred_time.get(q, 0.0))
                cap_norm = (cap / cap_max) if cap_max > 0 else 0.0
                t_norm = (t / t_max) if t_max > 0 else 0.0
                return self.w_cap * cap_norm + self.w_qual * qual - self.w_time * t_norm
            q_sorted = sorted(qpu_ids, key=score_eff, reverse=True)
            pick = q_sorted[0]
            assignment[lab] = pick
            eff_caps[pick] = max(0, eff_caps[pick] - 1)
        return assignment


@dataclass(frozen=True)
class MinMakespanGreedyAssignment(PartitionAssignmentPolicy):
    """Greedy assignment to minimize makespan using per-label predicted runtimes.

    If label_qpu_pred_time is provided in kwargs, it is used:
      label_qpu_pred_time[label][qpu] -> seconds

    Otherwise falls back to (label_costs * qpu_pred_time[qpu]) proxy.
    """
    w_quality: float = 0.05  # tie-breaker favoring higher-quality QPUs

    def assign(
        self,
        labels: List[Label],
        qpu_ids: List[str],
        *,
        label_costs: Dict[Label, float],
        qpu_caps: Dict[str, int],
        qpu_quality: Dict[str, float],
        qpu_pred_time: Dict[str, float],
        **kwargs,
    ) -> Dict[Label, str]:
        if not qpu_ids:
            raise ValueError("qpu_ids is empty")

        label_qpu_pred_time = kwargs.get("label_qpu_pred_time", None)

        # Current load per QPU (predicted seconds queued for labels assigned)
        load = {q: 0.0 for q in qpu_ids}
        eff_caps = {q: int(qpu_caps.get(q, 0)) for q in qpu_ids}

        labs = sorted(labels, key=lambda l: float(label_costs.get(l, 1.0)), reverse=True)
        assignment: Dict[Label, str] = {}

        def pred(label: Label, q: str) -> float:
            if label_qpu_pred_time is not None:
                try:
                    return float(label_qpu_pred_time[label][q])
                except Exception:
                    pass
            # fallback proxy
            return float(label_costs.get(label, 1.0)) * float(qpu_pred_time.get(q, 0.0))

        for lab in labs:
            best_q = None
            best_score = None
            for q in qpu_ids:
                if eff_caps.get(q, 0) <= 0:
                    continue
                makespan = load[q] + pred(lab, q)
                # subtract small quality tie-breaker (higher quality => smaller score)
                score = makespan - self.w_quality * float(qpu_quality.get(q, 0.0))
                if best_score is None or score < best_score:
                    best_score = score
                    best_q = q
            if best_q is None:
                # no capacity left; fall back to best quality
                best_q = max(qpu_ids, key=lambda qq: float(qpu_quality.get(qq, 0.0)))
            assignment[lab] = best_q
            load[best_q] = load.get(best_q, 0.0) + pred(lab, best_q)
            eff_caps[best_q] = max(0, eff_caps.get(best_q, 0) - 1)

        return assignment
