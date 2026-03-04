from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from .types import Task, TaskGraph


@dataclass
class JobRunRecord:
    job_id: str
    qpu_id: Optional[str]
    plan_kind: str
    submit_time_s: float
    t_schedule_s: float
    t_partition_s: float
    t_mapping_s: float
    t_execution_s: float
    t_reconstruction_s: float
    end_to_end_s: float
    fidelity_proxy: float
    fidelity_estimated: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsRecorder:
    """Collect per-job records plus (optional) per-task event objects.

    The scheduler currently exports tasks from `sched.task_log`. While we work toward
    a single canonical event stream, collecting tasks here provides an easy fallback
    and makes it harder to accidentally "lose" task events during refactors.
    """

    def __init__(self):
        self.records: List[JobRunRecord] = []
        self.task_events: List[Task] = []

    def add(self, rec: JobRunRecord) -> None:
        self.records.append(rec)

        # Best-effort task extraction from rec.details
        try:
            d = rec.details or {}
        except Exception:
            d = {}

        tasks = None
        try:
            tasks = d.get("tasks", None)
        except Exception:
            tasks = None

        if isinstance(tasks, list) and tasks and isinstance(tasks[0], Task):
            self.task_events.extend(tasks)
            return

        tg = None
        try:
            tg = d.get("task_graph", None)
        except Exception:
            tg = None
        if isinstance(tg, TaskGraph):
            self.task_events.extend(list(tg.tasks))