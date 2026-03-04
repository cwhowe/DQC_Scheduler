from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time
import os
import math

from .types import Job, RunToggles, Task, TaskGraph
from .hardware import QPUState
from .profiler import profile_circuit, rank_qpus
from .planner import Planner, PlannerConfig
from .quality import QualityModel
from .executor import Executor, ExecConfig
from .metrics import MetricsRecorder


@dataclass
class SchedulerConfig:
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    exec_cfg: ExecConfig = field(default_factory=ExecConfig)
    max_pending_attempts_per_tick: int = 32
    pending_fifo: bool = True  # if False, re-sort by job.priority desc each tick


@dataclass
class _PendingItem:
    job: Job
    toggles: RunToggles
    prof: Any  # CircuitProfile
    # NOTE: ranked is recomputed on each attempt (depends on current availability / quality heuristics)
    enqueue_time_s: float
    attempts: int = 0


class Scheduler:
    def __init__(self, qpus: Dict[str, QPUState], quality: QualityModel, cfg: SchedulerConfig):
        self.qpus = qpus
        self.quality = quality
        self.metrics = MetricsRecorder()
        self.planner = Planner(qpus, quality, cfg.planner)
        self.executor = Executor(qpus, quality, self.metrics, cfg.exec_cfg)
        self.cfg = cfg
        self.now_s = 0.0
        self._pending: List[_PendingItem] = []
        self.task_log: List[Task] = []
        self._task_ids_seen: set[str] = set()
        self._pending_retry_not_before: Dict[str, float] = {}
        self._pending_last_fail_reason: Dict[str, str] = {}

    def _maybe_log_tasks_from_record(self, rec) -> None:
        """Best-effort append Task events from a JobRunRecord into self.task_log.

        Contract:
          - Prefer det['tasks'] if present; otherwise fall back to det['task_graph'].tasks.
          - Deduplicate by Task.task_id to avoid double-logging when both are present.
        """
        try:
            det = getattr(rec, "details", None)
            if not isinstance(det, dict):
                return

            tasks = None
            t = det.get("tasks", None)
            if isinstance(t, list) and t:
                tasks = t
            else:
                tg = det.get("task_graph", None)
                tgs = getattr(tg, "tasks", None)
                if isinstance(tgs, list) and tgs:
                    tasks = tgs

            if not tasks:
                return

            for task in tasks:
                tid = getattr(task, "task_id", None)
                if tid is None:
                    continue
                if tid in self._task_ids_seen:
                    continue
                self._task_ids_seen.add(tid)
                self.task_log.append(task)
        except Exception:
            return

    def reserve_task(self, task: Task) -> None:
        """Reserve qubits for a Task on its assigned QPU (if any).

        Note: task logging is handled separately via _maybe_log_tasks_from_record to avoid duplicates.
        """

        if task.qpu_id is None or task.qubits is None:
            return

        qpu = self.qpus.get(task.qpu_id)
        if qpu is None:
            return

        dur = float(task.end_s) - float(task.start_s)
        if dur < 0.0:
            dur = 0.0

        qpu.reserve(
            job_id=task.job_id,
            qubits=set(task.qubits),
            start_s=float(task.start_s),
            duration_s=float(dur),
        )

    def _pending_retry_cooldown_s(self) -> float:
        try:
            return max(0.0, float(os.getenv("QDC_PENDING_RETRY_COOLDOWN_S", "0.5")))
        except Exception:
            return 0.5

    def _wait_replan_backoff_s(self) -> float:
        """Backoff after a D_WAIT plan to avoid re-planning the same blocked jobs every tick.

        If QDC_WAIT_REPLAN_BACKOFF_S is not set, default to the generic pending retry cooldown.
        """
        raw = os.getenv("QDC_WAIT_REPLAN_BACKOFF_S", None)
        if raw is None:
            try:
                return self._pending_retry_cooldown_s()
            except Exception:
                return 0.5
        try:
            return max(0.0, float(raw))
        except Exception:
            try:
                return self._pending_retry_cooldown_s()
            except Exception:
                return 0.5

    def _set_pending_backoff(self, job_id: Optional[str], reason: str, delay_s: Optional[float] = None) -> None:
        if not job_id:
            return
        d = self._pending_retry_cooldown_s() if delay_s is None else max(0.0, float(delay_s))
        self._pending_retry_not_before[job_id] = max(self._pending_retry_not_before.get(job_id, 0.0), self.now_s + d)
        self._pending_last_fail_reason[job_id] = str(reason)

    def _plan_is_schedulable(self, plan: Any) -> tuple[bool, str]:
        if plan is None:
            return False, "plan_none"
        kind = getattr(plan, "kind", None)
        if not isinstance(kind, str):
            return False, "plan_missing_kind"
        if kind == "D_WAIT":
            return True, "wait"

        for attr in ("pred_total_s", "pred_exec_s", "pred_recon_s"):
            if hasattr(plan, attr):
                try:
                    v = getattr(plan, attr)
                    if v is not None and not math.isfinite(float(v)):
                        return False, f"{attr}_nonfinite"
                except Exception:
                    return False, f"{attr}_invalid"

        if kind in ("A_NO_CUT_SINGLE", "B_CUT_SINGLE_SEQ"):
            qid = getattr(plan, "qpu_id", None)
            if not qid:
                return False, "single_plan_missing_qpu_id"
            return True, "ok"

        if kind == "C_CUT_MULTI_QPU":
            det = getattr(plan, "details", None)
            if not isinstance(det, dict):
                return False, "c_missing_details"
            assignment = det.get("assignment", None) or det.get("label_to_qpu", None) or det.get("assignments", None)
            labels_used = det.get("labels_used", None) or det.get("labels", None)
            if assignment in (None, {}, []):
                return False, "c_missing_assignment"
            if labels_used in (None, [], {}):
                return True, "ok_no_labels"
            return True, "ok"

        # Unknown plan kinds: let executor decide, but avoid obvious malformed qpu references if present.
        return True, "ok"


    def _sched_debug_thrash(self) -> bool:
        try:
            return bool(int(os.getenv("QDC_SCHED_DEBUG_THRASH", "0")))
        except Exception:
            return False


    def tick(self, dt_s: float) -> None:
        """Advance simulated time, release completed qubits, then retry pending jobs."""
        self.now_s += float(dt_s)
        for st in self.qpus.values():
            st.release_completed(self.now_s)
        self.try_schedule_pending()

    def submit_and_try_schedule(self, job: Job, toggles: RunToggles) -> Any:
        """Submit a job.

        Returns: (out, rec, plan, prof)
          - If the chosen plan is D_WAIT, the job is queued and (out, rec) are None.
        """
        # Preserve earliest submit time if this job is re-submitted
        if getattr(job, "submit_time_s", 0.0) == 0.0:
            job.submit_time_s = self.now_s

        t_sched0 = time.perf_counter()

        prof = profile_circuit(job.circuit)
        ranked = rank_qpus(
            prof,
            self.qpus,
            lambda qid, p: self.quality.fidelity_proxy_from_profile(qid, p),
        )
        prof.ranked_qpus = ranked

        plan = self.planner.choose_plan(job, prof, self.now_s, ranked)
        t_sched1 = time.perf_counter()

        out = None
        rec = None
        if plan.kind != "D_WAIT":
            out, rec = self.executor.run_job_plan(job, plan, self.now_s, toggles)
            rec.t_schedule_s = (t_sched1 - t_sched0)
            rec.end_to_end_s = (time.perf_counter() - t_sched0)

            try:
                if rec.details is None or not isinstance(rec.details, dict):
                    rec.details = {}
                rec.details.setdefault("queue_wait_s", max(0.0, self.now_s - float(job.submit_time_s)))
            except Exception:
                pass

            self._maybe_log_tasks_from_record(rec)

            if (not getattr(toggles, "simulate_only", False)) and bool(getattr(self.cfg.exec_cfg, "reserve_nonsim", False)):
                try:
                    det = rec.details if isinstance(getattr(rec, "details", None), dict) else {}
                    if bool(det.get("reservations_applied", False)):
                        tasks = []
                    else:
                        tasks = det.get("tasks", [])
                    if isinstance(tasks, list):
                        for t in tasks:
                            try:
                                self.reserve_task(t)
                            except Exception:
                                pass
                except Exception:
                    pass

            return out, rec, plan, prof

        self._enqueue_pending(job, toggles, prof)
        return None, None, plan, prof

    def _enqueue_pending(self, job: Job, toggles: RunToggles, prof: Any) -> None:
        jid = getattr(job, "job_id", None)
        for it in self._pending:
            if getattr(it.job, "job_id", None) == jid and jid is not None:
                it.toggles = toggles
                it.prof = prof
                return
        self._pending.append(_PendingItem(job=job, toggles=toggles, prof=prof, enqueue_time_s=self.now_s))
        jid = getattr(job, "job_id", None)
        if jid is not None and jid not in self._pending_retry_not_before:
            self._pending_retry_not_before[jid] = 0.0

    def pending_count(self) -> int:
        return len(self._pending)


    def try_schedule_pending(self) -> List[Tuple[Job, Any, Any]]:
        """Try to schedule jobs that were previously returned as D_WAIT.

        Returns a list of (job, plan, rec) for jobs that were successfully scheduled.
        Anti-thrash v2: do not re-attempt the same pending job more than once per pass.
        """
        if not self._pending:
            return []

        # Optionally reprioritize (priority: higher first); stable FIFO otherwise.
        if not self.cfg.pending_fifo:
            try:
                self._pending.sort(key=lambda it: float(getattr(it.job, "priority", 0.0)), reverse=True)
            except Exception:
                pass

        scheduled: List[Tuple[Job, Any, Any]] = []
        attempts_left = int(self.cfg.max_pending_attempts_per_tick)

        seen_this_pass: set[str] = set()
        no_progress_streak = 0

        # iterate over live list; remove scheduled items in-place.
        i = 0
        while i < len(self._pending) and attempts_left > 0:
            it = self._pending[i]
            attempts_left -= 1

            jid = getattr(it.job, "job_id", None)
            jid_key = str(jid) if jid is not None else f"_idx_{i}"

            # Never re-plan the same job twice in one pass.
            if jid_key in seen_this_pass:
                if self._sched_debug_thrash():
                    try:
                        self._emit_event("pending_skip_seen_this_pass", {"job_id": jid, "now_s": self.now_s})
                    except Exception:
                        pass
                i += 1
                continue
            seen_this_pass.add(jid_key)

            it.attempts += 1

            # Recompute ranking each attempt (availability + quality heuristics can change as time advances).
            ranked = rank_qpus(
                it.prof,
                self.qpus,
                lambda qid, p: self.quality.fidelity_proxy_from_profile(qid, p),
            )
            it.prof.ranked_qpus = ranked

            not_before = self._pending_retry_not_before.get(jid, 0.0) if jid is not None else 0.0
            if self.now_s < not_before:
                if self._sched_debug_thrash():
                    try:
                        self._emit_event("pending_backoff_skip", {"job_id": jid, "now_s": self.now_s, "not_before": not_before})
                    except Exception:
                        pass
                i += 1
                no_progress_streak += 1
                if no_progress_streak >= len(self._pending):
                    break
                continue

            plan = self.planner.choose_plan(it.job, it.prof, self.now_s, ranked)

            ok_plan, plan_reason = self._plan_is_schedulable(plan)
            if not ok_plan:
                self._set_pending_backoff(jid, plan_reason)
                if self._sched_debug_thrash():
                    try:
                        self._emit_event("pending_invalid_plan", {"job_id": jid, "reason": plan_reason})
                    except Exception:
                        pass
                i += 1
                no_progress_streak += 1
                if no_progress_streak >= len(self._pending):
                    break
                continue

            if plan.kind == "D_WAIT":
                wb = self._wait_replan_backoff_s()
                if wb > 0.0:
                    self._set_pending_backoff(jid, "wait", delay_s=wb)
                if self._sched_debug_thrash():
                    try:
                        self._emit_event("pending_wait_plan", {"job_id": jid, "backoff_s": wb})
                    except Exception:
                        pass
                i += 1
                no_progress_streak += 1
                if no_progress_streak >= len(self._pending):
                    break
                continue

            t_sched0 = time.perf_counter()
            try:
                out, rec = self.executor.run_job_plan(it.job, plan, self.now_s, it.toggles)
            except Exception as e:
                try:
                    self._emit_event("pending_schedule_error", {"job_id": it.job.job_id, "error": repr(e)})
                except Exception:
                    pass
                try:
                    self.pending_error[it.job.job_id] = repr(e)
                except Exception:
                    pass
                self._set_pending_backoff(getattr(it.job, "job_id", None), "executor_exception")
                if self._sched_debug_thrash():
                    try:
                        self._emit_event("pending_executor_exception", {"job_id": jid, "error": repr(e)})
                    except Exception:
                        pass
                i += 1
                no_progress_streak += 1
                if no_progress_streak >= len(self._pending):
                    break
                continue
            t_sched1 = time.perf_counter()

            if rec is None:
                self._set_pending_backoff(jid, "executor_returned_none")
                if self._sched_debug_thrash():
                    try:
                        self._emit_event("pending_executor_none", {"job_id": jid})
                    except Exception:
                        pass
                i += 1
                no_progress_streak += 1
                if no_progress_streak >= len(self._pending):
                    break
                continue

            rec.t_schedule_s = (t_sched1 - t_sched0)
            self._maybe_log_tasks_from_record(rec)
            rec.end_to_end_s = (time.perf_counter() - t_sched0)

            try:
                if rec.details is None or not isinstance(rec.details, dict):
                    rec.details = {}
                rec.details["queue_wait_s"] = max(0.0, self.now_s - float(it.job.submit_time_s))
                rec.details["pending_attempts"] = int(it.attempts)
            except Exception:
                pass

            if (not getattr(it.toggles, "simulate_only", False)) and bool(getattr(self.cfg.exec_cfg, "reserve_nonsim", False)):
                try:
                    det = rec.details if isinstance(getattr(rec, "details", None), dict) else {}
                    if bool(det.get("reservations_applied", False)):
                        tasks = []
                    else:
                        tasks = det.get("tasks", [])
                    if isinstance(tasks, list):
                        for t in tasks:
                            try:
                                self.reserve_task(t)
                            except Exception:
                                pass
                except Exception:
                    pass

            scheduled.append((it.job, plan, rec))
            no_progress_streak = 0

            try:
                if jid is not None:
                    self._pending_retry_not_before.pop(jid, None)
                    self._pending_last_fail_reason.pop(jid, None)
            except Exception:
                pass

            self._pending.pop(i)

            if self._pending and all((str(getattr(p.job, "job_id", None)) if getattr(p.job, "job_id", None) is not None else None) in seen_this_pass for p in self._pending):
                break

        return scheduled

    def submit(self, job: Job, toggles: RunToggles) -> None:
        """Enqueue a job without immediately executing it."""
        # Preserve earliest submit time
        if getattr(job, "submit_time_s", 0.0) == 0.0:
            job.submit_time_s = self.now_s

        prof = profile_circuit(job.circuit)
        self._enqueue_pending(job, toggles, prof)

    def step(self, now_s: float, max_to_schedule: Optional[int] = None):
        """Advance time and attempt to schedule pending jobs.

        Returns list of job_ids that started execution.
        """
        self.now_s = float(now_s)

        for st in self.qpus.values():
            st.release_completed(self.now_s)

        old = None
        if max_to_schedule is None:
            try:
                max_to_schedule = int(os.getenv("QDC_MAX_DECISIONS_PER_STEP", "2"))
            except Exception:
                max_to_schedule = None
        if max_to_schedule is not None:
            old = self.cfg.max_pending_attempts_per_tick
            self.cfg.max_pending_attempts_per_tick = int(max_to_schedule)

        try:
            scheduled = self.try_schedule_pending()
        finally:
            if old is not None:
                self.cfg.max_pending_attempts_per_tick = old

        return [job.job_id for job, _, _ in scheduled]