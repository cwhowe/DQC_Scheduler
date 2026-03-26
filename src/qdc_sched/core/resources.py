from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import heapq


@dataclass
class WorkerPool:
    """Simple finite worker pool with earliest-available scheduling."""
    pool_id: str
    n_workers: int
    _available: List[Tuple[float, int]] = field(init=False, repr=False)

    def __post_init__(self):
        n = max(1, int(self.n_workers or 1))
        self.n_workers = n
        self._available = [(0.0, i) for i in range(n)]
        heapq.heapify(self._available)

    def reserve(self, ready_time: float, duration_s: float) -> tuple[int, float, float, float]:
        ready = float(ready_time or 0.0)
        dur = max(0.0, float(duration_s or 0.0))
        avail_t, worker_idx = heapq.heappop(self._available)
        start_t = max(ready, float(avail_t))
        end_t = start_t + dur
        heapq.heappush(self._available, (end_t, worker_idx))
        queue_s = max(0.0, start_t - ready)
        return int(worker_idx), float(start_t), float(end_t), float(queue_s)