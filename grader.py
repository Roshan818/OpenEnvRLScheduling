"""
Graders for Smart Factory Scheduling tasks.

Each public function accepts an optional state argument:
  - Called with no argument  → runs a deterministic heuristic episode and returns a score.
  - Called with a dict/object → scores that state directly.

All functions return a float strictly in (0.0, 1.0).
No external dependencies — pure Python stdlib only.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


# ── Minimal in-process environment ───────────────────────────────────────────

TASK_CONFIGS = {
    "easy":   {"num_machines": 2, "num_jobs": 3,  "failure_rate": 0.00, "max_steps": 20,
                "job_time_range": (2, 4), "deadline_slack": (2, 5), "max_priority": 1},
    "medium": {"num_machines": 4, "num_jobs": 7,  "failure_rate": 0.08, "max_steps": 30,
                "job_time_range": (2, 5), "deadline_slack": (2, 6), "max_priority": 2},
    "hard":   {"num_machines": 6, "num_jobs": 12, "failure_rate": 0.15, "max_steps": 40,
                "job_time_range": (2, 6), "deadline_slack": (1, 5), "max_priority": 3},
}


class _Machine:
    __slots__ = ("id", "status", "current_job", "failure_rate")

    def __init__(self, mid: str, failure_rate: float) -> None:
        self.id          = mid
        self.status      = "idle"
        self.current_job: Optional[str] = None
        self.failure_rate = failure_rate


class _Job:
    __slots__ = ("id", "remaining_time", "deadline", "priority", "assigned_machine")

    def __init__(self, jid: str, remaining_time: int, deadline: int, priority: int) -> None:
        self.id               = jid
        self.remaining_time   = remaining_time
        self.deadline         = deadline
        self.priority         = priority
        self.assigned_machine: Optional[str] = None


class _MiniEnv:
    """Minimal pure-Python factory simulation — no external dependencies."""

    def __init__(self, task: str, seed: int = 42) -> None:
        cfg = TASK_CONFIGS[task]
        rng = random.Random(seed)
        self.task      = task
        self.time      = 0
        self.max_steps = cfg["max_steps"]
        self.late_jobs = 0
        self.machines: List[_Machine] = [
            _Machine(f"M{i+1}", cfg["failure_rate"])
            for i in range(cfg["num_machines"])
        ]
        self.jobs: List[_Job] = []
        for i in range(cfg["num_jobs"]):
            pt       = rng.randint(*cfg["job_time_range"])
            deadline = pt + rng.randint(*cfg["deadline_slack"])
            priority = rng.randint(1, cfg["max_priority"])
            self.jobs.append(_Job(f"J{i+1}", pt, deadline, priority))
        self.completed_jobs: List[_Job] = []
        self._rng = rng

    # ── heuristic action ──────────────────────────────────────────────────────
    def _heuristic(self):
        for m in self.machines:
            if m.status == "broken":
                return ("repair", m.id, None)
        for j in sorted(self.jobs, key=lambda x: (x.deadline, -x.priority)):
            for m in self.machines:
                if m.status == "idle":
                    return ("assign", j.id, m.id)
        return ("wait", None, None)

    def step(self, action) -> bool:
        kind, arg1, arg2 = action
        if kind == "assign":
            job     = next((j for j in self.jobs if j.id == arg1), None)
            machine = next((m for m in self.machines if m.id == arg2), None)
            if job and machine and machine.status == "idle":
                job.assigned_machine = machine.id
                machine.status       = "busy"
                machine.current_job  = job.id
        elif kind == "repair":
            machine = next((m for m in self.machines if m.id == arg1), None)
            if machine and machine.status == "broken":
                machine.status = "idle"

        self.time += 1

        for m in self.machines:
            if m.status == "busy":
                job = next((j for j in self.jobs if j.id == m.current_job), None)
                if job:
                    job.remaining_time -= 1
                    if job.remaining_time <= 0:
                        if self.time > job.deadline:
                            self.late_jobs += 1
                        self.jobs.remove(job)
                        self.completed_jobs.append(job)
                        m.status      = "idle"
                        m.current_job = None

            if m.status == "busy" and m.failure_rate > 0:
                if self._rng.random() < m.failure_rate:
                    stalled = next((j for j in self.jobs if j.id == m.current_job), None)
                    if stalled:
                        stalled.assigned_machine = None
                    m.status      = "broken"
                    m.current_job = None

        done = self.time >= self.max_steps or len(self.jobs) == 0
        return done

    def run_heuristic(self) -> None:
        for _ in range(self.max_steps):
            action = self._heuristic()
            done   = self.step(action)
            if done:
                break


# ── Score computation ─────────────────────────────────────────────────────────

def _compute(completed: int, on_time: int, total: int, late: int) -> float:
    if total == 0:
        return 0.001
    score = (
        0.5 * (completed / total)
        + 0.3 * (on_time / max(completed, 1))
        + 0.2 * max(0.0, 1.0 - late / max(completed, 1))
    )
    return round(max(0.001, min(0.999, score)), 4)


def _score_from_obj(obj: Any) -> float:
    """Score from an env object (has .completed_jobs, .jobs/.pending_jobs, .time, .late_jobs)."""
    if isinstance(obj, dict):
        done_list = obj.get("completed_jobs", []) or []
        pend_list = obj.get("pending_jobs", []) or []
        late      = int(obj.get("late_jobs", 0) or 0)
        t         = int(obj.get("time", 0) or 0)
        completed = len(done_list)
        total     = completed + len(pend_list)
        on_time   = sum(
            1 for j in done_list
            if (j.get("deadline", 0) if isinstance(j, dict) else getattr(j, "deadline", 0)) >= t
        )
    else:
        done_list = list(getattr(obj, "completed_jobs", []) or [])
        pend_list = list(getattr(obj, "jobs", getattr(obj, "pending_jobs", [])) or [])
        late      = int(getattr(obj, "late_jobs", 0) or 0)
        t         = int(getattr(obj, "time", 0) or 0)
        completed = len(done_list)
        total     = completed + len(pend_list)
        on_time   = sum(
            1 for j in done_list
            if getattr(j, "deadline", 0) >= t
        )
    return _compute(completed, on_time, total, late)


def _run_episode(task: str) -> float:
    env = _MiniEnv(task, seed=42)
    env.run_heuristic()
    completed = len(env.completed_jobs)
    total     = completed + len(env.jobs)
    on_time   = max(0, completed - env.late_jobs)   # on-time = completed minus late
    return _compute(completed, on_time, total, env.late_jobs)


# ── Public graders ────────────────────────────────────────────────────────────

def score_easy(state_or_env=None) -> float:
    """Grade an easy-task episode (2 machines, 3 jobs, no failures).
    Returns a float in (0.0, 1.0). Called with no argument for baseline scoring."""
    if state_or_env is not None:
        return _score_from_obj(state_or_env)
    return _run_episode("easy")


def score_medium(state_or_env=None) -> float:
    """Grade a medium-task episode (4 machines, 7 jobs, 8% failures).
    Returns a float in (0.0, 1.0). Called with no argument for baseline scoring."""
    if state_or_env is not None:
        return _score_from_obj(state_or_env)
    return _run_episode("medium")


def score_hard(state_or_env=None) -> float:
    """Grade a hard-task episode (6 machines, 12 jobs, 15% failures).
    Returns a float in (0.0, 1.0). Called with no argument for baseline scoring."""
    if state_or_env is not None:
        return _score_from_obj(state_or_env)
    return _run_episode("hard")
