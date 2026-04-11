"""
Graders for Smart Factory Scheduling tasks.

Each public function:
  - Accepts an optional state/env argument to score a finished episode.
  - When called with no argument, runs a deterministic heuristic episode
    and returns the score.
  - Always returns a float strictly in (0.0, 1.0).

This module is fully self-contained (stdlib only) so it works in any
Python 3.8+ environment regardless of what packages are installed.
The simulation implements the exact same RL dynamics as FactoryEnv.
"""

from __future__ import annotations
import random


# ── Minimal RL simulation (identical dynamics to FactoryEnv) ─────────────────

TASKS = {
    "easy": {
        "num_machines": 2, "num_jobs": 3, "failure_rate": 0.0,
        "max_priority": 1, "job_time_range": (2, 5),
        "deadline_slack": (4, 8), "max_steps": 20,
    },
    "medium": {
        "num_machines": 4, "num_jobs": 7, "failure_rate": 0.08,
        "max_priority": 2, "job_time_range": (3, 7),
        "deadline_slack": (2, 5), "max_steps": 30,
    },
    "hard": {
        "num_machines": 6, "num_jobs": 12, "failure_rate": 0.15,
        "max_priority": 3, "job_time_range": (3, 8),
        "deadline_slack": (1, 4), "max_steps": 40,
    },
}


class _Machine:
    __slots__ = ("id", "status", "current_job", "failure_rate")

    def __init__(self, id, failure_rate=0.0):
        self.id = id
        self.status = "idle"
        self.current_job = None
        self.failure_rate = failure_rate


class _Job:
    __slots__ = ("id", "remaining_time", "deadline", "priority", "assigned_machine")

    def __init__(self, id, remaining_time, deadline, priority=1):
        self.id = id
        self.remaining_time = remaining_time
        self.deadline = deadline
        self.priority = priority
        self.assigned_machine = None


class _Env:
    """Pure-Python FactoryEnv with identical RL dynamics."""

    def __init__(self, task="easy", seed=42):
        cfg = TASKS[task]
        rng = random.Random(seed)
        self.machines = [
            _Machine(f"M{i+1}", cfg["failure_rate"])
            for i in range(cfg["num_machines"])
        ]
        self.jobs = []
        for i in range(cfg["num_jobs"]):
            pt = rng.randint(*cfg["job_time_range"])
            dl = pt + rng.randint(*cfg["deadline_slack"])
            pr = rng.randint(1, cfg["max_priority"])
            self.jobs.append(_Job(f"J{i+1}", pt, dl, pr))
        self.completed_jobs = []
        self.late_jobs = 0
        self.time = 0
        self.max_steps = cfg["max_steps"]
        self._rng = rng

    def _find_job(self, jid):
        return next((j for j in self.jobs if j.id == jid), None) if jid else None

    def _find_machine(self, mid):
        return next((m for m in self.machines if m.id == mid), None) if mid else None

    def step(self, action_type, job_id=None, machine_id=None):
        if action_type == "assign_job":
            job = self._find_job(job_id)
            machine = self._find_machine(machine_id)
            if job and machine and machine.status == "idle":
                job.assigned_machine = machine.id
                machine.status = "busy"
                machine.current_job = job.id

        elif action_type == "repair":
            machine = self._find_machine(machine_id)
            if machine and machine.status == "broken":
                machine.status = "idle"

        self.time += 1

        for machine in self.machines:
            if machine.status == "busy":
                job = self._find_job(machine.current_job)
                if job:
                    job.remaining_time -= 1
                    if job.remaining_time <= 0:
                        if self.time > job.deadline:
                            self.late_jobs += 1
                        self.jobs.remove(job)
                        self.completed_jobs.append(job)
                        machine.status = "idle"
                        machine.current_job = None

            if machine.status == "busy" and machine.failure_rate > 0:
                if self._rng.random() < machine.failure_rate:
                    machine.status = "broken"
                    stalled = self._find_job(machine.current_job)
                    if stalled:
                        stalled.assigned_machine = None
                    machine.current_job = None

        return self.time >= self.max_steps or len(self.jobs) == 0


# ── Score formula ─────────────────────────────────────────────────────────────

def _compute(completed, on_time, total, late):
    if total == 0:
        return 0.001
    score = (
        0.5 * (completed / total)
        + 0.3 * (on_time / max(completed, 1))
        + 0.2 * max(0.0, 1.0 - late / max(completed, 1))
    )
    return round(max(0.001, min(0.999, score)), 4)


def _score_env(env):
    t = env.time
    completed = len(env.completed_jobs)
    total = completed + len(env.jobs)
    on_time = sum(1 for j in env.completed_jobs if j.deadline >= t)
    return _compute(completed, on_time, total, env.late_jobs)


def _score_obj(obj):
    """Score from a finished FactoryEnv object or state dict."""
    if isinstance(obj, dict):
        done_list = obj.get("completed_jobs", []) or []
        pend_list = obj.get("pending_jobs", []) or []
        late = int(obj.get("late_jobs", 0) or 0)
        t = int(obj.get("time", 0) or 0)
        completed = len(done_list)
        total = completed + len(pend_list)
        on_time = sum(
            1 for j in done_list
            if (j.get("deadline", 0) if isinstance(j, dict)
                else getattr(j, "deadline", 0)) >= t
        )
    else:
        done_list = list(getattr(obj, "completed_jobs", []) or [])
        pend_list = list(getattr(obj, "jobs",
                         getattr(obj, "pending_jobs", [])) or [])
        late = int(getattr(obj, "late_jobs", 0) or 0)
        t = int(getattr(obj, "time", 0) or 0)
        completed = len(done_list)
        total = completed + len(pend_list)
        on_time = sum(1 for j in done_list if getattr(j, "deadline", 0) >= t)
    return _compute(completed, on_time, total, late)


# ── Heuristic agent ───────────────────────────────────────────────────────────

def _heuristic(machines, jobs):
    """Earliest-deadline-first heuristic."""
    for m in machines:
        if m.status == "broken":
            return "repair", None, m.id
    for j in sorted(jobs, key=lambda x: (x.deadline, -x.priority)):
        for m in machines:
            if m.status == "idle":
                return "assign_job", j.id, m.id
    return "wait", None, None


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episode(task, seed=42):
    """Run a full heuristic episode and return the graded score."""
    # Try to use the real FactoryEnv from the package first.
    try:
        from factory_env.env import FactoryEnv
        from factory_env.models import FactoryAction

        env = FactoryEnv(task=task, seed=seed)
        obs = env.reset()
        for _ in range(obs.max_steps):
            if obs.done:
                break
            # Heuristic action selection
            broken = [m for m in obs.machines if m.status == "broken"]
            if broken:
                action = FactoryAction(action_type="repair",
                                       machine_id=broken[0].id)
            else:
                action = None
                for j in sorted(obs.pending_jobs,
                                 key=lambda x: (x.deadline, -x.priority)):
                    for m in obs.machines:
                        if m.status == "idle":
                            action = FactoryAction(action_type="assign_job",
                                                   job_id=j.id,
                                                   machine_id=m.id)
                            break
                    if action:
                        break
                if action is None:
                    action = FactoryAction(action_type="wait")
            obs = env.step(action)
        return _score_obj(env)

    except Exception:
        pass

    # Fallback: identical RL dynamics implemented in pure Python above.
    env = _Env(task=task, seed=seed)
    for _ in range(env.max_steps):
        action_type, job_id, machine_id = _heuristic(env.machines, env.jobs)
        done = env.step(action_type, job_id, machine_id)
        if done:
            break
    return _score_env(env)


# ── Public graders ────────────────────────────────────────────────────────────

def score_easy(state_or_env=None) -> float:
    """Grade an easy-task episode (2 machines, 3 jobs, no failures).
    Returns float in (0.0, 1.0)."""
    if state_or_env is not None:
        return _score_obj(state_or_env)
    return _run_episode("easy")


def score_medium(state_or_env=None) -> float:
    """Grade a medium-task episode (4 machines, 7 jobs, 8% failure rate).
    Returns float in (0.0, 1.0)."""
    if state_or_env is not None:
        return _score_obj(state_or_env)
    return _run_episode("medium")


def score_hard(state_or_env=None) -> float:
    """Grade a hard-task episode (6 machines, 12 jobs, 15% failure rate).
    Returns float in (0.0, 1.0)."""
    if state_or_env is not None:
        return _score_obj(state_or_env)
    return _run_episode("hard")
