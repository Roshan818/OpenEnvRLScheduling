"""
Graders for Smart Factory Scheduling tasks.

Each grader is self-contained: when called with no arguments it creates a
FactoryEnv, runs a deterministic heuristic episode, and returns a score
strictly in (0, 1).

Alternatively, pass an env object or state dict from an already-run episode:
    score_easy(env)       # env object with .completed_jobs, .jobs, .time …
    score_easy(state)     # dict with "completed_jobs", "pending_jobs", "time" …
"""

from __future__ import annotations


# ── internal helpers ──────────────────────────────────────────────────────────

def _compute(completed: int, on_time: int, total: int, late: int) -> float:
    if total == 0:
        return 0.001
    score = (
        0.5 * (completed / total)
        + 0.3 * (on_time / max(completed, 1))
        + 0.2 * max(0.0, 1.0 - late / max(completed, 1))
    )
    return round(max(0.001, min(0.999, score)), 4)


def _score_from(obj) -> float:
    """Accept env object or state dict and return a score."""
    if isinstance(obj, dict):
        done_jobs = obj.get("completed_jobs", []) or []
        pending   = obj.get("pending_jobs", []) or []
        late      = obj.get("late_jobs", 0) or 0
        t         = obj.get("time", 0) or 0
    else:
        done_jobs = list(getattr(obj, "completed_jobs", []) or [])
        pending   = list(
            getattr(obj, "jobs", getattr(obj, "pending_jobs", []))
        ) or []
        late = getattr(obj, "late_jobs", 0) or 0
        t    = getattr(obj, "time", 0) or 0

    completed = len(done_jobs)
    total     = completed + len(pending)
    on_time   = sum(
        1 for j in done_jobs
        if (j.get("deadline", 0) if isinstance(j, dict)
            else getattr(j, "deadline", 0)) >= t
    )
    return _compute(completed, on_time, total, late)


def _heuristic_action(obs):
    """Simple earliest-deadline-first heuristic."""
    from factory_env.models import FactoryAction
    for m in obs.machines:
        if m.status == "broken":
            return FactoryAction(action_type="repair", machine_id=m.id)
    for j in sorted(obs.pending_jobs, key=lambda x: (x.deadline, -x.priority)):
        for m in obs.machines:
            if m.status == "idle":
                return FactoryAction(
                    action_type="assign_job", job_id=j.id, machine_id=m.id
                )
    return None  # wait


def _run_episode(task: str, seed: int = 42) -> float:
    """Run one full heuristic episode and return the graded score."""
    from factory_env.env import FactoryEnv
    from factory_env.models import FactoryAction

    env = FactoryEnv(task=task, seed=seed)
    obs = env.reset()
    for _ in range(obs.max_steps):
        if obs.done:
            break
        action = _heuristic_action(obs) or FactoryAction(action_type="wait")
        obs = env.step(action)
    return _score_from(env)


# ── public graders ────────────────────────────────────────────────────────────

def score_easy(state_or_env=None) -> float:
    """Grade an easy-task episode (2 machines, 3 jobs, no failures)."""
    if state_or_env is not None:
        return _score_from(state_or_env)
    return _run_episode("easy")


def score_medium(state_or_env=None) -> float:
    """Grade a medium-task episode (4 machines, 7 jobs, 8% failures)."""
    if state_or_env is not None:
        return _score_from(state_or_env)
    return _run_episode("medium")


def score_hard(state_or_env=None) -> float:
    """Grade a hard-task episode (6 machines, 12 jobs, 15% failures)."""
    if state_or_env is not None:
        return _score_from(state_or_env)
    return _run_episode("hard")
