"""
Graders for Smart Factory Scheduling tasks.

Each public function:
  - Accepts an optional state/env argument to score a finished episode.
  - When called with no argument, runs a deterministic heuristic episode
    on the real FactoryEnv and returns the score.
  - Always returns a float strictly in (0.0, 1.0).
"""

from __future__ import annotations


# ── Score formula ─────────────────────────────────────────────────────────────

def _compute(completed: int, on_time: int, total: int, late: int) -> float:
    if total == 0:
        return 0.001
    score = (
        0.5 * (completed / total)
        + 0.3 * (on_time / max(completed, 1))
        + 0.2 * max(0.0, 1.0 - late / max(completed, 1))
    )
    return round(max(0.001, min(0.999, score)), 4)


def _score_obj(obj) -> float:
    """Score from a finished FactoryEnv object or state dict."""
    if isinstance(obj, dict):
        done_list = obj.get("completed_jobs", []) or []
        pend_list = obj.get("pending_jobs",   []) or []
        late      = int(obj.get("late_jobs", 0) or 0)
        t         = int(obj.get("time", 0) or 0)
        completed = len(done_list)
        total     = completed + len(pend_list)
        on_time   = sum(
            1 for j in done_list
            if (j.get("deadline", 0) if isinstance(j, dict)
                else getattr(j, "deadline", 0)) >= t
        )
    else:
        done_list = list(getattr(obj, "completed_jobs", []) or [])
        pend_list = list(getattr(obj, "jobs", getattr(obj, "pending_jobs", [])) or [])
        late      = int(getattr(obj, "late_jobs", 0) or 0)
        t         = int(getattr(obj, "time", 0) or 0)
        completed = len(done_list)
        total     = completed + len(pend_list)
        on_time   = sum(1 for j in done_list if getattr(j, "deadline", 0) >= t)
    return _compute(completed, on_time, total, late)


# ── Heuristic agent ───────────────────────────────────────────────────────────

def _heuristic(obs):
    """Earliest-deadline-first heuristic that runs on a FactoryObservation."""
    from factory_env.models import FactoryAction
    for m in obs.machines:
        if m.status == "broken":
            return FactoryAction(action_type="repair", machine_id=m.id)
    for j in sorted(obs.pending_jobs, key=lambda x: (x.deadline, -x.priority)):
        for m in obs.machines:
            if m.status == "idle":
                return FactoryAction(action_type="assign_job",
                                     job_id=j.id, machine_id=m.id)
    return None


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episode(task: str, seed: int = 42) -> float:
    """Run a full heuristic episode on FactoryEnv and return the graded score."""
    from factory_env.env import FactoryEnv
    from factory_env.models import FactoryAction

    env = FactoryEnv(task=task, seed=seed)
    obs = env.reset()
    for _ in range(obs.max_steps):
        if obs.done:
            break
        action = _heuristic(obs) or FactoryAction(action_type="wait")
        obs = env.step(action)
    return _score_obj(env)


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
