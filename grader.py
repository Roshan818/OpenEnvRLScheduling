"""
Graders for Smart Factory Scheduling tasks.
Called by the OpenEnv validator to score an episode.

Each grader accepts either:
  - an env object  (has .completed_jobs, .jobs, .time, .late_jobs attributes)
  - a state dict   (has "completed_jobs", "pending_jobs", "time", "late_jobs" keys)

Returns a float strictly in (0, 1).
"""


def _compute(completed, on_time, total, late):
    if total == 0:
        return 0.001
    score = (
        0.5 * (completed / total)
        + 0.3 * (on_time / max(completed, 1))
        + 0.2 * max(0.0, 1.0 - late / max(completed, 1))
    )
    return round(max(0.001, min(0.999, score)), 4)


def _score(state_or_env):
    if isinstance(state_or_env, dict):
        done = state_or_env.get("completed_jobs", []) or []
        pending = state_or_env.get("pending_jobs", []) or []
        late = state_or_env.get("late_jobs", 0) or 0
        t = state_or_env.get("time", 0) or 0
    else:
        done = list(getattr(state_or_env, "completed_jobs", []) or [])
        pending = list(getattr(state_or_env, "jobs", getattr(state_or_env, "pending_jobs", [])) or [])
        late = getattr(state_or_env, "late_jobs", 0) or 0
        t = getattr(state_or_env, "time", 0) or 0

    completed = len(done)
    total = completed + len(pending)
    on_time = sum(
        1 for j in done
        if (j.get("deadline", 0) if isinstance(j, dict) else getattr(j, "deadline", 0)) >= t
    )
    return _compute(completed, on_time, total, late)


def score_easy(state_or_env):
    """Grade an easy-task episode (2 machines, 3 jobs, no failures)."""
    return _score(state_or_env)


def score_medium(state_or_env):
    """Grade a medium-task episode (4 machines, 7 jobs, 8% failures)."""
    return _score(state_or_env)


def score_hard(state_or_env):
    """Grade a hard-task episode (6 machines, 12 jobs, 15% failures)."""
    return _score(state_or_env)
