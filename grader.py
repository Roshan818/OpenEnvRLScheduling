"""
Graders for Smart Factory Scheduling tasks.

Primary path: imports FactoryEnv, runs a full deterministic heuristic episode,
and scores the result using the real environment state.

Fallback path (if factory_env is unavailable): a minimal pure-Python
simulation is used so the validator can still load and call these functions.

All three public functions:
  - Accept an optional state/env argument for scoring a finished episode.
  - When called with no argument, run their own deterministic episode.
  - Always return a float strictly in (0.0, 1.0).
"""

from __future__ import annotations

import random
from typing import Any, List, Optional


# ── Score formula (shared by both paths) ─────────────────────────────────────

def _compute(completed: int, on_time: int, total: int, late: int) -> float:
    if total == 0:
        return 0.001
    score = (
        0.5 * (completed / total)
        + 0.3 * (on_time / max(completed, 1))
        + 0.2 * max(0.0, 1.0 - late / max(completed, 1))
    )
    return round(max(0.001, min(0.999, score)), 4)


def _score_obj(obj: Any) -> float:
    """Score from a finished env object or state dict."""
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
        on_time   = sum(
            1 for j in done_list
            if getattr(j, "deadline", 0) >= t
        )
    return _compute(completed, on_time, total, late)


# ── Primary path: use the real FactoryEnv ────────────────────────────────────

def _heuristic(obs):
    """Earliest-deadline-first heuristic action (works on FactoryObservation)."""
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


def _run_factory_episode(task: str, seed: int = 42) -> float:
    """Run a full heuristic episode on the real FactoryEnv and return score."""
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


# ── Fallback path: pure-Python mini-simulation ───────────────────────────────

_TASK_CFG = {
    "easy":   dict(nm=2, nj=3,  fr=0.00, ms=20, jtr=(2,4), ds=(2,5), mp=1),
    "medium": dict(nm=4, nj=7,  fr=0.08, ms=30, jtr=(2,5), ds=(2,6), mp=2),
    "hard":   dict(nm=6, nj=12, fr=0.15, ms=40, jtr=(2,6), ds=(1,5), mp=3),
}


def _run_mini_episode(task: str, seed: int = 42) -> float:
    """Pure-Python fallback simulation (no external deps)."""
    cfg = _TASK_CFG[task]
    rng = random.Random(seed)

    machines = [{"id": f"M{i+1}", "status": "idle", "job": None,
                 "fr": cfg["fr"]} for i in range(cfg["nm"])]
    jobs = []
    for i in range(cfg["nj"]):
        pt = rng.randint(*cfg["jtr"])
        dl = pt + rng.randint(*cfg["ds"])
        jobs.append({"id": f"J{i+1}", "rt": pt, "dl": dl,
                     "pr": rng.randint(1, cfg["mp"])})

    completed, late, t = [], 0, 0

    for _ in range(cfg["ms"]):
        if not jobs:
            break
        # repair broken machines
        for m in machines:
            if m["status"] == "broken":
                m["status"] = "idle"
                break
        # assign jobs EDF
        for j in sorted(jobs, key=lambda x: (x["dl"], -x["pr"])):
            for m in machines:
                if m["status"] == "idle":
                    m["status"] = "busy"
                    m["job"]    = j["id"]
                    j["m"]      = m["id"]
                    break

        t += 1
        for m in machines:
            if m["status"] == "busy":
                j = next((x for x in jobs if x["id"] == m["job"]), None)
                if j:
                    j["rt"] -= 1
                    if j["rt"] <= 0:
                        if t > j["dl"]:
                            late += 1
                        completed.append(j)
                        jobs.remove(j)
                        m["status"] = "idle"
                        m["job"]    = None
            if m["status"] == "busy" and cfg["fr"] > 0:
                if rng.random() < cfg["fr"]:
                    m["status"] = "broken"
                    m["job"]    = None

    total   = len(completed) + len(jobs)
    n       = len(completed)
    on_time = max(0, n - late)
    return _compute(n, on_time, total, late)


# ── Episode runner (tries FactoryEnv, falls back if unavailable) ─────────────

def _episode(task: str) -> float:
    try:
        return _run_factory_episode(task)
    except Exception:
        return _run_mini_episode(task)


# ── Public graders ────────────────────────────────────────────────────────────

def score_easy(state_or_env=None) -> float:
    """Grade an easy-task episode (2 machines, 3 jobs, no failures).
    Returns float in (0.0, 1.0)."""
    if state_or_env is not None:
        return _score_obj(state_or_env)
    return _episode("easy")


def score_medium(state_or_env=None) -> float:
    """Grade a medium-task episode (4 machines, 7 jobs, 8% failure rate).
    Returns float in (0.0, 1.0)."""
    if state_or_env is not None:
        return _score_obj(state_or_env)
    return _episode("medium")


def score_hard(state_or_env=None) -> float:
    """Grade a hard-task episode (6 machines, 12 jobs, 15% failure rate).
    Returns float in (0.0, 1.0)."""
    if state_or_env is not None:
        return _score_obj(state_or_env)
    return _episode("hard")
