def compute_score(completed, on_time, total_jobs, late_jobs, task="easy"):
    if total_jobs == 0:
        return 0.0
    completion_rate = completed / total_jobs
    on_time_rate = on_time / max(completed, 1)
    utilization_bonus = max(0.0, 1.0 - late_jobs / max(completed, 1))
    score = 0.5 * completion_rate + 0.3 * on_time_rate + 0.2 * utilization_bonus
    return round(max(0.0, min(1.0, score)), 4)


def score_episode(env) -> float:
    total = len(env.completed_jobs) + len(env.jobs)
    completed = len(env.completed_jobs)
    on_time = sum(1 for j in env.completed_jobs if env.time <= j.deadline)
    return compute_score(completed, on_time, total, env.late_jobs, env.task)
