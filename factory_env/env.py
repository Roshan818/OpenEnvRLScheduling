import random
from typing import List, Optional

# Lazy base-class: import openenv.core only when it's available.
# This lets FactoryEnv be imported (e.g. by the grader) even in minimal
# environments where openenv-core's gradio/PIL chain fails to load.
try:
    from openenv.core import Environment as _EnvBase
except Exception:
    _EnvBase = object  # type: ignore[assignment,misc]

from factory_env.models import FactoryAction, FactoryObservation, FactoryState, Machine, Job
from factory_env.tasks import TASKS


class FactoryEnv(_EnvBase):
    """Smart Factory Scheduling Environment — OpenEnv compliant."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task: str = "easy", seed: int = 42):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASKS.keys())}")
        self.task = task
        self.seed = seed
        self.config = TASKS[task]
        self._rng = random.Random(seed)
        self.machines: List[Machine] = []
        self.jobs: List[Job] = []
        self.completed_jobs: List[Job] = []
        self.late_jobs: int = 0
        self.time: int = 0
        self.max_steps: int = self.config["max_steps"]

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> FactoryObservation:
        # Allow task to be overridden at reset time (e.g. from inference script)
        task = kwargs.get("task", self.task)
        if task != self.task and task in TASKS:
            self.task = task
            self.config = TASKS[task]
            self.max_steps = self.config["max_steps"]

        use_seed = seed if seed is not None else self.seed
        self._rng = random.Random(use_seed)
        self.time = 0
        self.completed_jobs = []
        self.late_jobs = 0

        cfg = self.config
        self.machines = [
            Machine(id=f"M{i+1}", status="idle", failure_rate=cfg.get("failure_rate", 0.0))
            for i in range(cfg["num_machines"])
        ]
        self.jobs = []
        for i in range(cfg["num_jobs"]):
            proc_time = self._rng.randint(*cfg["job_time_range"])
            deadline = self.time + proc_time + self._rng.randint(*cfg["deadline_slack"])
            priority = self._rng.randint(1, cfg.get("max_priority", 1))
            self.jobs.append(Job(id=f"J{i+1}", remaining_time=proc_time, deadline=deadline, priority=priority))

        return self._make_obs(reward=None, done=False)

    def step(self, action: FactoryAction, timeout_s: Optional[float] = None, **kwargs) -> FactoryObservation:
        reward = 0.0

        if action.action_type == "assign_job":
            job = self._find_job(action.job_id)
            machine = self._find_machine(action.machine_id)
            if job is None or machine is None or machine.status != "idle":
                reward -= 0.1
            else:
                job.assigned_machine = machine.id
                machine.status = "busy"
                machine.current_job = job.id
                reward += 0.1
        elif action.action_type == "repair":
            machine = self._find_machine(action.machine_id)
            if machine and machine.status == "broken":
                machine.status = "idle"
                reward += 0.05
            else:
                reward -= 0.05

        self.time += 1

        for machine in self.machines:
            if machine.status == "busy":
                job = self._find_job(machine.current_job)
                if job:
                    job.remaining_time -= 1
                    if job.remaining_time <= 0:
                        on_time = self.time <= job.deadline
                        reward += (1.0 + 0.2 * job.priority) if on_time else 0.3
                        if not on_time:
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

        if self.jobs:
            reward -= sum(1 for m in self.machines if m.status == "idle") * 0.05
        for job in self.jobs:
            if self.time > job.deadline:
                reward -= 0.1

        done = self.time >= self.max_steps or len(self.jobs) == 0
        return self._make_obs(reward=reward, done=done)

    @property
    def state(self) -> FactoryState:
        return FactoryState(
            machines=list(self.machines),
            pending_jobs=list(self.jobs),
            completed_jobs=list(self.completed_jobs),
            time=self.time,
            task=self.task,
            late_jobs=self.late_jobs,
        )

    def _make_obs(self, reward, done: bool) -> FactoryObservation:
        return FactoryObservation(
            machines=list(self.machines),
            pending_jobs=list(self.jobs),
            completed_jobs=list(self.completed_jobs),
            time=self.time,
            max_steps=self.max_steps,
            task=self.task,
            reward=reward,
            done=done,
        )

    def _find_job(self, job_id: Optional[str]) -> Optional[Job]:
        return next((j for j in self.jobs if j.id == job_id), None) if job_id else None

    def _find_machine(self, machine_id: Optional[str]) -> Optional[Machine]:
        return next((m for m in self.machines if m.id == machine_id), None) if machine_id else None
