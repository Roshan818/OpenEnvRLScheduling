from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

# Lazy openenv base classes — fall back to pydantic BaseModel when the
# openenv.core import chain (which pulls in gradio/PIL) is unavailable.
try:
    from openenv.core import Action as BaseAction, Observation as BaseObservation, State as BaseState
except Exception:
    BaseAction = BaseModel       # type: ignore[assignment,misc]
    BaseObservation = BaseModel  # type: ignore[assignment,misc]
    BaseState = BaseModel        # type: ignore[assignment,misc]


class Machine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    status: str                       # idle | busy | broken
    current_job: Optional[str] = None
    failure_rate: float = 0.0


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    remaining_time: int
    deadline: int
    priority: int = 1
    assigned_machine: Optional[str] = None


class FactoryAction(BaseAction):
    """
    action_type: assign_job | repair | wait
    job_id:      required for assign_job
    machine_id:  required for assign_job / repair
    """
    action_type: str
    job_id: Optional[str] = None
    machine_id: Optional[str] = None


class FactoryObservation(BaseObservation):
    """Inherits done/reward/metadata from openenv base when available;
    defined here explicitly so the class works when falling back to BaseModel."""
    done: bool = False
    reward: Optional[float] = None
    machines: List[Machine] = Field(default_factory=list)
    pending_jobs: List[Job] = Field(default_factory=list)
    completed_jobs: List[Job] = Field(default_factory=list)
    time: int = 0
    max_steps: int = 20
    task: str = "easy"


class FactoryState(BaseState):
    machines: List[Machine] = Field(default_factory=list)
    pending_jobs: List[Job] = Field(default_factory=list)
    completed_jobs: List[Job] = Field(default_factory=list)
    time: int = 0
    task: str = "easy"
    late_jobs: int = 0


# Aliases for backward compatibility
Action = FactoryAction
Observation = FactoryObservation
