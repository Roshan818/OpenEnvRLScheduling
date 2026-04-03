from pydantic import BaseModel
from typing import List, Optional

class Machine(BaseModel):
   id: str
   status: str  # idle, busy, broken
   current_job: Optional[str] = None

class Job(BaseModel):
   id: str
   remaining_time: int
   deadline: int
   assigned_machine: Optional[str] = None

class Observation(BaseModel):
   machines: List[Machine]
   pending_jobs: List[Job]
   time: int

class Action(BaseModel):
   action_type: str  # assign_job, wait
   job_id: Optional[str] = None
   machine_id: Optional[str] = None

class Reward(BaseModel):
   value: float
