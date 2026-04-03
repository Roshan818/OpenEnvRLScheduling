import random
from typing import List
from factory_env.models import Observation, Action, Machine, Job

class FactoryEnv:
   def __init__(self, task="easy"):
       self.task = task
       self.time = 0
       self.max_steps = 20

   async def reset(self):
       random.seed(42)

       self.time = 0

       self.machines = [
           Machine(id="M1", status="idle"),
           Machine(id="M2", status="idle"),
       ]

       self.jobs = [
           Job(id="J1", remaining_time=3, deadline=10),
           Job(id="J2", remaining_time=2, deadline=8),
       ]

       return self._get_result(0.0, False)


   async def step(self, action: Action):
       reward = 0.0

       # Apply action
       if action.action_type == "assign_job":
           job = self._find_job(action.job_id)
           machine = self._find_machine(action.machine_id)

           if job and machine and machine.status == "idle":
               job.assigned_machine = machine.id
               machine.status = "busy"
               machine.current_job = job.id
               reward += 0.2
           else:
               reward -= 0.2  # invalid action

       # Simulate time
       self.time += 1

       for machine in self.machines:
           if machine.status == "busy":
               job = self._find_job(machine.current_job)
               job.remaining_time -= 1

               if job.remaining_time <= 0:
                   reward += 1.0
                   self.jobs.remove(job)
                   machine.status = "idle"
                   machine.current_job = None

       # Penalty for idle machines
       idle_count = sum(1 for m in self.machines if m.status == "idle")
       reward -= idle_count * 0.05

       done = self.time >= self.max_steps or len(self.jobs) == 0

       return self._get_result(reward, done)


   def state(self):
       return self._get_observation()

   def _get_observation(self):
       return Observation(
           machines=self.machines,
           pending_jobs=self.jobs,
           time=self.time,
       )

   def _get_result(self, reward, done):
       return type("Result", (), {
           "observation": self._get_observation(),
           "reward": reward,
           "done": done
       })

   def _find_job(self, job_id):
       return next((j for j in self.jobs if j.id == job_id), None)

   def _find_machine(self, machine_id):
       return next((m for m in self.machines if m.id == machine_id), None)

   async def close(self):
       pass

