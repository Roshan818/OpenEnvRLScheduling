"""
Inference Script — Smart Factory Scheduling Environment
========================================================
Mandatory env vars (per hackathon spec):
  OPENAI_API_KEY  API key (also accepts HF_TOKEN for HF router)
  API_BASE_URL    LLM endpoint  (default: HF router)
  MODEL_NAME      Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN        HuggingFace token
  FACTORY_TASK    easy | medium | hard  (default: easy)

STDOUT FORMAT:
  [START] task=<name> env=factory_env model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI

from factory_env.env import FactoryEnv
from factory_env.models import FactoryAction as Action
from factory_env.grader import score_episode

API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("FACTORY_TASK", "easy")
BENCHMARK = "factory_env"
TEMPERATURE = 0.2
MAX_TOKENS = 80
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are controlling a smart factory scheduling system.
    Goal: complete all jobs before their deadlines, keep machines busy, repair broken machines.
    Actions (respond with EXACTLY one line):
      assign_job <job_id> <machine_id>
      repair <machine_id>
      wait
    Respond with ONLY the action string.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


def build_prompt(step: int, obs, last_reward: float) -> str:
    machines = "\n".join(f"  {m.id}: {m.status}" + (f" ({m.current_job})" if m.current_job else "") for m in obs.machines)
    jobs = "\n".join(f"  {j.id}: remaining={j.remaining_time}, deadline={j.deadline}, priority={j.priority}" for j in obs.pending_jobs) or "  (none)"
    return f"Step {step}/{obs.max_steps} | time={obs.time} | last_reward={last_reward:+.2f}\nMachines:\n{machines}\nPending Jobs:\n{jobs}\nAction:"


def get_model_action(client: OpenAI, step: int, obs, last_reward: float) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(step, obs, last_reward)}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "wait").strip().splitlines()[0]
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "wait"


def parse_action(text: str) -> Action:
    try:
        parts = text.strip().split()
        if parts[0] == "assign_job" and len(parts) == 3:
            return Action(action_type="assign_job", job_id=parts[1], machine_id=parts[2])
        if parts[0] == "repair" and len(parts) == 2:
            return Action(action_type="repair", machine_id=parts[1])
    except Exception:
        pass
    return Action(action_type="wait")


def heuristic_action(obs) -> Tuple[Action, str]:
    for m in obs.machines:
        if m.status == "broken":
            return Action(action_type="repair", machine_id=m.id), f"repair {m.id}"
    for j in sorted(obs.pending_jobs, key=lambda x: (x.deadline, -x.priority)):
        for m in obs.machines:
            if m.status == "idle":
                s = f"assign_job {j.id} {m.id}"
                return Action(action_type="assign_job", job_id=j.id, machine_id=m.id), s
    return Action(action_type="wait"), "wait"


def run_task(task_name: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = FactoryEnv(task=task_name)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        last_reward = 0.0

        for step in range(1, obs.max_steps + 1):
            if obs.done:
                break
            action_text = get_model_action(client, step, obs, last_reward)
            action = parse_action(action_text)
            if action.action_type == "wait" and (obs.pending_jobs or any(m.status == "broken" for m in obs.machines)):
                action, action_text = heuristic_action(obs)
            obs = env.step(action)
            reward = obs.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            log_step(step, action_text, reward, obs.done, None)
            if obs.done:
                break

        score = score_episode(env)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    run_task(TASK_NAME)
