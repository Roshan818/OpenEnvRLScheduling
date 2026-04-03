"""
Factory Environment Inference Script
===================================
Follows OpenEnv evaluation format strictly.
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from factory_env.env import FactoryEnv
from factory_env.models import Action

# =========================
# ENV VARIABLES (MANDATORY)
# =========================
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("FACTORY_TASK", "easy")
BENCHMARK = "factory_env"

MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 100
SUCCESS_SCORE_THRESHOLD = 0.5

# =========================
# PROMPTS
# =========================
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a factory scheduling system.

    Your goal:
    - Assign jobs to machines efficiently
    - Minimize idle machines
    - Finish all jobs as fast as possible

    Available actions:
    1. assign_job <job_id> <machine_id>
    2. wait

    Rules:
    - Only assign jobs that exist
    - Only assign to idle machines
    - One action per step

    Respond ONLY with the action string.
    Example:
    assign_job J1 M1
    """
).strip()


# =========================
# LOGGING FUNCTIONS (STRICT FORMAT)
# =========================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# =========================
# PROMPT BUILDER
# =========================
def build_user_prompt(step, obs, last_reward):
    machines_str = "\n".join(
        [f"{m.id}: {m.status} (job={m.current_job})" for m in obs.machines]
    )

    jobs_str = "\n".join(
        [f"{j.id}: remaining={j.remaining_time}, deadline={j.deadline}" for j in obs.pending_jobs]
    ) or "None"

    return textwrap.dedent(
        f"""
        Step: {step}

        Current Time: {obs.time}

        Machines:
        {machines_str}

        Pending Jobs:
        {jobs_str}

        Last reward: {last_reward:.2f}

        What action do you take?
        """
    ).strip()


# =========================
# LLM CALL
# =========================
def get_model_action(client: OpenAI, step, obs, last_reward) -> str:
    try:
        user_prompt = build_user_prompt(step, obs, last_reward)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = (completion.choices[0].message.content or "").strip()
        return text if text else "wait"

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "wait"


# =========================
# ACTION PARSER
# =========================
def parse_action(text: str) -> Action:
    try:
        parts = text.strip().split()

        if parts[0] == "assign_job" and len(parts) == 3:
            return Action(
                action_type="assign_job",
                job_id=parts[1],
                machine_id=parts[2],
            )

        elif parts[0] == "wait":
            return Action(action_type="wait")

    except Exception:
        pass

    # fallback safe action
    return Action(action_type="wait")


# =========================
# SIMPLE HEURISTIC FALLBACK
# =========================
def heuristic_action(obs) -> Action:
    for job in obs.pending_jobs:
        for machine in obs.machines:
            if machine.status == "idle":
                return Action(
                    action_type="assign_job",
                    job_id=job.id,
                    machine_id=machine.id,
                )
    return Action(action_type="wait")


# =========================
# MAIN LOOP
# =========================
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = FactoryEnv(task=TASK_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # LLM decision
            action_text = get_model_action(client, step, obs, last_reward)

            # Parse action
            action = parse_action(action_text)

            # Fallback if invalid
            if action.action_type == "wait" and len(obs.pending_jobs) > 0:
                action = heuristic_action(obs)
                action_text = "heuristic_assign"

            # Step env
            result = await env.step(action)

            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step, action_text, reward, done, error)

            if done:
                break

        # Normalize score
        if rewards:
            score = sum(rewards) / len(rewards)
            score = max(0.0, min(1.0, score))

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close error: {e}", flush=True)

        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())