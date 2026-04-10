"""
Inference Script — Smart Factory Scheduling Environment
=======================================================
Runs an LLM agent against the factory_env server for all 3 tasks
(easy, medium, hard) and emits structured stdout logs.

Environment variables:
  HF_TOKEN        HuggingFace / API key  (no default — required)
  API_BASE_URL    LLM endpoint           (default: HF router)
  MODEL_NAME      Model identifier       (default: Qwen/Qwen2.5-72B-Instruct)
  IMAGE_NAME      Docker image name — if set, spins up a container
  ENV_URL         Server URL             (default: http://localhost:7860)
  FACTORY_TASK    Run a single task: easy | medium | hard  (default: run all 3)

STDOUT FORMAT  (one [START] / N [STEP] / one [END] per task):
  [START] task=<task> env=factory_env model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from factory_env.client import FactoryEnvClient
from factory_env.grader import compute_score
from factory_env.models import FactoryAction

# ── Configuration ────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY   = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME        = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
ENV_URL: str      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK: str    = "factory_env"
TEMPERATURE: float        = 0.2
MAX_TOKENS: int           = 80
SUCCESS_SCORE_THRESHOLD   = 0.5

# Run a single task if FACTORY_TASK is set, otherwise run all three
_single = os.getenv("FACTORY_TASK", "").strip()
TASKS: List[str] = [_single] if _single else ["easy", "medium", "hard"]

SYSTEM_PROMPT = textwrap.dedent("""
    You are controlling a smart factory scheduling system.
    Goal: complete all jobs before their deadlines, keep machines busy, repair broken machines.
    Actions (respond with EXACTLY one line):
      assign_job <job_id> <machine_id>
      repair <machine_id>
      wait
    Respond with ONLY the action string — no explanation.
""").strip()


# ── Log helpers ───────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action.replace(' ', '_')} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(step: int, obs, last_reward: float) -> str:
    machines = "\n".join(
        f"  {m.id}: {m.status}" + (f" ({m.current_job})" if m.current_job else "")
        for m in obs.machines
    )
    jobs = (
        "\n".join(
            f"  {j.id}: remaining={j.remaining_time}, deadline={j.deadline},"
            f" priority={j.priority}"
            for j in obs.pending_jobs
        )
        or "  (none)"
    )
    return (
        f"Step {step}/{obs.max_steps} | time={obs.time} | last_reward={last_reward:+.2f}\n"
        f"Machines:\n{machines}\nPending Jobs:\n{jobs}\nAction:"
    )


# ── LLM call ─────────────────────────────────────────────────────────────────
def get_model_action(client: OpenAI, step: int, obs, last_reward: float) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(step, obs, last_reward)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (resp.choices[0].message.content or "wait").strip()
        return text.splitlines()[0]
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "wait"


# ── Action parser + heuristic fallback ───────────────────────────────────────
def parse_action(text: str) -> FactoryAction:
    try:
        parts = text.strip().split()
        if parts[0] == "assign_job" and len(parts) == 3:
            return FactoryAction(action_type="assign_job",
                                 job_id=parts[1], machine_id=parts[2])
        if parts[0] == "repair" and len(parts) == 2:
            return FactoryAction(action_type="repair", machine_id=parts[1])
    except Exception:
        pass
    return FactoryAction(action_type="wait")


def heuristic_action(obs):
    for m in obs.machines:
        if m.status == "broken":
            return FactoryAction(action_type="repair", machine_id=m.id), f"repair {m.id}"
    for j in sorted(obs.pending_jobs, key=lambda x: (x.deadline, -x.priority)):
        for m in obs.machines:
            if m.status == "idle":
                s = f"assign_job {j.id} {m.id}"
                return FactoryAction(action_type="assign_job",
                                     job_id=j.id, machine_id=m.id), s
    return FactoryAction(action_type="wait"), "wait"


# ── Score from final state ────────────────────────────────────────────────────
def score_from_state(state, task: str) -> float:
    completed_jobs = getattr(state, "completed_jobs", []) or []
    pending_jobs   = getattr(state, "pending_jobs",   []) or []
    late_jobs      = getattr(state, "late_jobs", 0) or 0
    time           = getattr(state, "time",      0) or 0
    completed      = len(completed_jobs)
    total          = completed + len(pending_jobs)
    on_time = sum(
        1 for j in completed_jobs
        if (j.get("deadline", 0) if isinstance(j, dict)
            else j.deadline) >= time
    )
    return compute_score(completed, on_time, total, late_jobs, task)


# ── Single-task episode ───────────────────────────────────────────────────────
async def run_task(env_client: FactoryEnvClient,
                   llm_client: OpenAI,
                   task: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result     = await env_client.reset(task=task)
        obs        = result.observation
        last_reward = 0.0

        for step in range(1, obs.max_steps + 1):
            if result.done:
                break

            action_text = get_model_action(llm_client, step, obs, last_reward)
            action      = parse_action(action_text)

            if action.action_type == "wait" and (
                obs.pending_jobs
                or any(m.status == "broken" for m in obs.machines)
            ):
                action, action_text = heuristic_action(obs)

            result      = await env_client.step(action)
            obs         = result.observation
            reward      = result.reward or 0.0
            done        = result.done
            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_text,
                     reward=reward, done=done, error=None)
            if done:
                break

        try:
            state = await env_client.state()
            score = score_from_state(state, task)
        except Exception as exc:
            print(f"[DEBUG] state() failed: {exc}", flush=True)
            max_r = {"easy": 4.0, "medium": 12.0, "hard": 20.0}.get(task, 10.0)
            raw   = sum(rewards) / max_r if max_r > 0 else 0.0
            score = round(max(0.001, min(0.999, raw)), 4)

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────
async def _make_client() -> FactoryEnvClient:
    """Create and connect a fresh client for one task episode."""
    if IMAGE_NAME:
        print(f"[DEBUG] Spinning up Docker image: {IMAGE_NAME}", flush=True)
        return await FactoryEnvClient.from_docker_image(IMAGE_NAME)
    url = ENV_URL or "http://localhost:7860"
    print(f"[DEBUG] Connecting to: {url}", flush=True)
    client = FactoryEnvClient(base_url=url)
    await client.connect()
    return client


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        env_client = await _make_client()
        try:
            await run_task(env_client, llm_client, task)
        finally:
            try:
                await env_client.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
