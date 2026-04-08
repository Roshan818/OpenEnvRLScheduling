"""
Inference Script — Smart Factory Scheduling Environment
=======================================================
Connects to a running factory_env server via WebSocket and runs an LLM agent.

Mandatory env vars (per hackathon spec):
  HF_TOKEN       HuggingFace API key (also used as OPENAI_API_KEY)
  API_BASE_URL   LLM endpoint  (default: HF router)
  MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)

Optional env vars:
  ENV_URL        URL of running factory_env server (default: http://localhost:7860)
  IMAGE_NAME     Docker image name — if set, spins up a container instead of ENV_URL
  FACTORY_TASK   easy | medium | hard  (default: easy)

STDOUT FORMAT (strict — do not alter):
  [START] task=<name> env=factory_env model=<model>
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
API_KEY: Optional[str] = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
)
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME: str = os.getenv("FACTORY_TASK", "easy")
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK: str = "factory_env"
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 80
SUCCESS_SCORE_THRESHOLD: float = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are controlling a smart factory scheduling system.
    Goal: complete all jobs before their deadlines, keep machines busy, repair broken machines.
    Actions (respond with EXACTLY one line):
      assign_job <job_id> <machine_id>
      repair <machine_id>
      wait
    Respond with ONLY the action string — no explanation.
""").strip()


# ── Log helpers (strict format required by judges) ────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
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
            f"  {j.id}: remaining={j.remaining_time}, deadline={j.deadline}, priority={j.priority}"
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
                {"role": "user", "content": build_prompt(step, obs, last_reward)},
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
            return FactoryAction(action_type="assign_job", job_id=parts[1], machine_id=parts[2])
        if parts[0] == "repair" and len(parts) == 2:
            return FactoryAction(action_type="repair", machine_id=parts[1])
    except Exception:
        pass
    return FactoryAction(action_type="wait")


def heuristic_action(obs):
    """Fallback heuristic when LLM returns an ineffective wait."""
    for m in obs.machines:
        if m.status == "broken":
            return FactoryAction(action_type="repair", machine_id=m.id), f"repair {m.id}"
    for j in sorted(obs.pending_jobs, key=lambda x: (x.deadline, -x.priority)):
        for m in obs.machines:
            if m.status == "idle":
                s = f"assign_job {j.id} {m.id}"
                return FactoryAction(action_type="assign_job", job_id=j.id, machine_id=m.id), s
    return FactoryAction(action_type="wait"), "wait"


# ── Score from final state ────────────────────────────────────────────────────
def score_from_state(state, task: str) -> float:
    """Compute episode score from WebSocket state response."""
    completed_jobs = getattr(state, "completed_jobs", []) or []
    pending_jobs = getattr(state, "pending_jobs", []) or []
    late_jobs = getattr(state, "late_jobs", 0) or 0
    time = getattr(state, "time", 0) or 0

    completed = len(completed_jobs)
    total = completed + len(pending_jobs)

    # on_time: jobs whose deadline hasn't passed by end of episode (matches grader)
    on_time = sum(
        1 for j in completed_jobs
        if (j.get("deadline", 0) if isinstance(j, dict) else j.deadline) >= time
    )

    return compute_score(completed, on_time, total, late_jobs, task)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # Connect to environment — Docker image, direct URL, or localhost
    if LOCAL_IMAGE_NAME:
        print(f"[DEBUG] Spinning up Docker image: {LOCAL_IMAGE_NAME}", flush=True)
        env = await FactoryEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        url = ENV_URL or "http://localhost:7860"
        print(f"[DEBUG] Connecting to: {url}", flush=True)
        env = FactoryEnvClient(base_url=url)
        await env.connect()

    try:
        result = await env.reset(task=TASK_NAME)
        obs = result.observation
        last_reward = 0.0

        for step in range(1, obs.max_steps + 1):
            if result.done:
                break

            action_text = get_model_action(llm_client, step, obs, last_reward)
            action = parse_action(action_text)

            # Heuristic fallback: if LLM returns wait but there's work to do
            if action.action_type == "wait" and (
                obs.pending_jobs or any(m.status == "broken" for m in obs.machines)
            ):
                action, action_text = heuristic_action(obs)

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_text, reward=reward, done=done, error=None)

            if done:
                break

        # Compute score from final WebSocket state
        try:
            state = await env.state()
            score = score_from_state(state, TASK_NAME)
        except Exception as exc:
            print(f"[DEBUG] state() failed, falling back to reward sum: {exc}", flush=True)
            max_reward = {"easy": 4.0, "medium": 12.0, "hard": 20.0}.get(TASK_NAME, 10.0)
            score = min(max(sum(rewards) / max_reward, 0.0), 1.0)

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
