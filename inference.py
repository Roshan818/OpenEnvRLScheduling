"""
Inference Script — Smart Factory Scheduling Environment
=======================================================
Runs an LLM agent against a running factory_env server for all 3 tasks
(easy, medium, hard) and emits structured stdout logs.

Environment variables:
  HF_TOKEN        HuggingFace / API key  (no default — required)
  API_BASE_URL    LLM endpoint           (default: HF router)
  MODEL_NAME      Model identifier       (default: Qwen/Qwen2.5-72B-Instruct)
  ENV_URL         Server URL             (default: http://localhost:7860)
  FACTORY_TASK    Run a single task: easy | medium | hard  (default: run all 3)

STDOUT FORMAT  (one [START] / N [STEP] / one [END] per task):
  [START] task=<task> env=factory_env model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "factory_env"
TEMPERATURE  = 0.2
MAX_TOKENS   = 80
SUCCESS_SCORE_THRESHOLD = 0.5

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
def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action.replace(' ', '_')} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── WebSocket client (raw, no openenv dependency) ────────────────────────────
class EnvSession:
    """Minimal async WebSocket session for the factory_env server."""

    def __init__(self, base_url: str):
        ws_url = base_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
        self._ws_url = ws_url + "/ws"
        self._ws = None

    async def connect(self) -> None:
        import websockets
        self._ws = await websockets.connect(self._ws_url)

    async def _send_recv(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and return the response data dict."""
        await self._ws.send(json.dumps(msg))
        raw      = await self._ws.recv()
        response = json.loads(raw)
        if response.get("type") == "error":
            err = response.get("data", {})
            raise RuntimeError(f"Server error: {err.get('message', err)}")
        return response.get("data", {})

    async def reset(self, task: str) -> Dict[str, Any]:
        """Returns the inner observation dict directly."""
        data = await self._send_recv({"type": "reset", "data": {"task": task}})
        # Response: {"observation": {...}, "reward": null, "done": false}
        # Extract inner observation so callers get machines/pending_jobs/etc. directly
        return data.get("observation", data)

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Returns {"observation": {...}, "reward": float, "done": bool}."""
        return await self._send_recv({"type": "step", "data": action})

    async def state(self) -> Dict[str, Any]:
        """WSStateMessage has no data field — send without it."""
        return await self._send_recv({"type": "state"})

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(step: int, obs: Dict[str, Any], last_reward: float) -> str:
    machines = "\n".join(
        f"  {m['id']}: {m['status']}" + (f" ({m['current_job']})" if m.get("current_job") else "")
        for m in obs.get("machines", [])
    )
    jobs = (
        "\n".join(
            f"  {j['id']}: remaining={j['remaining_time']}, deadline={j['deadline']},"
            f" priority={j['priority']}"
            for j in obs.get("pending_jobs", [])
        )
        or "  (none)"
    )
    return (
        f"Step {step}/{obs.get('max_steps', '?')} | time={obs.get('time', 0)} | "
        f"last_reward={last_reward:+.2f}\n"
        f"Machines:\n{machines}\nPending Jobs:\n{jobs}\nAction:"
    )


# ── LLM call ─────────────────────────────────────────────────────────────────
def get_model_action(client: OpenAI, step: int, obs: Dict[str, Any],
                     last_reward: float) -> str:
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
def parse_action(text: str) -> Optional[Dict[str, Any]]:
    try:
        parts = text.strip().split()
        if parts[0] == "assign_job" and len(parts) == 3:
            return {"action_type": "assign_job", "job_id": parts[1], "machine_id": parts[2]}
        if parts[0] == "repair" and len(parts) == 2:
            return {"action_type": "repair", "machine_id": parts[1]}
    except Exception:
        pass
    return None


def heuristic_action(obs: Dict[str, Any]):
    machines = obs.get("machines", [])
    jobs     = obs.get("pending_jobs", [])
    for m in machines:
        if m["status"] == "broken":
            return {"action_type": "repair", "machine_id": m["id"]}, f"repair {m['id']}"
    for j in sorted(jobs, key=lambda x: (x["deadline"], -x["priority"])):
        for m in machines:
            if m["status"] == "idle":
                s = f"assign_job {j['id']} {m['id']}"
                return {"action_type": "assign_job", "job_id": j["id"], "machine_id": m["id"]}, s
    return {"action_type": "wait"}, "wait"


# ── Score helpers ─────────────────────────────────────────────────────────────
def compute_score(completed: int, on_time: int, total: int, late: int) -> float:
    if total == 0:
        return 0.001
    score = (
        0.5 * (completed / total)
        + 0.3 * (on_time / max(completed, 1))
        + 0.2 * max(0.0, 1.0 - late / max(completed, 1))
    )
    return round(max(0.001, min(0.999, score)), 4)


def score_from_state(state: Dict[str, Any], task: str) -> float:
    completed_jobs = state.get("completed_jobs", []) or []
    pending_jobs   = state.get("pending_jobs",   []) or []
    late_jobs      = state.get("late_jobs", 0) or 0
    t              = state.get("time", 0) or 0
    completed      = len(completed_jobs)
    total          = completed + len(pending_jobs)
    on_time = sum(
        1 for j in completed_jobs
        if (j.get("deadline", 0) if isinstance(j, dict) else j.deadline) >= t
    )
    return compute_score(completed, on_time, total, late_jobs)


# ── Single-task episode ───────────────────────────────────────────────────────
async def run_task(env_url: str, llm_client: OpenAI, task: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task)

    session = EnvSession(env_url)
    try:
        await session.connect()
        obs         = await session.reset(task=task)
        # obs is the inner observation dict (machines, pending_jobs, max_steps, …)
        # "done" and "reward" are NOT in the inner obs — track them separately
        done        = False
        last_reward = 0.0
        max_steps   = obs.get("max_steps", 40)

        for step in range(1, max_steps + 1):
            if done:
                break

            action_text = get_model_action(llm_client, step, obs, last_reward)
            action      = parse_action(action_text)

            # Heuristic fallback when LLM returns wait but work remains
            if action is None or (
                action.get("action_type") == "wait"
                and (obs.get("pending_jobs") or any(m["status"] == "broken"
                     for m in obs.get("machines", [])))
            ):
                action, action_text = heuristic_action(obs)

            result      = await session.step(action)
            # step() returns {"observation": {...}, "reward": float, "done": bool}
            obs         = result.get("observation", result)
            reward      = result.get("reward") or 0.0
            done        = result.get("done", False)
            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_text, reward=reward,
                     done=done, error=None)
            if done:
                break

        try:
            state = await session.state()
            score = score_from_state(state, task)
        except Exception as exc:
            print(f"[DEBUG] state() failed: {exc}", flush=True)
            max_r = {"easy": 4.0, "medium": 12.0, "hard": 20.0}.get(task, 10.0)
            raw   = sum(rewards) / max_r if max_r > 0 else 0.0
            score = round(max(0.001, min(0.999, raw)), 4)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] run_task({task}) error: {exc}", flush=True)
    finally:
        try:
            await session.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    url        = ENV_URL or "http://localhost:7860"
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"[DEBUG] Connecting to: {url}", flush=True)

    for task in TASKS:
        await run_task(url, llm_client, task)


if __name__ == "__main__":
    asyncio.run(main())
