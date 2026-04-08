"""
RL Training Loop — Smart Factory Scheduling
============================================
Strategy: Online In-Context RL — best trajectory fed as few-shot example each episode.

Usage:
  export OPENAI_API_KEY=sk-...          # OpenAI
  export ANTHROPIC_API_KEY=sk-ant-...   # Claude
  python train.py --task easy --episodes 10 --provider openai
  python train.py --task medium --episodes 10 --provider claude
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from factory_env.env import FactoryEnv
from factory_env.grader import score_episode
from factory_env.models import FactoryAction as Action


def get_openai_client():
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    base = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
    return OpenAI(api_key=key, base_url=base)


def get_claude_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@dataclass
class Step:
    step: int
    obs_text: str
    action_text: str
    reward: float
    done: bool


@dataclass
class Episode:
    episode_num: int
    task: str
    steps: List[Step] = field(default_factory=list)
    total_reward: float = 0.0
    score: float = 0.0
    completed: int = 0
    late: int = 0

    def to_few_shot(self, max_steps: int = 6) -> str:
        lines = [f"# Best trajectory so far (score={self.score:.2f}, completed={self.completed} jobs)"]
        for s in self.steps[:max_steps]:
            lines.append(f"[Obs] {s.obs_text}")
            lines.append(f"[Action] {s.action_text}  → reward: {s.reward:+.2f}")
        return "\n".join(lines)


SYSTEM_PROMPT = """You are an expert factory scheduling AI.
Goal: complete all jobs before deadlines, keep machines busy, repair broken machines.
Actions (one per step):
  assign_job <job_id> <machine_id>
  repair <machine_id>
  wait
Tips: Fix broken machines first. Sort by earliest deadline. High-priority jobs give bonus reward."""


def obs_to_text(obs) -> str:
    machines = ", ".join(f"{m.id}:{m.status}" + (f"({m.current_job})" if m.current_job else "") for m in obs.machines)
    jobs = ", ".join(f"{j.id}[t={j.remaining_time},dl={j.deadline},p={j.priority}]" for j in obs.pending_jobs) or "none"
    return f"t={obs.time} | machines: {machines} | pending: {jobs}"


def call_llm(messages: list, provider: str, client, model: str) -> str:
    try:
        if provider == "claude":
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msgs = [m for m in messages if m["role"] != "system"]
            resp = client.messages.create(model=model, max_tokens=60, system=system, messages=user_msgs)
            return resp.content[0].text.strip().splitlines()[0]
        else:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=60)
            return (resp.choices[0].message.content or "wait").strip().splitlines()[0]
    except Exception as e:
        print(f"  [LLM error] {e}")
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


def run_episode(task, episode_num, provider, client, model, best_episode, seed=42, verbose=True) -> Episode:
    env = FactoryEnv(task=task, seed=seed)
    obs = env.reset()
    last_reward = 0.0
    ep = Episode(episode_num=episode_num, task=task)

    if verbose:
        print(f"\n  Episode {episode_num} | task={task} | seed={seed}")
        print(f"  {len(obs.machines)} machines, {len(obs.pending_jobs)} jobs, {obs.max_steps} steps")

    for step in range(1, obs.max_steps + 1):
        if obs.done:
            break

        obs_text = obs_to_text(obs)
        few_shot = best_episode.to_few_shot() if best_episode and step == 1 else ""
        user = f"{few_shot}\n\n---\n" if few_shot else ""
        user += f"Step {step} | Last reward: {last_reward:+.2f}\n{obs_text}\n\nAction:"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        action_text = call_llm(messages, provider, client, model)
        action = parse_action(action_text)

        if action.action_type == "wait" and (obs.pending_jobs or any(m.status == "broken" for m in obs.machines)):
            action, action_text = heuristic_action(obs)

        obs = env.step(action)
        reward = obs.reward or 0.0
        last_reward = reward
        ep.steps.append(Step(step, obs_text, action_text, reward, obs.done))
        ep.total_reward += reward

        if verbose:
            marker = "✓" if reward > 0.5 else ("✗" if reward < -0.05 else "·")
            print(f"    [{marker}] step={step:2d}  {action_text:<30s}  r={reward:+.2f}")

        if obs.done:
            break

    ep.score = score_episode(env)
    ep.completed = len(env.completed_jobs)
    ep.late = env.late_jobs

    if verbose:
        print(f"  → score={ep.score:.4f}  completed={ep.completed}  late={ep.late}")

    return ep


def train(task, num_episodes, provider, model, save_dir="runs", verbose=True):
    print(f"\n{'='*60}")
    print(f"  Smart Factory RL Training")
    print(f"  Task: {task} | Episodes: {num_episodes} | Provider: {provider} | Model: {model}")
    print(f"{'='*60}")

    client = get_claude_client() if provider == "claude" else get_openai_client()
    Path(save_dir).mkdir(exist_ok=True)

    scores = []
    best_episode = None

    for ep_num in range(1, num_episodes + 1):
        ep = run_episode(task, ep_num, provider, client, model, best_episode, seed=42 + ep_num - 1, verbose=verbose)
        scores.append(ep.score)
        if best_episode is None or ep.score > best_episode.score:
            best_episode = ep
            print(f"  ★ New best: score={ep.score:.4f}")
        if ep_num < num_episodes:
            time.sleep(1.0)

    print(f"\n{'='*60}")
    print(f"  Training Complete — {num_episodes} episodes | Task: {task}")
    print(f"  First: {scores[0]:.4f} | Last: {scores[-1]:.4f} | Best: {max(scores):.4f}")
    print(f"\n  Score per episode:")
    for i, s in enumerate(scores, 1):
        print(f"    ep{i:02d}: {s:.4f}  {'█' * int(s * 20)}")

    out = Path(save_dir) / f"{task}_{provider}_{num_episodes}ep.json"
    out.write_text(json.dumps({"task": task, "provider": provider, "model": model, "num_episodes": num_episodes, "scores": scores, "best_score": max(scores), "final_score": scores[-1]}, indent=2))
    print(f"\n  Results saved → {out}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--provider", default="openai", choices=["openai", "claude"])
    parser.add_argument("--model", default="")
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not args.model:
        args.model = "claude-sonnet-4-6" if args.provider == "claude" else "gpt-4o-mini"
    train(args.task, args.episodes, args.provider, args.model, args.save_dir, not args.quiet)


if __name__ == "__main__":
    main()
