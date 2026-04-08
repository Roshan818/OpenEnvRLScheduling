"""
Smart Factory Scheduling — Interactive Gradio Demo
Run: python app.py  →  http://localhost:7860
"""
import asyncio, os
from typing import List, Optional, Tuple
import gradio as gr
from factory_env.env import FactoryEnv
from factory_env.grader import score_episode
from factory_env.models import FactoryAction as Action

_env: Optional[FactoryEnv] = None
_obs = None
_rewards: List[float] = []
_history: List[dict] = []
_step_num: int = 0

STATUS_EMOJI = {"idle": "🟢", "busy": "🔵", "broken": "🔴"}
SYSTEM_PROMPT = "You are a factory scheduler. Reply with ONE action:\n  assign_job <job_id> <machine_id>\n  repair <machine_id>\n  wait"


def _llm_client(provider, api_key):
    if "Claude" in provider:
        import anthropic
        return ("claude", anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY")))
    from openai import OpenAI
    base = "https://api.openai.com/v1" if "OpenAI" in provider else "https://router.huggingface.co/v1"
    return ("openai", OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN"), base_url=base))


def _call_llm(provider_tuple, model, obs, last_reward, step):
    kind, client = provider_tuple
    machines = "\n".join(f"  {m.id}: {m.status}" + (f" ({m.current_job})" if m.current_job else "") for m in obs.machines)
    jobs = "\n".join(f"  {j.id}: t={j.remaining_time} dl={j.deadline} p={j.priority}" for j in obs.pending_jobs) or "  (none)"
    user = f"Step {step}/{obs.max_steps} | t={obs.time} | reward={last_reward:+.2f}\nMachines:\n{machines}\nJobs:\n{jobs}\nAction:"
    try:
        if kind == "claude":
            r = client.messages.create(model=model, max_tokens=50, system=SYSTEM_PROMPT, messages=[{"role":"user","content":user}])
            return r.content[0].text.strip().splitlines()[0]
        else:
            r = client.chat.completions.create(model=model, temperature=0.2, max_tokens=50,
                messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user}])
            return (r.choices[0].message.content or "wait").strip().splitlines()[0]
    except Exception as e:
        return f"wait  # {e}"


def _parse(text):
    try:
        p = text.strip().split()
        if p[0] == "assign_job" and len(p) == 3: return Action(action_type="assign_job", job_id=p[1], machine_id=p[2])
        if p[0] == "repair" and len(p) == 2: return Action(action_type="repair", machine_id=p[1])
    except: pass
    return Action(action_type="wait")


def _heuristic(obs) -> Tuple[Action, str]:
    for m in obs.machines:
        if m.status == "broken": return Action(action_type="repair", machine_id=m.id), f"repair {m.id}"
    for j in sorted(obs.pending_jobs, key=lambda x: (x.deadline, -x.priority)):
        for m in obs.machines:
            if m.status == "idle":
                return Action(action_type="assign_job", job_id=j.id, machine_id=m.id), f"assign_job {j.id} {m.id}"
    return Action(action_type="wait"), "wait"


def _render_state(obs):
    if obs is None: return "*Reset to start*"
    lines = [f"### ⏱ Time: {obs.time} / {obs.max_steps}",
             "\n**Machines**", "| ID | Status | Job |", "|---|---|---|"]
    for m in obs.machines:
        lines.append(f"| {m.id} | {STATUS_EMOJI.get(m.status,'')} {m.status} | {m.current_job or '—'} |")
    lines.append("\n**Pending Jobs**")
    if obs.pending_jobs:
        lines += ["| ID | Remaining | Deadline | Priority |", "|---|---|---|---|"]
        for j in sorted(obs.pending_jobs, key=lambda x: x.deadline):
            urgent = "🔥" if obs.time + j.remaining_time > j.deadline else ""
            lines.append(f"| {j.id} {urgent} | {j.remaining_time} | {j.deadline} | {'★'*j.priority} |")
    else:
        lines.append("*All jobs completed! ✅*")
    if obs.completed_jobs:
        lines.append(f"\n**Completed:** {len(obs.completed_jobs)} ✅")
    return "\n".join(lines)


def _render_log(history):
    if not history: return "*No steps yet*"
    rows = ["| Step | Action | Reward | Done |", "|---|---|---|---|"]
    for h in history[-15:]:
        r = h["reward"]; icon = "🟢" if r > 0.3 else ("🔴" if r < -0.05 else "🟡")
        rows.append(f"| {h['step']} | `{h['action']}` | {icon} {r:+.2f} | {'✅' if h['done'] else ''} |")
    return "\n".join(rows)


def _render_score(rewards, env):
    if not rewards or not env: return ""
    s = score_episode(env)
    bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
    return f"**Score:** {s:.4f}  `[{bar}]`\n**Completed:** {len(env.completed_jobs)}  |  **Late:** {env.late_jobs}  |  **Total Reward:** {sum(rewards):.2f}"


def reset_env(task):
    global _env, _obs, _rewards, _history, _step_num
    _env = FactoryEnv(task=task, seed=42); _obs = _env.reset()
    _rewards = []; _history = []; _step_num = 0
    return _render_state(_obs), _render_log([]), "", f"✅ Reset — **{task}**: {len(_obs.machines)} machines, {len(_obs.pending_jobs)} jobs"


def manual_step(text):
    global _obs, _rewards, _history, _step_num
    if _env is None: return _render_state(None), _render_log([]), "", "⚠ Reset first."
    if _obs.done: return _render_state(_obs), _render_log(_history), _render_score(_rewards, _env), "✅ Episode done."
    _step_num += 1
    _obs = _env.step(_parse(text.strip()))
    r = _obs.reward or 0.0; _rewards.append(r); _history.append({"step": _step_num, "action": text.strip(), "reward": r, "done": _obs.done})
    status = f"Step {_step_num}: `{text.strip()}` → **{r:+.2f}**"
    if _obs.done: status += f"\n\n🏁 Done! Score: **{score_episode(_env):.4f}**"
    return _render_state(_obs), _render_log(_history), _render_score(_rewards, _env), status


def heuristic_step():
    global _obs, _rewards, _history, _step_num
    if _env is None: return _render_state(None), _render_log([]), "", "⚠ Reset first."
    if _obs.done: return _render_state(_obs), _render_log(_history), _render_score(_rewards, _env), "✅ Episode done."
    action, action_text = _heuristic(_obs)
    _step_num += 1
    _obs = _env.step(action)
    r = _obs.reward or 0.0; _rewards.append(r); _history.append({"step": _step_num, "action": f"[H] {action_text}", "reward": r, "done": _obs.done})
    status = f"[Heuristic] Step {_step_num}: `{action_text}` → **{r:+.2f}**"
    if _obs.done: status += f"\n\n🏁 Done! Score: **{score_episode(_env):.4f}**"
    return _render_state(_obs), _render_log(_history), _render_score(_rewards, _env), status


def llm_step(provider, api_key, model):
    global _obs, _rewards, _history, _step_num
    if _env is None: return _render_state(None), _render_log([]), "", "⚠ Reset first.", ""
    if _obs.done: return _render_state(_obs), _render_log(_history), _render_score(_rewards, _env), "✅ Episode done.", ""
    try: client = _llm_client(provider, api_key)
    except Exception as e: return _render_state(_obs), _render_log(_history), "", f"⚠ {e}", ""
    action_text = _call_llm(client, model, _obs, _rewards[-1] if _rewards else 0.0, _step_num + 1)
    action = _parse(action_text)
    if action.action_type == "wait" and (_obs.pending_jobs or any(m.status == "broken" for m in _obs.machines)):
        action, action_text = _heuristic(_obs)
        action_text = f"[fallback] {action_text}"
    _step_num += 1
    _obs = _env.step(action)
    r = _obs.reward or 0.0; _rewards.append(r); _history.append({"step": _step_num, "action": f"[LLM] {action_text}", "reward": r, "done": _obs.done})
    status = f"[LLM] Step {_step_num}: `{action_text}` → **{r:+.2f}**"
    if _obs.done: status += f"\n\n🏁 Done! Score: **{score_episode(_env):.4f}**"
    return _render_state(_obs), _render_log(_history), _render_score(_rewards, _env), status, action_text


def run_full_episode(provider, api_key, model, task):
    global _env, _obs, _rewards, _history, _step_num
    _env = FactoryEnv(task=task, seed=42); _obs = _env.reset()
    _rewards = []; _history = []; _step_num = 0
    try: client = _llm_client(provider, api_key)
    except Exception as e: return _render_state(_obs), _render_log([]), "", f"⚠ {e}", ""
    log_lines = []
    while not _obs.done and _step_num < _obs.max_steps:
        action_text = _call_llm(client, model, _obs, _rewards[-1] if _rewards else 0.0, _step_num + 1)
        action = _parse(action_text)
        if action.action_type == "wait" and (_obs.pending_jobs or any(m.status == "broken" for m in _obs.machines)):
            action, action_text = _heuristic(_obs); action_text = f"[fb] {action_text}"
        _step_num += 1; _obs = _env.step(action)
        r = _obs.reward or 0.0; _rewards.append(r)
        _history.append({"step": _step_num, "action": action_text, "reward": r, "done": _obs.done})
        log_lines.append(f"Step {_step_num:2d}: {action_text:<35s} r={r:+.2f}")
    s = score_episode(_env)
    status = f"🏁 **Done!** Score: **{s:.4f}** | Completed: {len(_env.completed_jobs)} | Late: {_env.late_jobs}"
    return _render_state(_obs), _render_log(_history), _render_score(_rewards, _env), status, "\n".join(log_lines)


def build_ui():
    with gr.Blocks(title="Smart Factory RL") as demo:
        gr.Markdown("# 🏭 Smart Factory Scheduling — Interactive RL Demo")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Setup")
                task_dd = gr.Dropdown(["easy","medium","hard"], value="easy", label="Task")
                provider_dd = gr.Dropdown(["OpenAI (GPT)","Claude (Anthropic)","HuggingFace Router"], value="OpenAI (GPT)", label="Provider")
                api_key_box = gr.Textbox(label="API Key", type="password", placeholder="sk-... or sk-ant-...")
                model_box = gr.Textbox(label="Model", value="gpt-4o-mini")
                reset_btn = gr.Button("🔄 Reset", variant="primary")
                gr.Markdown("### 🎮 Manual")
                manual_input = gr.Textbox(label="Action", placeholder="assign_job J1 M1  |  repair M2  |  wait")
                with gr.Row():
                    manual_btn = gr.Button("▶ Execute")
                    heuristic_btn = gr.Button("🤖 Heuristic Step")
                gr.Markdown("### 🧠 LLM")
                with gr.Row():
                    llm_step_btn = gr.Button("🔮 LLM Step", variant="secondary")
                    llm_ep_btn = gr.Button("⚡ Run Full Episode", variant="primary")
                llm_out = gr.Textbox(label="LLM Output", interactive=False)
                status_md = gr.Markdown("*Press Reset to start*")
            with gr.Column(scale=2):
                gr.Markdown("### 🏭 Factory State")
                state_md = gr.Markdown("*Reset to start*")
                gr.Markdown("### 📊 Score")
                score_md = gr.Markdown("")
                gr.Markdown("### 📋 Step Log")
                log_md = gr.Markdown("*No steps yet*")
        reset_btn.click(reset_env, [task_dd], [state_md, log_md, score_md, status_md])
        manual_btn.click(manual_step, [manual_input], [state_md, log_md, score_md, status_md])
        heuristic_btn.click(heuristic_step, [], [state_md, log_md, score_md, status_md])
        llm_step_btn.click(llm_step, [provider_dd, api_key_box, model_box], [state_md, log_md, score_md, status_md, llm_out])
        llm_ep_btn.click(run_full_episode, [provider_dd, api_key_box, model_box, task_dd], [state_md, log_md, score_md, status_md, llm_out])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, show_error=True, theme=gr.themes.Soft())
