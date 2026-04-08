"""
Combined Server — Smart Factory Scheduling
- Root /         → Gradio interactive UI (humans)
- /health        → {"status": "healthy"}
- /reset /step /state /schema → OpenEnv API (automated judges)
"""
import os
import gradio as gr
from openenv.core import create_app
from factory_env.env import FactoryEnv
from factory_env.models import FactoryAction, FactoryObservation
from app import build_ui

TASK = os.getenv("FACTORY_TASK", "easy")

# 1. OpenEnv FastAPI app — registers /health /reset /step /state /schema
app = create_app(
    env=lambda: FactoryEnv(task=TASK, seed=42),
    action_cls=FactoryAction,
    observation_cls=FactoryObservation,
    env_name="factory_env",
)

# 2. Mount Gradio UI at root — API routes above take priority over this mount
app = gr.mount_gradio_app(app, build_ui(), path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
