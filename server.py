"""
Smart Factory Scheduling — OpenEnv Server
==========================================
Routes (HTTP + WebSocket):
  GET  /          →  Gradio UI (set ENABLE_WEB_INTERFACE=1) or redirect to /web
  GET  /health    →  {"status": "healthy"}
  POST /reset     →  reset episode, returns observation
  POST /step      →  execute action, returns observation + reward + done
  GET  /state     →  current environment state
  GET  /schema    →  action / observation JSON schemas
  WS   /ws        →  persistent WebSocket session (used by FactoryEnvClient)

Set ENABLE_WEB_INTERFACE=1 to enable the built-in Gradio UI at /web.
"""
import os
from openenv.core import create_app
from factory_env.env import FactoryEnv
from factory_env.models import FactoryAction, FactoryObservation

TASK = os.getenv("FACTORY_TASK", "easy")

app = create_app(
    env=lambda: FactoryEnv(task=TASK, seed=42),
    action_cls=FactoryAction,
    observation_cls=FactoryObservation,
    env_name="factory_env",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
