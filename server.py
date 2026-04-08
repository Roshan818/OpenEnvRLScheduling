"""
OpenEnv HTTP Server — Smart Factory Scheduling
Routes: GET /health  POST /reset  POST /step  GET /state  GET /schema
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
