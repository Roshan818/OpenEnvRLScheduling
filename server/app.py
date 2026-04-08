"""
Smart Factory Scheduling — OpenEnv Server Entry Point
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


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=int(os.getenv("PORT", port)))


if __name__ == "__main__":
    main()
