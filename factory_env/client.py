# Copyright (c) Meta Platforms, Inc. and affiliates — OpenEnv compliant client
"""
FactoryEnvClient — WebSocket client for the Smart Factory Scheduling environment.

Usage:
    # Connect to a running server directly
    async with FactoryEnvClient(base_url="http://localhost:7860") as env:
        result = await env.reset()
        result = await env.step(FactoryAction(action_type="wait"))

    # Spin up a Docker container and connect
    env = await FactoryEnvClient.from_docker_image("factory-env:latest")

    # Connect to a HuggingFace Space
    env = await FactoryEnvClient.from_env("Aldrimore/RLScheduling")
"""

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from factory_env.models import FactoryAction, FactoryObservation


class FactoryEnvClient(EnvClient[FactoryAction, FactoryObservation, State]):
    """
    WebSocket client for the Smart Factory Scheduling environment.

    Connects to a running factory_env server over WebSocket for
    efficient multi-step interactions.
    """

    def _step_payload(self, action: FactoryAction) -> Dict[str, Any]:
        """Serialize FactoryAction to JSON payload for the step message."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FactoryObservation]:
        """Parse server response into StepResult[FactoryObservation]."""
        obs_data = payload.get("observation", {})
        obs = FactoryObservation(
            **obs_data,
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State.

        State has extra="allow" so all FactoryState fields (late_jobs,
        completed_jobs, pending_jobs, time, task) are accessible as attributes.
        """
        return State(**payload)
