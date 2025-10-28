from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..sim.adapters.base import Action, Observation, Goal


class Controller(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: Observation, goal: Goal) -> Action: ...


@dataclass
class SimpleJointVelocityController:
    """Very simple P controller in joint space for skeleton/testing.

    Expects obs["joint_pos"] and goal["joint_pos"]. Outputs action {"qvel": ...}.
    """

    kp: float = 0.5

    def reset(self) -> None:  # pragma: no cover - nothing to reset in this simple controller
        return None

    def act(self, obs: Observation, goal: Goal) -> Action:
        q = np.asarray(obs.get("joint_pos", np.zeros(6, dtype=np.float32)), dtype=np.float32)
        q_goal = np.asarray(goal.get("joint_pos", np.zeros_like(q)), dtype=np.float32)
        qerr = q_goal - q
        qvel = self.kp * qerr
        return {"qvel": qvel}
