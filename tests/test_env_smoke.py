from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from applied_planning.envs.lite6_pick_place_env import Lite6PickPlaceEnv
from applied_planning.sim.adapters.base import SimulationAdapter, Observation, Action


class DummyAdapter(SimulationAdapter):
    def reset(self, seed: Optional[int] = None) -> Observation:
        rng = np.random.default_rng(seed)
        return {
            "ee_pos": rng.standard_normal(3).astype(np.float32),
            "joint_pos": rng.standard_normal(6).astype(np.float32),
        }

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        obs = {
            "ee_pos": np.zeros(3, dtype=np.float32),
            "joint_pos": np.zeros(6, dtype=np.float32),
        }
        return obs, 0.0, False, {}

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        return None

    def render(self, mode: str = "human"):
        return None

    def attach_camera(self, name: str, cfg) -> None:
        return None

    def close(self) -> None:
        return None


def test_env_reset_and_step():
    env = Lite6PickPlaceEnv(adapter=DummyAdapter())
    obs, _ = env.reset(seed=0)
    assert "ee_pos" in obs and obs["ee_pos"].shape == (3,)
    assert "joint_pos" in obs and obs["joint_pos"].shape == (6,)

    action = np.zeros(6, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
