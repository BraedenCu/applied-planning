from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise RuntimeError("gymnasium is required for envs. Install with `pip install gymnasium`. ") from e

from ..sim.adapters.base import SimulationAdapter, Observation, Action


class Lite6PickPlaceEnv(gym.Env):
    """Skeleton Gymnasium environment that consumes a SimulationAdapter.

    - Observation: dict with 'ee_pos' (3,) and 'joint_pos' (6,) placeholders
    - Action: joint velocity command (6,)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, adapter: Optional[SimulationAdapter] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.adapter = adapter
        self.render_mode = render_mode
        self.observation_space = spaces.Dict(
            {
                "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "joint_pos": spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self._goal = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if self.adapter is not None:
            obs = self.adapter.reset(seed=seed)
            return self._coerce_obs(obs), {}
        # Dummy observation for skeleton operation
        obs = {"ee_pos": np.zeros(3, dtype=np.float32), "joint_pos": np.zeros(6, dtype=np.float32)}
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if self.adapter is not None:
            obs, reward, terminated, info = self.adapter.step({"qvel": action})
            return self._coerce_obs(obs), float(reward), bool(terminated), False, info
        # Dummy transition: move ee_pos towards goal by a fraction
        obs = {"ee_pos": np.zeros(3, dtype=np.float32), "joint_pos": np.zeros(6, dtype=np.float32)}
        dist = float(np.linalg.norm(obs["ee_pos"] - self._goal))
        reward = -dist
        terminated = dist < 1e-2
        info: Dict[str, Any] = {}
        return obs, reward, terminated, False, info

    def render(self):  # noqa: D401 - gym-compatible render
        if self.adapter is not None:
            return self.adapter.render(mode=self.render_mode or "human")
        return None

    @staticmethod
    def _coerce_obs(obs: Observation) -> Observation:
        # Ensure expected keys exist; fill with zeros if missing
        ee = obs.get("ee_pos")
        jp = obs.get("joint_pos")
        if ee is None:
            obs["ee_pos"] = np.zeros(3, dtype=np.float32)
        if jp is None:
            obs["joint_pos"] = np.zeros(6, dtype=np.float32)
        return obs
