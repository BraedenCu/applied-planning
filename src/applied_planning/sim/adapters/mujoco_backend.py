from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import mujoco  # type: ignore
except Exception:  # pragma: no cover - optional at dev time
    mujoco = None  # lazy import; class guards usage

from .base import Action, Observation, SimulationAdapter


class MujocoLite6Adapter(SimulationAdapter):
    """MuJoCo simulation adapter for uFactory Lite 6.

    Note: This is a skeleton. It avoids importing MuJoCo at module import time
    so the package can be installed without MuJoCo initially. Instantiation or
    method calls will fail with a clear error if MuJoCo is not available.
    """

    def __init__(self, model_path: str, viewer: bool = True) -> None:
        if mujoco is None:
            raise RuntimeError(
                "MuJoCo is not installed. Please `pip install mujoco` and ensure "
                "the model assets are available."
            )
        self.model_path = model_path
        self.viewer_enabled = viewer
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._viewer = None

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            # MuJoCo doesn't use a global RNG for physics; seed if you randomize initial states
            pass
        mujoco.mj_resetData(self.model, self.data)
        if self.viewer_enabled and self._viewer is None:
            try:
                from mujoco import viewer as mj_viewer  # type: ignore

                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            except Exception as e:
                # Viewer is optional; continue headless, but log a brief warning.
                print(f"MuJoCo viewer could not be launched (running without viewer): {e}")
                self._viewer = None
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        # Expect action as joint velocity or torque; for now assume joint velocity dict
        qvel_cmd = action.get("qvel")
        if qvel_cmd is not None:
            import numpy as np

            self.data.qvel[: len(qvel_cmd)] = np.asarray(qvel_cmd)
        mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()
        obs = {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}
        reward = 0.0
        terminated = False
        info: Dict[str, Any] = {}
        return obs, reward, terminated, info

    def get_state(self) -> Dict[str, Any]:
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}

    def set_state(self, state: Dict[str, Any]) -> None:
        import numpy as np

        qpos = state.get("qpos")
        qvel = state.get("qvel")
        if qpos is not None:
            self.data.qpos[: len(qpos)] = np.asarray(qpos)
        if qvel is not None:
            self.data.qvel[: len(qvel)] = np.asarray(qvel)
        mujoco.mj_forward(self.model, self.data)

    def render(self, mode: str = "human"):
        # For headless, you could render offscreen; left as future work
        return None

    def attach_camera(self, name: str, cfg) -> None:
        # Hook for adding/controlling cameras; left as future work
        return None

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
