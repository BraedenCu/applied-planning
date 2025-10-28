from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import mujoco  # type: ignore
except Exception:  # pragma: no cover - optional at dev time
    mujoco = None  # lazy import; class guards usage

from .base import Action, Observation, SimulationAdapter


class MujocoLite6Adapter(SimulationAdapter):
    """MuJoCo simulation adapter for uFactory Lite 6 with path planning support.

    Note: This is a skeleton. It avoids importing MuJoCo at module import time
    so the package can be installed without MuJoCo initially. Instantiation or
    method calls will fail with a clear error if MuJoCo is not available.
    """

    def __init__(
        self,
        model_path: str,
        viewer: bool = True,
        control_dt: float = 0.01,
        render_mode: str = "passive"
    ) -> None:
        """Initialize MuJoCo adapter for Lite 6.

        Args:
            model_path: Path to MuJoCo XML model
            viewer: If True, attempt to create a viewer (may not work in notebooks)
            control_dt: Control timestep in seconds
            render_mode: "passive" for interactive viewer, "offscreen" for rendering to images,
                        "none" to disable rendering
        """
        if mujoco is None:
            raise RuntimeError(
                "MuJoCo is not installed. Please `pip install mujoco` and ensure "
                "the model assets are available."
            )
        self.model_path = model_path
        self.viewer_enabled = viewer and render_mode == "passive"
        self.render_mode = render_mode
        self.control_dt = control_dt
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._viewer = None
        self._renderer = None
        self._current_path: Optional[List[np.ndarray]] = None
        self._path_index = 0

        # Initialize offscreen renderer if requested
        if render_mode == "offscreen":
            self._init_offscreen_renderer()

    def _init_offscreen_renderer(self) -> None:
        """Initialize offscreen renderer for notebook/headless use."""
        try:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            print("Offscreen renderer initialized (notebook-friendly mode)")
        except Exception as e:
            print(f"Could not initialize offscreen renderer: {e}")
            self._renderer = None

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            # MuJoCo doesn't use a global RNG for physics; seed if you randomize initial states
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)

        # Only try to launch viewer if explicitly enabled and not already created
        if self.viewer_enabled and self._viewer is None:
            try:
                # Try using the standard viewer which works better on macOS
                import mujoco.viewer as mj_viewer  # type: ignore

                # Launch in a non-blocking way
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
                print("✓ MuJoCo viewer launched")
                print("  (If you don't see a window, the viewer may not be supported)")
                print("  Alternative: Use render_mode='offscreen' for visualization")
            except Exception as e:
                # Viewer is optional; continue headless
                print(f"⚠ Viewer could not be launched: {e}")
                print(f"  Running in headless mode")
                print(f"  Tip: Use render_mode='offscreen' for notebook visualization")
                self.viewer_enabled = False
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

    def render(self, mode: str = "human", camera_id: Optional[int] = None):
        """Render the current scene.

        Args:
            mode: "human" for viewer sync, "rgb_array" for offscreen rendering
            camera_id: Optional camera ID for offscreen rendering

        Returns:
            None for "human" mode, RGB array for "rgb_array" mode
        """
        if mode == "human":
            if self._viewer is not None:
                self._viewer.sync()
            return None
        elif mode == "rgb_array":
            if self._renderer is None:
                # Initialize renderer on-demand
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)

            # Update scene
            mujoco.mj_forward(self.model, self.data)

            # Render from specified camera
            if camera_id is not None:
                self._renderer.update_scene(self.data, camera=camera_id)
            else:
                self._renderer.update_scene(self.data)

            return self._renderer.render()
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

    def plan_and_execute_path(
        self,
        goal: np.ndarray,
        planner_type: str = "rrt",
        joint_limits: Optional[Dict[int, Tuple[float, float]]] = None,
        collision_fn: Optional[callable] = None,
        execute: bool = True
    ) -> Optional[List[np.ndarray]]:
        """Plan a path from current configuration to goal and optionally execute it.

        Args:
            goal: Goal joint configuration (6DOF)
            planner_type: Type of planner ("rrt" or "linear")
            joint_limits: Joint limits as {joint_idx: (min, max)}
            collision_fn: Optional collision checking function
            execute: If True, execute the path immediately

        Returns:
            List of waypoints in the planned path, or None if planning failed
        """
        from ...control.planners import plan_joint_path

        start = self.data.qpos[:6].copy()

        constraints = {
            "joint_limits": joint_limits,
            "collision_fn": collision_fn
        }

        path = plan_joint_path(start, goal, self.model, constraints, planner_type)

        if path is not None:
            self._current_path = path
            self._path_index = 0

            if execute:
                self.execute_path(path)

        return path

    def execute_path(
        self,
        path: List[np.ndarray],
        velocity_control: bool = False,
        speed_factor: float = 1.0,
        steps_per_waypoint: int = 10
    ) -> None:
        """Execute a planned path in the simulator with smooth motion.

        Args:
            path: List of joint waypoints to follow
            velocity_control: If True, use velocity control; else position control (recommended)
            speed_factor: Scaling factor for execution speed (1.0 = normal, <1.0 = slower)
            steps_per_waypoint: Number of simulation steps between waypoints for smooth motion
        """
        import time

        if velocity_control:
            # Velocity control mode
            for i, waypoint in enumerate(path):
                if i < len(path) - 1:
                    next_waypoint = path[i + 1]
                    velocity = (next_waypoint - waypoint) / self.control_dt * speed_factor
                    action = {"qvel": velocity}
                else:
                    action = {"qvel": np.zeros(6)}

                self.step(action)

                if self._viewer is not None:
                    time.sleep(self.control_dt / speed_factor)
        else:
            # Position control mode (smoother visualization)
            for i in range(len(path) - 1):
                current = path[i]
                next_waypoint = path[i + 1]

                # Interpolate between waypoints for smooth motion
                for step in range(steps_per_waypoint):
                    alpha = step / steps_per_waypoint
                    interpolated = (1 - alpha) * current + alpha * next_waypoint

                    # Set position directly
                    self.set_state({"qpos": interpolated, "qvel": np.zeros(len(interpolated))})

                    # Step simulation for physics
                    mujoco.mj_step(self.model, self.data)

                    # Sync viewer
                    if self._viewer is not None:
                        self._viewer.sync()
                        time.sleep(self.control_dt / speed_factor)

            # Final position
            self.set_state({"qpos": path[-1], "qvel": np.zeros(len(path[-1]))})

    def get_joint_limits(self) -> Dict[int, Tuple[float, float]]:
        """Extract joint limits from the MuJoCo model.

        Returns:
            Dictionary mapping joint index to (min, max) limits
        """
        joint_limits = {}
        for i in range(min(6, self.model.njnt)):
            qpos_idx = self.model.jnt_qposadr[i]
            if self.model.jnt_limited[i]:
                joint_limits[qpos_idx] = (
                    float(self.model.jnt_range[i, 0]),
                    float(self.model.jnt_range[i, 1])
                )
        return joint_limits

    def check_collision(self, q: Optional[np.ndarray] = None) -> bool:
        """Check if the current or given configuration is in collision.

        Args:
            q: Optional joint configuration to check; if None, use current state

        Returns:
            True if collision detected, False otherwise
        """
        if q is not None:
            # Temporarily set configuration
            old_qpos = self.data.qpos.copy()
            self.data.qpos[:len(q)] = q
            mujoco.mj_forward(self.model, self.data)

        # Check for collisions using MuJoCo contact detection
        has_collision = self.data.ncon > 0

        if q is not None:
            # Restore original configuration
            self.data.qpos[:] = old_qpos
            mujoco.mj_forward(self.model, self.data)

        return has_collision

    def get_end_effector_pose(self) -> np.ndarray:
        """Get current end-effector pose.

        Returns:
            7D pose vector [x, y, z, qw, qx, qy, qz]
        """
        # Assuming the end-effector is the last body or a specific site
        # You may need to adjust the body/site name based on your MJCF
        try:
            ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
            pos = self.data.xpos[ee_body_id].copy()
            quat = self.data.xquat[ee_body_id].copy()  # [w, x, y, z] in MuJoCo
            return np.concatenate([pos, quat])
        except Exception:
            # Fallback: use the last body
            pos = self.data.xpos[-1].copy()
            quat = self.data.xquat[-1].copy()
            return np.concatenate([pos, quat])

    def render_notebook(self, width: int = 640, height: int = 480):
        """Render current state for display in Jupyter notebook.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            RGB array suitable for display with matplotlib or IPython.display
        """
        if self._renderer is None or \
           self._renderer.width != width or \
           self._renderer.height != height:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)

        mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data)
        return self._renderer.render()
