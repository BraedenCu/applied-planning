from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise RuntimeError("gymnasium is required for envs. Install with `pip install gymnasium`.") from e

from ..sim.adapters.mujoco_backend import MujocoLite6Adapter


class Lite6CubePointingEnv(gym.Env):
    """Gymnasium environment for training a robot to point at a cube.

    The robot must learn to move its end-effector as close as possible to the cube's centroid.

    Observation Space:
        - cube_pos: (3,) - xyz position of the cube centroid
        - joint_pos: (6,) - current joint positions
        - ee_pos: (3,) - current end-effector position

    Action Space:
        - (6,) - target joint positions (normalized to [-1, 1])

    Reward:
        - Negative Euclidean distance between end-effector and cube
        - Optional bonus for getting very close

    Episode Termination:
        - Success: end-effector within success_threshold of cube
        - Timeout: max_steps exceeded
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        model_path: str,
        render_mode: Optional[str] = None,
        max_steps: int = 500,
        success_threshold: float = 0.05,  # 5cm
        reward_scale: float = 10.0,
        cube_placement_radius: float = 0.3,
        ee_site_name: str = "end_effector"
    ):
        """Initialize the cube pointing environment.

        Args:
            model_path: Path to MuJoCo XML model file
            render_mode: "human" for viewer, "rgb_array" for offscreen rendering
            max_steps: Maximum steps per episode
            success_threshold: Distance threshold for success (meters)
            reward_scale: Multiplier for reward shaping
            cube_placement_radius: Radius for random cube placement
            ee_site_name: Name of end-effector site in model
        """
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.reward_scale = reward_scale
        self.cube_placement_radius = cube_placement_radius
        self._current_step = 0

        # Initialize MuJoCo adapter with a single cube
        self.adapter = MujocoLite6Adapter(
            model_path=model_path,
            num_cubes=1,
            cube_placement_radius=cube_placement_radius,
            viewer=(render_mode == "human"),
            render_mode="passive" if render_mode == "human" else "offscreen",
            ee_site_name=ee_site_name
        )

        # Get joint limits from model
        self.joint_limits = self.adapter.get_joint_limits()

        # Define observation space: cube_pos (3) + joint_pos (6) + ee_pos (3) = 12
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

        # Define action space: 6 joint positions normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        # Store joint limit ranges for denormalization
        self._joint_mins = np.array([self.joint_limits.get(i, (-np.pi, np.pi))[0] for i in range(6)])
        self._joint_maxs = np.array([self.joint_limits.get(i, (-np.pi, np.pi))[1] for i in range(6)])

        # Track previous distance for reward shaping
        self._prev_distance = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Reset the adapter (randomizes cube position)
        self.adapter.reset(seed=seed, randomize_cubes=True)

        # Reset step counter
        self._current_step = 0

        # Get initial observation
        obs = self._get_observation()

        # Initialize previous distance
        cube_pos = obs[:3]
        ee_pos = obs[9:12]
        self._prev_distance = float(np.linalg.norm(ee_pos - cube_pos))

        info = {
            "cube_pos": cube_pos,
            "ee_pos": ee_pos,
            "distance": self._prev_distance
        }

        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        """Execute one step in the environment.

        Args:
            action: Normalized joint positions [-1, 1]

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success)
            truncated: Whether episode was truncated (timeout)
            info: Additional information
        """
        # Denormalize action to actual joint positions
        action = np.clip(action, -1.0, 1.0)
        target_joints = self._denormalize_action(action)

        # Execute action by setting joint positions directly
        self.adapter.set_state({
            "qpos": target_joints,
            "qvel": np.zeros(len(target_joints))
        })

        # Step simulation a few times for physics to settle
        import mujoco
        for _ in range(10):
            mujoco.mj_step(self.adapter.model, self.adapter.data)
            if self.adapter._viewer is not None:
                self.adapter._viewer.sync()

        # Get new observation
        obs = self._get_observation()
        cube_pos = obs[:3]
        ee_pos = obs[9:12]

        # Calculate reward
        distance = float(np.linalg.norm(ee_pos - cube_pos))
        reward = self._compute_reward(distance)

        # Update step counter
        self._current_step += 1

        # Check termination conditions
        terminated = distance < self.success_threshold
        truncated = self._current_step >= self.max_steps

        # Build info dict
        info = {
            "cube_pos": cube_pos,
            "ee_pos": ee_pos,
            "distance": distance,
            "is_success": terminated
        }

        # Update previous distance for next step
        self._prev_distance = distance

        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def _get_observation(self) -> np.ndarray:
        """Get current observation.

        Returns:
            Flat observation vector [cube_pos(3), joint_pos(6), ee_pos(3)]
        """
        # Get cube position (index 0 since we only have 1 cube)
        cube_positions = self.adapter.get_cube_positions()
        cube_pos = cube_positions[0] if len(cube_positions) > 0 else np.zeros(3)

        # Get joint positions (first 6 joints)
        joint_pos = self.adapter.data.qpos[:6].copy()

        # Get end-effector position
        ee_pos = self.adapter.get_ee_position()

        # Concatenate into flat observation
        obs = np.concatenate([cube_pos, joint_pos, ee_pos])

        return obs

    def _compute_reward(self, distance: float) -> float:
        """Compute reward based on distance to cube.

        Args:
            distance: Current distance from end-effector to cube

        Returns:
            Reward value
        """
        # Primary reward: negative distance (encourage getting closer)
        reward = -distance * self.reward_scale

        # Bonus for improvement from previous step
        if self._prev_distance is not None:
            improvement = self._prev_distance - distance
            reward += improvement * self.reward_scale * 2.0  # Extra weight on improvement

        # Success bonus
        if distance < self.success_threshold:
            reward += 100.0

        return reward

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action [-1, 1] to actual joint positions.

        Args:
            action: Normalized action in [-1, 1]

        Returns:
            Actual joint positions within joint limits
        """
        # Map from [-1, 1] to [joint_min, joint_max]
        return self._joint_mins + (action + 1.0) * 0.5 * (self._joint_maxs - self._joint_mins)

    def render(self):
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "human":
            if self.adapter._viewer is not None:
                self.adapter._viewer.sync()
            return None
        elif self.render_mode == "rgb_array":
            return self.adapter.render_notebook()
        return None

    def close(self):
        """Clean up resources."""
        self.adapter.close()
