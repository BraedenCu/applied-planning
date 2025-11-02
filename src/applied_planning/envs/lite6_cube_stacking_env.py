from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..sim.adapters.mujoco_backend import MujocoLite6Adapter


class Lite6CubeStackingEnv(gym.Env):
    """Gymnasium environment for training a robot to stack cubes.

    The robot must learn to pick up one cube and stack it on top of another cube,
    while avoiding self-collisions between robot links.

    Observation Space:
        - cube1_pos: (3,) - xyz position of the first cube (to be picked up)
        - cube2_pos: (3,) - xyz position of the second cube (base/target)
        - joint_pos: (6,) - current joint positions
        - ee_pos: (3,) - current end-effector position
        - gripper_pos: (1,) - gripper opening (0=closed, 1=open)

    Action Space:
        - (7,) - [target joint positions (6), gripper action (1)]
          All normalized to [-1, 1]

    Reward:
        - Negative vertical distance between cubes (encourage stacking)
        - Bonus for cube1 being above cube2
        - Penalty for cubes being far apart horizontally
        - Large penalty for self-collisions
        - Success bonus for achieving stack

    Episode Termination:
        - Success: cube1 stacked on cube2 within threshold
        - Collision: self-collision detected (optional)
        - Timeout: max_steps exceeded
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        model_path: str,
        render_mode: Optional[str] = None,
        max_steps: int = 500,
        success_threshold: float = 0.08,  # 8cm vertical alignment
        horizontal_threshold: float = 0.05,  # 5cm horizontal alignment
        reward_scale: float = 10.0,
        cube_placement_radius: float = 0.3,
        ee_site_name: str = "end_effector",
        collision_penalty: float = 100.0,
        terminate_on_collision: bool = False
    ):
        """Initialize the cube stacking environment.

        Args:
            model_path: Path to MuJoCo XML model file
            render_mode: "human" for viewer, "rgb_array" for offscreen rendering
            max_steps: Maximum steps per episode
            success_threshold: Vertical distance threshold for success (meters)
            horizontal_threshold: Horizontal distance threshold for alignment (meters)
            reward_scale: Multiplier for reward shaping
            cube_placement_radius: Radius for random cube placement
            ee_site_name: Name of end-effector site in model
            collision_penalty: Penalty for self-collision (default: 100.0)
            terminate_on_collision: If True, end episode on self-collision (default: False)
        """
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.horizontal_threshold = horizontal_threshold
        self.reward_scale = reward_scale
        self.cube_placement_radius = cube_placement_radius
        self.collision_penalty = collision_penalty
        self.terminate_on_collision = terminate_on_collision
        self._current_step = 0

        # Initialize MuJoCo adapter with TWO cubes
        self.adapter = MujocoLite6Adapter(
            model_path=model_path,
            num_cubes=2,
            cube_placement_radius=cube_placement_radius,
            viewer=(render_mode == "human"),
            render_mode="passive" if render_mode == "human" else "offscreen",
            ee_site_name=ee_site_name
        )

        # Get joint limits from model
        self.joint_limits = self.adapter.get_joint_limits()

        # Define observation space: cube1_pos (3) + cube2_pos (3) + joint_pos (6) + ee_pos (3) + gripper (1) = 16
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32
        )

        # Define action space: 6 joint positions + 1 gripper action, normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # Store joint limit ranges for denormalization
        self._joint_mins = np.array([self.joint_limits.get(i, (-np.pi, np.pi))[0] for i in range(6)])
        self._joint_maxs = np.array([self.joint_limits.get(i, (-np.pi, np.pi))[1] for i in range(6)])

        # Track previous distance for reward shaping
        self._prev_vertical_distance = None
        self._prev_horizontal_distance = None

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

        # Reset the adapter (randomizes cube positions)
        self.adapter.reset(seed=seed, randomize_cubes=True)

        # Reset step counter
        self._current_step = 0

        # Get initial observation
        obs = self._get_observation()

        # Initialize previous distances
        cube1_pos = obs[:3]
        cube2_pos = obs[3:6]
        self._prev_vertical_distance = float(abs(cube1_pos[2] - cube2_pos[2]))
        horizontal_dist = np.linalg.norm(cube1_pos[:2] - cube2_pos[:2])
        self._prev_horizontal_distance = float(horizontal_dist)

        info = {
            "cube1_pos": cube1_pos,
            "cube2_pos": cube2_pos,
            "vertical_distance": self._prev_vertical_distance,
            "horizontal_distance": self._prev_horizontal_distance,
            "is_grasping": False,  # Not grasping at start
            "gripper_pos": self.adapter.get_gripper_position()
        }

        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        """Execute one step in the environment.

        Args:
            action: Normalized [joint positions (6), gripper action (1)] in [-1, 1]

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success or collision)
            truncated: Whether episode was truncated (timeout)
            info: Additional information
        """
        # Denormalize action to actual joint positions
        action = np.clip(action, -1.0, 1.0)
        target_joints = self._denormalize_action(action[:6])
        gripper_action = action[6]  # -1=close, 1=open

        # Check for self-collision BEFORE executing the action
        has_collision = self.adapter.check_self_collision(target_joints)

        # Execute action: set joint positions AND gripper
        self.adapter.set_state({
            "qpos": target_joints,
            "qvel": np.zeros(len(target_joints)),
            "gripper": gripper_action  # Control gripper opening
        })

        # Step simulation a few times for physics to settle
        import mujoco
        for _ in range(10):
            mujoco.mj_step(self.adapter.model, self.adapter.data)
            if self.adapter._viewer is not None:
                self.adapter._viewer.sync()

        # Get new observation
        obs = self._get_observation()
        cube1_pos = obs[:3]
        cube2_pos = obs[3:6]

        # Calculate distances
        vertical_distance = float(abs(cube1_pos[2] - cube2_pos[2]))
        horizontal_distance = float(np.linalg.norm(cube1_pos[:2] - cube2_pos[:2]))

        # Check if cube1 is being grasped
        is_grasping = self.adapter.check_gripper_grasping(cube_idx=0)

        # Calculate reward
        reward = self._compute_reward(
            vertical_distance,
            horizontal_distance,
            cube1_pos[2],
            cube2_pos[2],
            has_collision,
            is_grasping
        )

        # Update step counter
        self._current_step += 1

        # Check termination conditions
        # Success: cube1 is above cube2 with proper alignment
        is_stacked = (
            cube1_pos[2] > cube2_pos[2] and  # cube1 is above cube2
            vertical_distance < self.success_threshold and
            horizontal_distance < self.horizontal_threshold
        )

        collision_terminate = has_collision and self.terminate_on_collision
        terminated = is_stacked or collision_terminate
        truncated = self._current_step >= self.max_steps

        # Build info dict
        info = {
            "cube1_pos": cube1_pos,
            "cube2_pos": cube2_pos,
            "vertical_distance": vertical_distance,
            "horizontal_distance": horizontal_distance,
            "is_success": is_stacked,
            "has_collision": has_collision,
            "is_grasping": is_grasping,
            "gripper_pos": self.adapter.get_gripper_position()
        }

        # Update previous distances for next step
        self._prev_vertical_distance = vertical_distance
        self._prev_horizontal_distance = horizontal_distance

        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def _get_observation(self) -> np.ndarray:
        """Get current observation.

        Returns:
            Flat observation vector [cube1_pos(3), cube2_pos(3), joint_pos(6), ee_pos(3), gripper(1)]
        """
        # Get cube positions
        cube_positions = self.adapter.get_cube_positions()
        cube1_pos = cube_positions[0] if len(cube_positions) > 0 else np.zeros(3)
        cube2_pos = cube_positions[1] if len(cube_positions) > 1 else np.zeros(3)

        # Get joint positions (first 6 joints)
        joint_pos = self.adapter.data.qpos[:6].copy()

        # Get end-effector position
        ee_pos = self.adapter.get_ee_position()

        # Get gripper position using the adapter's method
        gripper_pos = np.array([self.adapter.get_gripper_position()])

        # Concatenate into flat observation
        obs = np.concatenate([cube1_pos, cube2_pos, joint_pos, ee_pos, gripper_pos])

        return obs

    def _compute_reward(
        self,
        vertical_distance: float,
        horizontal_distance: float,
        cube1_z: float,
        cube2_z: float,
        has_collision: bool = False,
        is_grasping: bool = False
    ) -> float:
        """Compute reward based on stacking progress and collision status.

        Args:
            vertical_distance: Vertical distance between cubes
            horizontal_distance: Horizontal distance between cubes
            cube1_z: Z position of cube1
            cube2_z: Z position of cube2
            has_collision: Whether a self-collision occurred
            is_grasping: Whether cube1 is being grasped

        Returns:
            Reward value
        """
        # Primary reward: minimize vertical distance when cube1 is above cube2
        reward = 0.0

        # IMPORTANT: Reward grasping cube1 (critical for learning to pick up)
        if is_grasping:
            reward += 100.0  # Large reward for grasping

            # Extra rewards when grasping and moving cube1 upward
            if cube1_z > 0.05:  # Cube lifted off ground
                reward += 50.0 * cube1_z  # Reward proportional to height

        # Encourage cube1 to be above cube2
        if cube1_z > cube2_z:
            # Reward for small vertical distance when cube1 is above
            reward += (1.0 - vertical_distance) * self.reward_scale

            # Bonus for horizontal alignment
            reward += (1.0 - horizontal_distance) * self.reward_scale * 0.5

            # Extra bonus for very close vertical distance
            if vertical_distance < 0.1:  # Within 10cm
                reward += 50.0
        else:
            # Penalty when cube1 is below cube2 (wrong direction)
            reward -= vertical_distance * self.reward_scale * 0.5

        # Reward improvement in vertical distance
        if self._prev_vertical_distance is not None:
            vertical_improvement = self._prev_vertical_distance - vertical_distance
            reward += vertical_improvement * self.reward_scale * 2.0

        # Reward improvement in horizontal alignment
        if self._prev_horizontal_distance is not None:
            horizontal_improvement = self._prev_horizontal_distance - horizontal_distance
            reward += horizontal_improvement * self.reward_scale

        # Success bonus (cube properly stacked)
        if (cube1_z > cube2_z and
            vertical_distance < self.success_threshold and
            horizontal_distance < self.horizontal_threshold):
            reward += 200.0

        # Collision penalty (strongly discourage self-collisions)
        if has_collision:
            reward -= self.collision_penalty

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
