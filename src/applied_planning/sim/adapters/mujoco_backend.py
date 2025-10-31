from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import time
import mujoco
import mujoco.viewer as mj_viewer  # type: ignore

from ...control.planners import plan_joint_path, plan_cartesian_path
from ...control.kinematics import MujocoIKSolver, compute_forward_kinematics
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
        render_mode: str = "passive",
        ee_site_name: str = "attachment_site",
        num_cubes: int = 0,
        cube_placement_radius: float = 0.3
    ) -> None:
        """Initialize MuJoCo adapter for Lite 6.

        Args:
            model_path: Path to MuJoCo XML model
            viewer: If True, attempt to create a viewer (may not work in notebooks)
            control_dt: Control timestep in seconds
            render_mode: "passive" for interactive viewer, "offscreen" for rendering to images,
                        "none" to disable rendering
            ee_site_name: Name of end-effector site/body in the model (default: "attachment_site" for Lite6)
            num_cubes: Number of cubes to add to the scene (default: 0)
            cube_placement_radius: Radius within which to randomly place cubes (default: 0.3m)
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
        self.ee_site_name = ee_site_name
        self.num_cubes = num_cubes
        self.cube_placement_radius = cube_placement_radius

        # Load model with cubes if requested
        if num_cubes > 0:
            self.model = self._load_model_with_cubes(model_path, num_cubes, cube_placement_radius)
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)

        self.data = mujoco.MjData(self.model)
        self._viewer = None
        self._renderer = None
        self._current_path: Optional[List[np.ndarray]] = None
        self._path_index = 0

        # Initialize IK solver
        self._ik_solver = MujocoIKSolver(self.model, self.data, ee_site_name)

        # Initialize offscreen renderer if requested
        if render_mode == "offscreen":
            self._init_offscreen_renderer()

    def _load_model_with_cubes(
        self,
        model_path: str,
        num_cubes: int,
        placement_radius: float
    ) -> mujoco.MjModel:
        """Load MuJoCo model and add cubes to the scene.

        Args:
            model_path: Path to base MuJoCo XML model
            num_cubes: Number of cubes to add
            placement_radius: Radius within which to place cubes randomly

        Returns:
            MuJoCo model with cubes added
        """
        import xml.etree.ElementTree as ET
        import tempfile
        import os
        from pathlib import Path

        model_path_obj = Path(model_path)
        model_dir = model_path_obj.parent

        # Check if this is scene.xml with includes - if so, use lite6.xml directly
        base_xml_path = model_path

        if model_path_obj.name == 'scene.xml':
            # Use lite6.xml which is self-contained
            lite6_path = model_dir / 'lite6.xml'
            if lite6_path.exists():
                base_xml_path = str(lite6_path)
            else:
                print(f"Warning: lite6.xml not found at {lite6_path}, using scene.xml")

        # Parse the base XML file
        tree = ET.parse(base_xml_path)
        root = tree.getroot()

        # Find or create worldbody element
        worldbody = root.find('worldbody')
        if worldbody is None:
            worldbody = ET.SubElement(root, 'worldbody')

        # Check if floor already exists in the worldbody
        has_floor = any(geom.get('name') == 'floor' for geom in worldbody.findall('geom'))

        # Add scene elements (floor, lights, textures) if not present
        if not has_floor:
            # Add floor
            floor = ET.SubElement(worldbody, 'geom')
            floor.set('name', 'floor')
            floor.set('size', '0 0 0.05')
            floor.set('type', 'plane')
            floor.set('material', 'groundplane')

            # Add light
            light = ET.SubElement(worldbody, 'light')
            light.set('pos', '0 0 1.5')
            light.set('dir', '0 0 -1')
            light.set('directional', 'true')

            # Add scene materials to assets
            asset = root.find('asset')
            if asset is None:
                asset = ET.SubElement(root, 'asset')

            # Add groundplane texture
            texture = ET.SubElement(asset, 'texture')
            texture.set('type', '2d')
            texture.set('name', 'groundplane')
            texture.set('builtin', 'checker')
            texture.set('mark', 'edge')
            texture.set('rgb1', '0.2 0.3 0.4')
            texture.set('rgb2', '0.1 0.2 0.3')
            texture.set('markrgb', '0.8 0.8 0.8')
            texture.set('width', '300')
            texture.set('height', '300')

            # Add groundplane material
            material = ET.SubElement(asset, 'material')
            material.set('name', 'groundplane')
            material.set('texture', 'groundplane')
            material.set('texuniform', 'true')
            material.set('texrepeat', '5 5')
            material.set('reflectance', '0.2')

        # Add cube assets (materials and textures)
        asset = root.find('asset')
        if asset is None:
            asset = ET.SubElement(root, 'asset')

        # Define cube colors
        cube_colors = [
            ('cube_red', '0.8 0.2 0.2 1'),
            ('cube_green', '0.2 0.8 0.2 1'),
            ('cube_blue', '0.2 0.2 0.8 1'),
            ('cube_yellow', '0.8 0.8 0.2 1'),
            ('cube_purple', '0.6 0.2 0.8 1'),
            ('cube_orange', '0.9 0.5 0.1 1'),
        ]

        for i, (mat_name, rgba) in enumerate(cube_colors[:num_cubes]):
            material = ET.SubElement(asset, 'material')
            material.set('name', mat_name)
            material.set('rgba', rgba)

        # Add cubes to worldbody with random positions
        cube_size = 0.03  # 3cm cube
        for i in range(num_cubes):
            # Random position within radius (circular distribution)
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.1, placement_radius)  # Avoid center
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = cube_size  # Place at cube half-height above ground

            # Create body for cube
            body = ET.SubElement(worldbody, 'body')
            body.set('name', f'cube_{i}')
            body.set('pos', f'{x:.4f} {y:.4f} {z:.4f}')

            # Add freejoint for physics simulation
            freejoint = ET.SubElement(body, 'freejoint')
            freejoint.set('name', f'cube_{i}_joint')

            # Add cube geometry
            geom = ET.SubElement(body, 'geom')
            geom.set('name', f'cube_{i}_geom')
            geom.set('type', 'box')
            geom.set('size', f'{cube_size} {cube_size} {cube_size}')
            geom.set('material', cube_colors[i % len(cube_colors)][0])
            geom.set('mass', '0.05')  # 50g cube
            geom.set('friction', '1.0 0.005 0.0001')

        # Create temp directory in same location as model to preserve relative paths
        temp_dir = model_dir
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.xml',
            delete=False,
            dir=temp_dir  # Critical: keep temp file in same dir for mesh paths
        ) as f:
            tree.write(f, encoding='unicode')
            temp_path = f.name

        try:
            # Load model from temporary file
            model = mujoco.MjModel.from_xml_path(temp_path)
            print(f"✓ Added {num_cubes} cubes to scene within {placement_radius}m radius")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return model

    def _init_offscreen_renderer(self) -> None:
        """Initialize offscreen renderer for notebook/headless use."""
        try:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            print("Offscreen renderer initialized (notebook-friendly mode)")
        except Exception as e:
            print(f"Could not initialize offscreen renderer: {e}")
            self._renderer = None

    def reset(self, seed: Optional[int] = None, randomize_cubes: bool = True) -> Observation:
        if seed is not None:
            # MuJoCo doesn't use a global RNG for physics; seed if you randomize initial states
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize cube positions if requested
        if randomize_cubes and self.num_cubes > 0:
            self._randomize_cube_positions()

        # Only try to launch viewer if explicitly enabled and not already created
        if self.viewer_enabled and self._viewer is None:
            try:
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

    def plan_and_execute_cartesian_path(
        self,
        goal_pos: np.ndarray,
        goal_quat: Optional[np.ndarray] = None,
        planner_type: str = "rrt",
        joint_limits: Optional[Dict[int, Tuple[float, float]]] = None,
        collision_fn: Optional[callable] = None,
        execute: bool = True,
        num_cartesian_waypoints: int = 20
    ) -> Optional[List[np.ndarray]]:
        """Plan a path to Cartesian goal and optionally execute it.

        This method:
        1. Uses IK to convert the Cartesian goal to joint space
        2. Plans a collision-free path in joint space
        3. Optionally executes the path

        Args:
            goal_pos: Goal position [x, y, z] in meters
            goal_quat: Optional goal orientation [w, x, y, z]. If None, position-only IK
            planner_type: Type of planner ("rrt" or "linear")
            joint_limits: Joint limits as {joint_idx: (min, max)}
            collision_fn: Optional collision checking function
            execute: If True, execute the path immediately
            num_cartesian_waypoints: Number of waypoints for Cartesian interpolation

        Returns:
            List of joint waypoints in the planned path, or None if planning failed
        """
        # Get current joint configuration
        start = self.data.qpos[:6].copy()

        # Solve IK for goal
        print(f"  Solving IK for goal position: {goal_pos}")
        goal_joints = self._ik_solver.solve(goal_pos, goal_quat, q_init=start)

        if goal_joints is None:
            print("  ✗ IK failed: Could not find joint solution for target pose")
            return None

        print(f"  ✓ IK solution found: {goal_joints}")

        # Plan path in joint space
        constraints = {
            "joint_limits": joint_limits,
            "collision_fn": collision_fn
        }

        path = plan_joint_path(start, goal_joints[:6], self.model, constraints, planner_type)

        if path is not None:
            self._current_path = path
            self._path_index = 0

            if execute:
                self.execute_path(path)

        return path

    def plan_and_execute_cartesian_straight_line(
        self,
        goal_pos: np.ndarray,
        goal_quat: Optional[np.ndarray] = None,
        num_waypoints: int = 50,
        collision_fn: Optional[callable] = None,
        execute: bool = True
    ) -> Optional[List[np.ndarray]]:
        """Plan a straight-line path in Cartesian space to the goal.

        This method interpolates in Cartesian space (not joint space) and uses IK
        to solve for joint configurations at each waypoint. This ensures the
        end-effector follows a straight line.

        Args:
            goal_pos: Goal position [x, y, z]
            goal_quat: Optional goal orientation [w, x, y, z]
            num_waypoints: Number of waypoints to interpolate
            collision_fn: Optional collision checking function
            execute: If True, execute the path immediately

        Returns:
            List of joint waypoints, or None if IK fails for any waypoint
        """
        # Get current end-effector pose
        current_pos, current_quat = compute_forward_kinematics(
            self.model, self.data, self.data.qpos[:6], self.ee_site_name
        )

        print(f"  Current EE position: {current_pos}")
        print(f"  Goal EE position: {goal_pos}")

        # Use current orientation if goal orientation not specified
        if goal_quat is None:
            goal_quat = current_quat

        # Interpolate in Cartesian space
        joint_path = []
        current_q = self.data.qpos[:6].copy()

        for i, t in enumerate(np.linspace(0, 1, num_waypoints)):
            # Interpolate position
            waypoint_pos = (1 - t) * current_pos + t * goal_pos

            # Interpolate orientation (SLERP would be better, but linear works for small angles)
            waypoint_quat = (1 - t) * current_quat + t * goal_quat
            waypoint_quat = waypoint_quat / np.linalg.norm(waypoint_quat)  # Normalize

            # Solve IK for this waypoint
            q = self._ik_solver.solve(waypoint_pos, waypoint_quat, q_init=current_q)

            if q is None:
                print(f"  ✗ IK failed at waypoint {i}/{num_waypoints}")
                return None

            # Check collision if function provided
            if collision_fn is not None and collision_fn(q):
                print(f"  ✗ Collision detected at waypoint {i}/{num_waypoints}")
                return None

            joint_path.append(q[:6])
            current_q = q[:6]  # Use this as seed for next IK solve

        print(f"  ✓ Cartesian straight-line path generated with {len(joint_path)} waypoints")

        if execute:
            self.execute_path(joint_path)

        return joint_path

    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position [x, y, z]."""
        pos, _ = compute_forward_kinematics(
            self.model, self.data, self.data.qpos[:6], self.ee_site_name
        )
        return pos

    def get_ee_pose(self) -> np.ndarray:
        """Get current end-effector pose [x, y, z, qw, qx, qy, qz]."""
        pos, quat = compute_forward_kinematics(
            self.model, self.data, self.data.qpos[:6], self.ee_site_name
        )
        return np.concatenate([pos, quat])

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Solve inverse kinematics for a target pose.

        Args:
            target_pos: Target position [x, y, z]
            target_quat: Optional target quaternion [w, x, y, z]

        Returns:
            Joint configuration or None if IK failed
        """
        return self._ik_solver.solve(target_pos, target_quat, q_init=self.data.qpos[:6])

    def _randomize_cube_positions(self) -> None:
        """Randomize positions of all cubes within the placement radius."""
        cube_size = 0.03
        for i in range(self.num_cubes):
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'cube_{i}')

                # Random position within radius
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0.1, self.cube_placement_radius)
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = cube_size  # Start at cube half-height

                # Find the qpos index for this body's freejoint
                # Freejoint has 7 DOF: 3 position + 4 quaternion
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'cube_{i}_joint')
                qpos_adr = self.model.jnt_qposadr[joint_id]

                # Set position
                self.data.qpos[qpos_adr:qpos_adr+3] = [x, y, z]
                # Set quaternion to identity [w, x, y, z]
                self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
                # Zero velocity
                qvel_adr = self.model.jnt_dofadr[joint_id]
                self.data.qvel[qvel_adr:qvel_adr+6] = 0

            except Exception as e:
                print(f"Warning: Could not randomize cube {i}: {e}")

        # Update physics
        mujoco.mj_forward(self.model, self.data)

    def get_cube_positions(self) -> List[np.ndarray]:
        """Get positions of all cubes in the scene.

        Returns:
            List of 3D positions [x, y, z] for each cube
        """
        positions = []
        for i in range(self.num_cubes):
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'cube_{i}')
                pos = self.data.xpos[body_id].copy()
                positions.append(pos)
            except Exception as e:
                print(f"Warning: Could not get position for cube {i}: {e}")
        return positions

    def get_cube_pose(self, cube_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get pose of a specific cube.

        Args:
            cube_idx: Index of the cube (0 to num_cubes-1)

        Returns:
            Tuple of (position [x,y,z], quaternion [w,x,y,z])
        """
        if cube_idx >= self.num_cubes:
            raise ValueError(f"Cube index {cube_idx} out of range (0-{self.num_cubes-1})")

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'cube_{cube_idx}')
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()  # [w, x, y, z] in MuJoCo

        return pos, quat
