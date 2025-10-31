"""Visual servoing controllers for camera-in-hand configurations."""

from __future__ import annotations

from typing import Optional, Callable, Tuple
import numpy as np
import time


class PositionBasedVisualServo:
    """Position-Based Visual Servoing (PBVS) controller.

    For eye-in-hand configuration where camera is mounted on end-effector.
    Uses 3D object pose estimates to generate motion commands.
    """

    def __init__(
        self,
        control_gain: float = 0.5,
        position_tolerance: float = 0.005,  # 5mm
        orientation_tolerance: float = 0.05,  # ~3 degrees
        max_velocity: float = 0.1,  # m/s
        max_angular_velocity: float = 0.5  # rad/s
    ):
        """Initialize PBVS controller.

        Args:
            control_gain: Proportional gain for velocity control (0-1)
            position_tolerance: Distance threshold to consider goal reached (meters)
            orientation_tolerance: Orientation threshold (radians)
            max_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        self.control_gain = control_gain
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity

    def compute_velocity_command(
        self,
        current_ee_pose: np.ndarray,
        target_pose_camera: np.ndarray,
        T_ee_camera: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Compute end-effector velocity command based on visual feedback.

        Args:
            current_ee_pose: Current EE pose [x, y, z, qw, qx, qy, qz] in base frame
            target_pose_camera: Target pose in camera frame [x, y, z, qw, qx, qy, qz]
            T_ee_camera: 4x4 transform from end-effector to camera

        Returns:
            Tuple of (velocity_command, goal_reached)
            velocity_command: [vx, vy, vz, wx, wy, wz] in base frame
            goal_reached: True if within tolerance
        """
        # Transform target from camera frame to end-effector frame
        target_pos_camera = target_pose_camera[:3]
        target_pos_ee = self._transform_point(target_pos_camera, T_ee_camera)

        # Position error in end-effector frame
        pos_error = target_pos_ee

        # Check if goal reached
        if np.linalg.norm(pos_error) < self.position_tolerance:
            return np.zeros(6), True

        # Compute velocity command (proportional control)
        linear_velocity = self.control_gain * pos_error

        # Clip to max velocity
        velocity_norm = np.linalg.norm(linear_velocity)
        if velocity_norm > self.max_velocity:
            linear_velocity = linear_velocity / velocity_norm * self.max_velocity

        # For simplicity, no angular velocity (can be added for orientation control)
        angular_velocity = np.zeros(3)

        velocity_command = np.concatenate([linear_velocity, angular_velocity])
        return velocity_command, False

    @staticmethod
    def _transform_point(point: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Transform 3D point by 4x4 transformation matrix."""
        point_homo = np.append(point, 1.0)
        transformed = T @ point_homo
        return transformed[:3]


class VisualServoController:
    """High-level visual servoing controller with replanning capabilities."""

    def __init__(
        self,
        sim_adapter,
        camera_interface,
        hand_eye_calibration,
        detection_function: Callable,
        control_rate: float = 10.0,  # Hz
        replan_threshold: float = 0.02  # 2cm position change triggers replan
    ):
        """Initialize visual servo controller.

        Args:
            sim_adapter: MujocoLite6Adapter or similar
            camera_interface: Camera interface with get_frames() method
            hand_eye_calibration: HandEyeCalibration instance
            detection_function: Function that takes color_image and depth_frame,
                              returns (pixel_coords, detected) tuple
            control_rate: Control loop frequency (Hz)
            replan_threshold: How much target can move before replanning (meters)
        """
        self.sim = sim_adapter
        self.camera = camera_interface
        self.calibration = hand_eye_calibration
        self.detect_object = detection_function
        self.control_rate = control_rate
        self.replan_threshold = replan_threshold
        self.dt = 1.0 / control_rate

        self.pbvs = PositionBasedVisualServo()
        self.last_target_pos = None

    def servo_to_target(
        self,
        timeout: float = 30.0,
        use_replanning: bool = True
    ) -> bool:
        """Execute visual servoing to reach detected target.

        Args:
            timeout: Maximum time to reach target (seconds)
            use_replanning: If True, replan when target moves significantly

        Returns:
            True if target reached, False if timeout or detection lost
        """
        start_time = time.time()
        iterations = 0

        print("\nStarting visual servoing...")
        print(f"  Control rate: {self.control_rate} Hz")
        print(f"  Replan threshold: {self.replan_threshold*100:.1f} cm")

        while (time.time() - start_time) < timeout:
            iterations += 1

            # 1. Capture image
            depth_frame, color_frame = self.camera.get_frames()
            color_image = np.asanyarray(color_frame.get_data())

            # 2. Detect target
            pixel_coords, detected = self.detect_object(color_image, depth_frame)

            if not detected:
                print(f"  [{iterations}] Target lost - stopping")
                return False

            # 3. Get 3D position in camera frame
            target_pos_camera = self.camera.deproject_pixel_to_point(
                pixel_coords, depth_frame
            )

            # 4. Transform to base frame
            ee_pose = self.sim.get_ee_pose()
            target_pos_base = self.calibration.transform_point_to_base(
                target_pos_camera, ee_pose
            )

            # 5. Check if we need to replan (target moved)
            if use_replanning and self.last_target_pos is not None:
                position_change = np.linalg.norm(target_pos_base - self.last_target_pos)
                if position_change > self.replan_threshold:
                    print(f"  [{iterations}] Target moved {position_change*100:.1f}cm - replanning...")
                    self._replan_to_target(target_pos_base)

            self.last_target_pos = target_pos_base.copy()

            # 6. Compute velocity command
            target_pose_camera = np.concatenate([
                target_pos_camera,
                np.array([1, 0, 0, 0])  # Identity quaternion
            ])

            velocity_cmd, goal_reached = self.pbvs.compute_velocity_command(
                ee_pose,
                target_pose_camera,
                self.calibration.T_ee_camera
            )

            if goal_reached:
                print(f"  [{iterations}] Target reached!")
                return True

            # 7. Convert Cartesian velocity to joint velocity
            joint_velocity = self._cartesian_to_joint_velocity(velocity_cmd)

            # 8. Execute motion
            self.sim.step({"qvel": joint_velocity})

            # 9. Sleep to maintain control rate
            time.sleep(self.dt)

            # 10. Periodic status
            if iterations % int(self.control_rate) == 0:
                distance = np.linalg.norm(target_pos_camera)
                print(f"  [{iterations}] Distance to target: {distance*100:.1f} cm")

        print(f"  Timeout after {timeout}s")
        return False

    def _replan_to_target(self, target_pos_base: np.ndarray):
        """Replan path to updated target position."""
        path = self.sim.plan_and_execute_cartesian_path(
            goal_pos=target_pos_base,
            planner_type="linear",  # Fast replanning
            execute=False
        )
        # Note: In a real system, you'd execute this smoothly
        # For now, we continue with visual servoing

    def _cartesian_to_joint_velocity(self, cart_velocity: np.ndarray) -> np.ndarray:
        """Convert Cartesian velocity to joint velocity using Jacobian.

        Args:
            cart_velocity: [vx, vy, vz, wx, wy, wz] in base frame

        Returns:
            Joint velocities for 6DOF arm
        """
        import mujoco

        # Get Jacobian at current configuration
        jacp = np.zeros((3, self.sim.model.nv))
        jacr = np.zeros((3, self.sim.model.nv))

        try:
            site_id = mujoco.mj_name2id(
                self.sim.model, mujoco.mjtObj.mjOBJ_SITE, self.sim.ee_site_name
            )
            mujoco.mj_jacSite(self.sim.model, self.sim.data, jacp, jacr, site_id)
        except Exception:
            # Fallback to body Jacobian
            body_id = mujoco.mj_name2id(
                self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "link6"
            )
            mujoco.mj_jacBody(self.sim.model, self.sim.data, jacp, jacr, body_id)

        # Combine position and rotation Jacobian
        jac = np.vstack([jacp[:, :6], jacr[:, :6]])  # Only first 6 joints

        # Use pseudoinverse to get joint velocities
        jac_pinv = np.linalg.pinv(jac)
        joint_velocity = jac_pinv @ cart_velocity

        return joint_velocity


class HybridController:
    """Hybrid controller combining path planning with visual servoing.

    Use path planning for large motions, switch to visual servoing for precision.
    """

    def __init__(
        self,
        sim_adapter,
        camera_interface,
        hand_eye_calibration,
        detection_function: Callable,
        switch_distance: float = 0.1  # 10cm - switch to servoing within this distance
    ):
        """Initialize hybrid controller.

        Args:
            sim_adapter: MujocoLite6Adapter or similar
            camera_interface: Camera interface
            hand_eye_calibration: HandEyeCalibration instance
            detection_function: Object detection function
            switch_distance: Distance threshold to switch from planning to servoing
        """
        self.sim = sim_adapter
        self.camera = camera_interface
        self.calibration = hand_eye_calibration
        self.detect_object = detection_function
        self.switch_distance = switch_distance

        self.servo_controller = VisualServoController(
            sim_adapter, camera_interface, hand_eye_calibration, detection_function
        )

    def approach_target(self) -> bool:
        """Approach target using hybrid planning + servoing strategy.

        Returns:
            True if target reached, False otherwise
        """
        print("="*70)
        print("Hybrid Control: Planning + Visual Servoing")
        print("="*70)

        # Phase 1: Detect target from current position
        print("\n[Phase 1] Detecting target...")
        depth_frame, color_frame = self.camera.get_frames()
        color_image = np.asanyarray(color_frame.get_data())
        pixel_coords, detected = self.detect_object(color_image, depth_frame)

        if not detected:
            print("✗ No target detected")
            return False

        # Get target position in base frame
        target_pos_camera = self.camera.deproject_pixel_to_point(
            pixel_coords, depth_frame
        )
        ee_pose = self.sim.get_ee_pose()
        target_pos_base = self.calibration.transform_point_to_base(
            target_pos_camera, ee_pose
        )

        print(f"✓ Target detected at: {target_pos_base}")

        # Phase 2: Plan to approach position (offset from target)
        distance_to_target = np.linalg.norm(target_pos_base - ee_pose[:3])
        print(f"\n[Phase 2] Distance to target: {distance_to_target*100:.1f} cm")

        if distance_to_target > self.switch_distance:
            print(f"  Using path planning (distance > {self.switch_distance*100:.1f}cm)")

            # Approach to within switch_distance
            direction = (target_pos_base - ee_pose[:3]) / distance_to_target
            approach_pos = target_pos_base - direction * self.switch_distance

            print(f"  Planning to approach position: {approach_pos}")
            path = self.sim.plan_and_execute_cartesian_path(
                goal_pos=approach_pos,
                planner_type="rrt",
                execute=True
            )

            if path is None:
                print("✗ Path planning failed")
                return False

            print("✓ Reached approach position")

        # Phase 3: Visual servoing for precise approach
        print(f"\n[Phase 3] Switching to visual servoing...")
        success = self.servo_controller.servo_to_target(
            timeout=15.0,
            use_replanning=True
        )

        if success:
            print("\n✓ Target reached with high precision!")
        else:
            print("\n✗ Visual servoing failed")

        return success
