"""Hardware adapter for ufactory Lite6 robot using xArm SDK."""

from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import numpy as np
import time
from xarm.wrapper import XArmAPI

from ..sim.adapters.base import Action, Observation, SimulationAdapter


class XArmLite6Adapter(SimulationAdapter):
    """Hardware adapter for controlling real ufactory Lite6 robot.

    Provides same interface as MujocoLite6Adapter for seamless sim-to-real transfer.
    """

    def __init__(
        self,
        robot_ip: str = "192.168.1.161",
        control_dt: float = 0.01,
        max_velocity: float = 50.0,  # deg/s
        max_acceleration: float = 500.0,  # deg/s^2
    ):
        """Initialize connection to real Lite6 robot.

        Args:
            robot_ip: IP address of the robot
            control_dt: Control timestep in seconds
            max_velocity: Maximum joint velocity in degrees/second
            max_acceleration: Maximum joint acceleration in degrees/second^2
        """
        self.robot_ip = robot_ip
        self.control_dt = control_dt
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

        # Connect to robot
        print(f"Connecting to Lite6 at {robot_ip}...")
        self.arm = XArmAPI(robot_ip)

        # Enable motion and set mode
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Position mode (0=position, 1=servo, 2=joint teaching)
        self.arm.set_state(state=0)  # Ready state

        # Set velocity and acceleration limits
        # Note: These are set per-move, not globally in newer SDK versions
        self.arm.set_joint_maxacc(max_acceleration)

        # Initialize velocity control target
        self._target_qpos = None

        print(f"✓ Connected to Lite6")
        print(f"  Version: {self.arm.version}")
        print(f"  State: {self.arm.state}")

    def reset(self, seed: Optional[int] = None, **kwargs) -> Observation:
        """Reset robot to home position.

        Args:
            seed: Random seed (unused for real hardware)

        Returns:
            Current observation
        """
        # Ensure robot is in correct state
        self.arm.set_mode(0)  # Position mode
        self.arm.set_state(state=0)  # Ready state

        # Move to home position [0, 0, 90, 0, 90, 0] degrees
        home_deg = [0, 0, 90, 0, 90, 0]
        print("Moving to home position...")

        # Use non-blocking move with timeout
        code = self.arm.set_servo_angle(
            angle=home_deg,
            speed=50,  # Moderate speed
            wait=False,  # Non-blocking
            is_radian=False
        )

        if code != 0:
            print(f"Warning: Move command returned code {code}")

        # Wait for motion to complete with timeout
        timeout = 10.0  # seconds
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if robot is still moving
            if not self.arm.get_is_moving():
                break
            time.sleep(0.1)

        if time.time() - start_time >= timeout:
            print("Warning: Timeout waiting for home position")
        else:
            print("✓ Home position reached")

        # Initialize velocity control target (only first 6 joints, ignore gripper)
        angles = self.arm.get_servo_angle()[1]  # Returns (code, angles)
        self._target_qpos = np.deg2rad(angles[:6])  # Only use first 6 joints

        return self.get_state()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute one control step.

        Args:
            action: Dictionary with 'qvel' key containing joint velocities (rad/s)

        Returns:
            Tuple of (observation, reward, terminated, info)
        """
        qvel_cmd = action.get("qvel")

        if qvel_cmd is not None:
            qvel_cmd = np.asarray(qvel_cmd)

            # Initialize target if not set (only first 6 joints)
            if self._target_qpos is None:
                angles = self.arm.get_servo_angle()[1]
                self._target_qpos = np.deg2rad(angles[:6])

            # Integrate velocity to position
            self._target_qpos += qvel_cmd * self.control_dt

            # Convert to degrees and send command
            target_deg = np.rad2deg(self._target_qpos).tolist()

            # Use set_servo_angle with speed and acceleration limits
            # wait=False for non-blocking motion
            self.arm.set_servo_angle(
                angle=target_deg,
                speed=self.max_velocity,
                mvacc=self.max_acceleration,
                wait=False,
                is_radian=False
            )

        # Small delay to maintain control rate
        time.sleep(self.control_dt)

        obs = self.get_state()
        reward = 0.0
        terminated = False
        info: Dict[str, Any] = {}

        return obs, reward, terminated, info

    def get_state(self) -> Dict[str, Any]:
        """Get current robot state.

        Returns:
            Dictionary with 'qpos' and 'qvel' keys
        """
        # Get joint positions (only first 6 joints, ignore gripper)
        code, angles = self.arm.get_servo_angle()
        qpos = np.deg2rad(angles[:6]) if code == 0 else np.zeros(6)

        # Get joint velocities (if available)
        # Note: xArm SDK may not provide real-time velocities
        qvel = np.zeros(6)  # Placeholder

        return {"qpos": qpos, "qvel": qvel}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set robot state (position).

        Args:
            state: Dictionary with 'qpos' key containing joint positions (radians)
        """
        qpos = state.get("qpos")
        if qpos is not None:
            qpos_array = np.asarray(qpos)
            target_deg = np.rad2deg(qpos_array).tolist()

            # Move to position and wait
            self.arm.set_servo_angle(angle=target_deg, wait=True)

            # Update velocity control target
            self._target_qpos = qpos_array[:6].copy()

    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position [x, y, z] in meters.

        Returns:
            3D position vector
        """
        code, pos = self.arm.get_position()
        if code == 0:
            # xArm returns position in mm, convert to meters
            return np.array(pos[:3]) / 1000.0
        return np.zeros(3)

    def get_ee_pose(self) -> np.ndarray:
        """Get current end-effector pose [x, y, z, roll, pitch, yaw].

        Returns:
            6D pose vector (position in meters, orientation in radians)
        """
        code, pos = self.arm.get_position()
        if code == 0:
            # Convert position from mm to meters, orientation already in degrees
            pose = np.array(pos)
            pose[:3] = pose[:3] / 1000.0  # Convert to meters
            pose[3:] = np.deg2rad(pose[3:])  # Convert to radians
            return pose
        return np.zeros(6)

    def render(self, mode: str = "human", **kwargs):
        """Render - not applicable for real hardware."""
        if mode == "human":
            print(f"Real robot at {self.robot_ip}")
            state = self.get_state()
            print(f"Joint positions (deg): {np.rad2deg(state['qpos'])}")
            print(f"EE position (m): {self.get_ee_position()}")
        return None

    def close(self) -> None:
        """Disconnect from robot."""
        if hasattr(self, 'arm'):
            try:
                print("Disconnecting from robot...")
                self.arm.disconnect()
                print("✓ Disconnected")
            except Exception as e:
                print(f"Warning: Disconnect error (may already be disconnected): {e}")

    def emergency_stop(self) -> None:
        """Emergency stop the robot."""
        print("⚠ EMERGENCY STOP")
        self.arm.emergency_stop()

    def clear_error(self) -> None:
        """Clear robot errors and re-enable motion."""
        print("Clearing errors...")
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_state(state=0)
        print("✓ Errors cleared")

    def reconnect(self) -> bool:
        """Attempt to reconnect to robot.

        Returns:
            True if reconnection successful, False otherwise
        """
        try:
            print("Attempting to reconnect...")

            # Close existing connection if any
            if hasattr(self, 'arm'):
                try:
                    self.arm.disconnect()
                except:
                    pass

            # Wait a moment
            time.sleep(1)

            # Reconnect
            self.arm = XArmAPI(self.robot_ip)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(state=0)
            self.arm.set_joint_maxacc(self.max_acceleration)

            # Reinitialize target position
            angles = self.arm.get_servo_angle()[1]
            self._target_qpos = np.deg2rad(angles[:6])

            print("✓ Reconnected successfully")
            return True

        except Exception as e:
            print(f"✗ Reconnection failed: {e}")
            return False

    def get_robot_info(self) -> Dict[str, Any]:
        """Get robot information and status.

        Returns:
            Dictionary with robot info
        """
        return {
            "version": self.arm.version,
            "state": self.arm.state,
            "error_code": self.arm.error_code,
            "warn_code": self.arm.warn_code,
            "position": self.arm.position,
            "angles": self.arm.angles,
        }
