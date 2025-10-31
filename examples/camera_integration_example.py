#!/usr/bin/env python3
"""
Example showing how to integrate RealSense camera for vision-based planning.

This is a reference implementation showing the complete workflow from
camera capture to path planning and execution.

Requirements:
    - pyrealsense2 (pip install pyrealsense2)
    - Camera mounted on robot end-effector with known transform
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applied_planning.sim.adapters.mujoco_backend import MujocoLite6Adapter


class HandEyeCalibration:
    """Manages transformation between camera frame and robot base frame."""

    def __init__(self, T_ee_camera: np.ndarray):
        """Initialize hand-eye calibration.

        Args:
            T_ee_camera: 4x4 transformation matrix from end-effector to camera frame
        """
        self.T_ee_camera = T_ee_camera

    def transform_point_to_base(
        self,
        point_camera: np.ndarray,
        ee_pose: np.ndarray
    ) -> np.ndarray:
        """Transform point from camera frame to robot base frame.

        Args:
            point_camera: 3D point in camera frame [x, y, z]
            ee_pose: Current end-effector pose [x, y, z, qw, qx, qy, qz]

        Returns:
            3D point in robot base frame [x, y, z]
        """
        # Construct transformation matrix from base to end-effector
        T_base_ee = self._pose_to_matrix(ee_pose)

        # Chain transformations: base -> ee -> camera
        T_base_camera = T_base_ee @ self.T_ee_camera

        # Transform point
        point_camera_homo = np.append(point_camera, 1.0)
        point_base_homo = T_base_camera @ point_camera_homo

        return point_base_homo[:3]

    @staticmethod
    def _pose_to_matrix(pose: np.ndarray) -> np.ndarray:
        """Convert pose vector to 4x4 transformation matrix.

        Args:
            pose: [x, y, z, qw, qx, qy, qz]

        Returns:
            4x4 transformation matrix
        """
        import mujoco
        pos = pose[:3]
        quat = pose[3:]  # [qw, qx, qy, qz]

        # Convert quaternion to rotation matrix
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)

        # Build 4x4 matrix
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos

        return T


class RealSenseCameraInterface:
    """Interface for RealSense camera (requires pyrealsense2)."""

    def __init__(self):
        """Initialize RealSense camera."""
        try:
            import pyrealsense2 as rs
            self.rs = rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 not installed. Install with: pip install pyrealsense2"
            )

        # Configure streams
        self.pipeline = self.rs.pipeline()
        config = self.rs.config()
        config.enable_stream(self.rs.stream.depth, 640, 480, self.rs.format.z16, 30)
        config.enable_stream(self.rs.stream.color, 640, 480, self.rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(config)

        # Get intrinsics
        depth_stream = self.profile.get_stream(self.rs.stream.depth)
        self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        """Capture depth and color frames."""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return depth_frame, color_frame

    def deproject_pixel_to_point(
        self,
        pixel: Tuple[int, int],
        depth_frame
    ) -> np.ndarray:
        """Convert 2D pixel + depth to 3D point in camera frame.

        Args:
            pixel: (u, v) pixel coordinates
            depth_frame: RealSense depth frame

        Returns:
            3D point [x, y, z] in camera frame (meters)
        """
        u, v = pixel
        depth = depth_frame.get_distance(u, v)
        point = self.rs.rs2_deproject_pixel_to_point(
            self.intrinsics,
            [u, v],
            depth
        )
        return np.array(point)

    def close(self):
        """Stop the camera pipeline."""
        self.pipeline.stop()


def detect_object_in_image(color_image: np.ndarray) -> Optional[Tuple[int, int]]:
    """Detect object in color image and return pixel coordinates.

    This is a placeholder - implement your own object detection here
    using OpenCV, deep learning, etc.

    Args:
        color_image: BGR color image from camera

    Returns:
        (u, v) pixel coordinates of detected object, or None if not found
    """
    # Example: Use color-based detection, blob detection, or neural network
    # For now, return center of image as placeholder
    height, width = color_image.shape[:2]
    return (width // 2, height // 2)


def main():
    """Main example workflow."""
    print("="*70)
    print("Camera-Based Pick-and-Place Demo")
    print("="*70)

    # 1. Initialize robot simulator
    print("\n[1/5] Initializing robot...")
    model_path = "src/applied_planning/sim/assets/ufactory_lite6/lite6.xml"
    sim = MujocoLite6Adapter(
        model_path,
        viewer=True,
        render_mode="passive"
        # ee_site_name defaults to "attachment_site" for Lite6
    )
    sim.reset()
    print("✓ Robot initialized")

    # 2. Initialize camera (in real setup)
    print("\n[2/5] Camera initialization...")
    print("  Note: This example requires actual RealSense camera")
    print("  Skipping camera initialization for demo purposes")

    # Example hand-eye calibration matrix (eye-in-hand configuration)
    # This should be obtained through calibration procedure
    T_ee_camera = np.array([
        [0, -1, 0, 0.05],   # Camera 5cm offset from EE
        [0, 0, -1, 0.0],
        [1, 0, 0, 0.0],
        [0, 0, 0, 1.0]
    ])
    calibration = HandEyeCalibration(T_ee_camera)
    print("✓ Hand-eye calibration loaded")

    # 3. Move to scanning position
    print("\n[3/5] Moving to scanning position...")
    scan_pos = np.array([0.3, 0.0, 0.4])  # 30cm forward, 40cm up
    scan_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Looking down

    path = sim.plan_and_execute_cartesian_path(
        goal_pos=scan_pos,
        goal_quat=scan_quat,
        planner_type="rrt",
        execute=True
    )
    if path is None:
        print("✗ Failed to reach scanning position")
        return
    print("✓ At scanning position")

    # 4. Capture and process image
    print("\n[4/5] Capturing image and detecting object...")

    # In real implementation with camera:
    # camera = RealSenseCameraInterface()
    # depth_frame, color_frame = camera.get_frames()
    # color_image = np.asanyarray(color_frame.get_data())
    # object_pixel = detect_object_in_image(color_image)
    #
    # if object_pixel is None:
    #     print("✗ No object detected")
    #     return
    #
    # # Get 3D point in camera frame
    # object_point_camera = camera.deproject_pixel_to_point(object_pixel, depth_frame)

    # For demo: simulate detected object 15cm in front of camera
    object_point_camera = np.array([0.0, 0.0, 0.15])  # 15cm along camera Z-axis

    print(f"  Detected object at camera coordinates: {object_point_camera}")

    # Transform to robot base frame
    current_ee_pose = sim.get_ee_pose()
    object_point_base = calibration.transform_point_to_base(
        object_point_camera,
        current_ee_pose
    )
    print(f"  Object in base frame: {object_point_base}")

    # 5. Plan and execute grasp
    print("\n[5/5] Planning path to object...")

    # Approach from above (offset by 5cm for safety)
    approach_pos = object_point_base + np.array([0.0, 0.0, 0.05])

    path = sim.plan_and_execute_cartesian_straight_line(
        goal_pos=approach_pos,
        goal_quat=scan_quat,  # Maintain downward orientation
        num_waypoints=50,
        execute=True
    )

    if path is None:
        print("✗ Failed to reach object")
        return

    print("✓ Reached approach position")
    print("\nNow you would:")
    print("  - Close gripper")
    print("  - Move down to grasp position")
    print("  - Pick up object")
    print("  - Move to place location")

    sim.close()
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
