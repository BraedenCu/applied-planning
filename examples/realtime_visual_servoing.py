#!/usr/bin/env python3
"""
Real-time visual servoing demo showing camera feedback during motion.

This demonstrates three approaches:
1. Static planning - plan once, execute blindly
2. Reactive replanning - replan when target moves
3. Visual servoing - continuous feedback control
4. Hybrid - plan for coarse motion, servo for precision

Usage:
    python examples/realtime_visual_servoing.py --mode servoing
    python examples/realtime_visual_servoing.py --mode hybrid
"""

import argparse
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applied_planning.sim.adapters.mujoco_backend import MujocoLite6Adapter
from applied_planning.control.visual_servoing import (
    VisualServoController,
    HybridController,
    PositionBasedVisualServo
)


class MockCamera:
    """Mock camera for simulation (replaces real RealSense)."""

    def __init__(self, sim_adapter):
        self.sim = sim_adapter
        # Simulate moving target
        self.target_pos_world = np.array([0.3, 0.1, 0.3])
        self.target_velocity = np.array([0.01, 0.005, 0.0])  # Moves slowly

    def get_frames(self):
        """Simulate getting frames."""
        # Update target position (simulate moving object)
        self.target_pos_world += self.target_velocity * 0.1

        # Keep target in workspace
        if abs(self.target_pos_world[0]) > 0.4:
            self.target_velocity[0] *= -1
        if abs(self.target_pos_world[1]) > 0.2:
            self.target_velocity[1] *= -1

        return MockDepthFrame(self), MockColorFrame()

    def deproject_pixel_to_point(self, pixel, depth_frame):
        """Get 3D point in camera frame."""
        # Get current EE pose
        ee_pos = self.sim.get_ee_position()

        # Vector from camera to target (in world frame)
        vector_world = self.target_pos_world - ee_pos

        # For simplicity, assume camera frame = world frame (no rotation)
        # In real system, you'd transform based on camera orientation
        return vector_world


class MockDepthFrame:
    def __init__(self, camera):
        self.camera = camera

    def get_distance(self, u, v):
        """Get depth at pixel."""
        return np.linalg.norm(self.camera.target_pos_world - self.camera.sim.get_ee_position())


class MockColorFrame:
    def get_data(self):
        """Get color image data."""
        return np.zeros((480, 640, 3), dtype=np.uint8)


class MockHandEyeCalibration:
    """Mock hand-eye calibration."""

    def __init__(self):
        # Identity transform (camera at EE with no offset)
        self.T_ee_camera = np.eye(4)

    def transform_point_to_base(self, point_camera, ee_pose):
        """Transform point from camera to base frame."""
        # For this mock, camera frame = base frame
        ee_pos = ee_pose[:3]
        return ee_pos + point_camera


def mock_detect_object(color_image, depth_frame=None):
    """Mock object detection - always detects at center."""
    return (320, 240), True  # Center of 640x480 image


def demo_static_planning(sim: MujocoLite6Adapter, camera: MockCamera):
    """Demo 1: Static planning - plan once, execute blindly."""
    print("\n" + "="*70)
    print("Demo 1: Static Planning (No Feedback)")
    print("="*70)

    # Detect target once
    depth_frame, color_frame = camera.get_frames()
    pixel, detected = mock_detect_object(None, depth_frame)
    target_pos_camera = camera.deproject_pixel_to_point(pixel, depth_frame)
    ee_pose = sim.get_ee_pose()
    calibration = MockHandEyeCalibration()
    target_pos_base = calibration.transform_point_to_base(target_pos_camera, ee_pose)

    print(f"Target detected at: {target_pos_base}")

    # Plan path
    print("Planning path...")
    path = sim.plan_and_execute_cartesian_path(
        goal_pos=target_pos_base,
        planner_type="linear",
        execute=True
    )

    # Check final error (target may have moved)
    final_ee_pos = sim.get_ee_position()
    current_target_pos = camera.target_pos_world
    error = np.linalg.norm(final_ee_pos - current_target_pos)

    print(f"\nResult:")
    print(f"  Final EE position: {final_ee_pos}")
    print(f"  Current target position: {current_target_pos}")
    print(f"  Position error: {error*1000:.1f} mm")
    print(f"  {'✓ Success' if error < 0.02 else '✗ Missed target (it moved!)'}")


def demo_reactive_replanning(sim: MujocoLite6Adapter, camera: MockCamera):
    """Demo 2: Replan periodically based on camera feedback."""
    print("\n" + "="*70)
    print("Demo 2: Reactive Replanning")
    print("="*70)

    calibration = MockHandEyeCalibration()
    replan_interval = 0.5  # seconds
    last_replan_time = time.time()
    max_duration = 10.0
    start_time = time.time()

    print(f"Replanning every {replan_interval}s based on camera feedback\n")

    iteration = 0
    while (time.time() - start_time) < max_duration:
        iteration += 1

        # Check if we should replan
        if (time.time() - last_replan_time) > replan_interval:
            # Get current target position from camera
            depth_frame, color_frame = camera.get_frames()
            pixel, detected = mock_detect_object(None, depth_frame)
            target_pos_camera = camera.deproject_pixel_to_point(pixel, depth_frame)
            ee_pose = sim.get_ee_pose()
            target_pos_base = calibration.transform_point_to_base(target_pos_camera, ee_pose)

            distance = np.linalg.norm(target_pos_base - ee_pose[:3])
            print(f"[{iteration}] Replanning to target at {target_pos_base} (dist: {distance*100:.1f}cm)")

            if distance < 0.02:  # Within 2cm
                print(f"✓ Target reached!")
                break

            # Replan
            path = sim.plan_and_execute_cartesian_path(
                goal_pos=target_pos_base,
                planner_type="linear",
                execute=False
            )

            if path and len(path) > 0:
                # Execute a few steps
                for i in range(min(10, len(path))):
                    sim.set_state({"qpos": path[i], "qvel": np.zeros(6)})
                    time.sleep(0.01)

            last_replan_time = time.time()

    # Final error check
    final_ee_pos = sim.get_ee_position()
    current_target_pos = camera.target_pos_world
    error = np.linalg.norm(final_ee_pos - current_target_pos)
    print(f"\nFinal error: {error*1000:.1f} mm")


def demo_visual_servoing(sim: MujocoLite6Adapter, camera: MockCamera):
    """Demo 3: Pure visual servoing with continuous feedback."""
    print("\n" + "="*70)
    print("Demo 3: Visual Servoing (Continuous Feedback)")
    print("="*70)

    calibration = MockHandEyeCalibration()
    controller = VisualServoController(
        sim, camera, calibration, mock_detect_object,
        control_rate=20.0,  # 20 Hz control
        replan_threshold=0.05
    )

    success = controller.servo_to_target(timeout=15.0, use_replanning=False)

    if success:
        print("\n✓ Visual servoing successful!")
    else:
        print("\n✗ Visual servoing failed")

    # Final accuracy
    final_ee_pos = sim.get_ee_position()
    current_target_pos = camera.target_pos_world
    error = np.linalg.norm(final_ee_pos - current_target_pos)
    print(f"Final error: {error*1000:.1f} mm")


def demo_hybrid(sim: MujocoLite6Adapter, camera: MockCamera):
    """Demo 4: Hybrid planning + servoing."""
    print("\n" + "="*70)
    print("Demo 4: Hybrid (Planning + Servoing)")
    print("="*70)

    calibration = MockHandEyeCalibration()
    controller = HybridController(
        sim, camera, calibration, mock_detect_object,
        switch_distance=0.15  # Switch to servoing within 15cm
    )

    success = controller.approach_target()

    if success:
        print("\n✓ Hybrid control successful!")
    else:
        print("\n✗ Hybrid control failed")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time visual servoing demo"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["static", "reactive", "servoing", "hybrid", "all"],
        default="hybrid",
        help="Control mode to demonstrate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="src/applied_planning/sim/assets/ufactory_lite6/lite6.xml",
        help="Path to MuJoCo XML model"
    )
    args = parser.parse_args()

    # Resolve model path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    print("="*70)
    print("Real-Time Visual Servoing Demo")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Mode:  {args.mode.upper()}")
    print("="*70)

    # Initialize simulator
    print("\nInitializing simulator...")
    sim = MujocoLite6Adapter(
        str(model_path),
        viewer=False,  # Headless for demo
        render_mode="none"
    )
    sim.reset()
    print("✓ Simulator initialized")

    # Mock camera
    camera = MockCamera(sim)

    # Run demo based on mode
    if args.mode == "static":
        demo_static_planning(sim, camera)
    elif args.mode == "reactive":
        demo_reactive_replanning(sim, camera)
    elif args.mode == "servoing":
        demo_visual_servoing(sim, camera)
    elif args.mode == "hybrid":
        demo_hybrid(sim, camera)
    elif args.mode == "all":
        # Reset between demos
        demo_static_planning(sim, camera)
        sim.reset()
        time.sleep(1)

        demo_reactive_replanning(sim, camera)
        sim.reset()
        time.sleep(1)

        demo_visual_servoing(sim, camera)
        sim.reset()
        time.sleep(1)

        demo_hybrid(sim, camera)

    sim.close()
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()
