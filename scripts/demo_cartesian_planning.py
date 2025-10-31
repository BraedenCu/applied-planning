#!/usr/bin/env python3
"""
Demo script for Cartesian-space path planning with camera-based goals.

This demonstrates how to plan paths based on 3D points in Cartesian space,
such as those derived from a RealSense camera attached to the end-effector.

Usage:
    python scripts/demo_cartesian_planning.py
    python scripts/demo_cartesian_planning.py --mode straight  # straight-line in Cartesian space
    python scripts/demo_cartesian_planning.py --no-viewer      # headless mode
"""

import argparse
import numpy as np
import time
from pathlib import Path

# Add src to path if running directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applied_planning.sim.adapters.mujoco_backend import MujocoLite6Adapter


def main():
    parser = argparse.ArgumentParser(
        description="Demo Cartesian-space path planning for camera-based robot control"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="src/applied_planning/sim/assets/ufactory_lite6/lite6.xml",
        help="Path to MuJoCo XML model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rrt", "linear", "straight"],
        default="rrt",
        help="Planning mode: 'rrt' or 'linear' plans in joint space, 'straight' plans straight line in Cartesian space"
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless without viewer"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Execution speed factor (1.0 = normal, 0.5 = slower for better visualization)"
    )
    args = parser.parse_args()

    # Resolve model path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("\nTip: Run this script from the project root directory:")
        print("  cd /Users/braeden/Development/applied-planning")
        print("  python scripts/demo_cartesian_planning.py")
        return

    print("="*70)
    print("Cartesian-Space Robot Path Planning Demo")
    print("Camera-Based Goal Specification")
    print("="*70)
    print(f"Model:   {model_path}")
    print(f"Mode:    {args.mode.upper()}")
    print(f"Viewer:  {'Enabled' if not args.no_viewer else 'Disabled'}")
    print("="*70)

    # Initialize simulator
    print("\n[1/6] Initializing simulator...")
    sim = MujocoLite6Adapter(
        str(model_path),
        viewer=not args.no_viewer,
        render_mode="passive" if not args.no_viewer else "none",
        control_dt=0.01
        # ee_site_name defaults to "attachment_site" for Lite6
    )
    sim.reset()
    print("✓ Simulator initialized")

    # Get current end-effector pose
    print("\n[2/6] Getting current end-effector pose...")
    current_pos = sim.get_ee_position()
    current_pose = sim.get_ee_pose()

    print(f"  Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}] m")
    print(f"  Orientation: [{current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}, {current_pose[6]:.3f}] (quat)")

    # Define Cartesian goal (e.g., from camera detection)
    print("\n[3/6] Defining Cartesian goal...")
    print("  (In real application, this would come from camera/vision system)")

    # Example: Move 10cm forward, 5cm up, and 5cm to the right
    goal_pos = current_pos + np.array([0.10, 0.05, 0.05])
    # Keep same orientation (or you could specify a new one)
    goal_quat = current_pose[3:]  # [qw, qx, qy, qz]

    print(f"  Goal position: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}] m")
    print(f"  Goal orientation: [{goal_quat[0]:.3f}, {goal_quat[1]:.3f}, {goal_quat[2]:.3f}, {goal_quat[3]:.3f}]")

    # Calculate Cartesian distance
    distance = np.linalg.norm(goal_pos - current_pos)
    print(f"  Cartesian distance: {distance*100:.1f} cm")

    # Get joint limits for planning
    print("\n[4/6] Setting up constraints...")
    joint_limits = sim.get_joint_limits()
    print(f"  Joint limits: {len(joint_limits)} joints with limits")

    # Plan path
    print(f"\n[5/6] Planning path using {args.mode.upper()} method...")
    start_time = time.time()

    if args.mode == "straight":
        # Plan straight line in Cartesian space
        print("  Planning straight-line path in Cartesian space...")
        path = sim.plan_and_execute_cartesian_straight_line(
            goal_pos=goal_pos,
            goal_quat=goal_quat,
            num_waypoints=50,
            collision_fn=None,  # Set to sim.check_collision for collision checking
            execute=False
        )
    else:
        # Plan using RRT or linear in joint space after IK
        print(f"  Planning {args.mode.upper()} path in joint space with IK...")
        path = sim.plan_and_execute_cartesian_path(
            goal_pos=goal_pos,
            goal_quat=goal_quat,
            planner_type=args.mode,
            joint_limits=joint_limits,
            collision_fn=None,  # Set to sim.check_collision for collision checking
            execute=False
        )

    plan_time = time.time() - start_time

    if path is None:
        print("\n✗ FAILED: No path found")
        print("  This could be due to:")
        print("  - IK failure (goal pose unreachable)")
        print("  - Joint limits violated")
        print("  - Collision detected")
        sim.close()
        return

    # Calculate path statistics
    joint_path_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
    print(f"\n✓ SUCCESS!")
    print(f"  Waypoints:        {len(path)}")
    print(f"  Joint path length: {joint_path_length:.3f} rad")
    print(f"  Cartesian distance: {distance*100:.1f} cm")
    print(f"  Planning time:    {plan_time:.3f} sec")

    # Execute path
    print(f"\n[6/6] Executing path...")
    if not args.no_viewer:
        print(f"{'='*70}")
        print("Watch the viewer window for robot motion")
        print(f"{'='*70}\n")
        time.sleep(1.0)

    sim.execute_path(path, velocity_control=False, speed_factor=args.speed)

    # Verify final position
    final_pos = sim.get_ee_position()
    position_error = np.linalg.norm(final_pos - goal_pos)

    print(f"\n✓ Execution complete!")
    print(f"  Target position:   [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
    print(f"  Achieved position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"  Position error:    {position_error*1000:.2f} mm")

    if not args.no_viewer:
        print("\nKeeping viewer open. Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")

    sim.close()
    print("\nDemo complete!")


def demo_camera_based_workflow():
    """
    Example workflow showing how to integrate with a RealSense camera
    for pick-and-place based on visual detection.

    This is a conceptual example - actual camera integration would require:
    - pyrealsense2 library
    - Camera calibration
    - Hand-eye calibration (camera frame -> robot base frame)
    """
    print("\n" + "="*70)
    print("Camera-Based Workflow Example (Conceptual)")
    print("="*70)
    print("""
# Typical workflow for camera-based planning:

# 1. Capture image from RealSense camera on end-effector
depth_frame, color_frame = camera.get_frames()

# 2. Detect object in image (e.g., using computer vision)
object_pixel_coords = detect_object(color_frame)  # e.g., [u, v]

# 3. Get 3D point from depth
object_point_camera = camera.deproject_pixel_to_point(
    object_pixel_coords,
    depth_frame
)  # Returns [x, y, z] in camera frame

# 4. Transform to robot base frame using hand-eye calibration
T_base_camera = get_hand_eye_transform()
object_point_base = T_base_camera @ object_point_camera

# 5. Plan path to object location
sim.plan_and_execute_cartesian_path(
    goal_pos=object_point_base,
    goal_quat=desired_grasp_orientation,
    planner_type="rrt"
)

# 6. Execute grasp and pick-and-place
""")


if __name__ == "__main__":
    main()

    # Optionally show conceptual workflow
    import sys
    if "--show-workflow" in sys.argv:
        demo_camera_based_workflow()
