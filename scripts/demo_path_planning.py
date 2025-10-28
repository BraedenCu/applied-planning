#!/usr/bin/env python3
"""
Demo script for 6DOF robot arm path planning with real-time MuJoCo visualization.

Run this script (NOT in Jupyter) to see the robot motion in the MuJoCo viewer.

Usage:
    python scripts/demo_path_planning.py
    python scripts/demo_path_planning.py --planner linear
    python scripts/demo_path_planning.py --no-viewer  # headless mode
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
    parser = argparse.ArgumentParser(description="Demo 6DOF path planning with MuJoCo")
    parser.add_argument(
        "--model",
        type=str,
        default="src/applied_planning/sim/assets/ufactory_lite6/lite6.xml",
        help="Path to MuJoCo XML model"
    )
    parser.add_argument(
        "--planner",
        type=str,
        choices=["rrt", "linear"],
        default="rrt",
        help="Planner type: rrt or linear"
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless without viewer"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Execution speed factor (1.0 = normal, 0.5 = slower, 2.0 = faster)"
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
        print("  python scripts/demo_path_planning.py")
        return

    print("="*70)
    print("6DOF Robot Arm Path Planning Demo")
    print("="*70)
    print(f"Model:   {model_path}")
    print(f"Planner: {args.planner.upper()}")
    print(f"Viewer:  {'Enabled' if not args.no_viewer else 'Disabled'}")
    print("="*70)

    # Initialize simulator with viewer
    print("\n[1/5] Initializing simulator...")
    sim = MujocoLite6Adapter(
        str(model_path),
        viewer=not args.no_viewer,
        render_mode="passive" if not args.no_viewer else "none",
        control_dt=0.01
    )
    sim.reset()
    print("✓ Simulator initialized")

    # Get initial configuration
    start = sim.data.qpos[:6].copy()
    print(f"\n[2/5] Current joint configuration:")
    print(f"  {start}")

    # Define goal configuration
    goal = np.array([0.5, -0.3, 0.8, 0.0, 1.0, 0.0])
    print(f"\n[3/5] Goal joint configuration:")
    print(f"  {goal}")

    # Get joint limits
    joint_limits = sim.get_joint_limits()
    print(f"\n[4/5] Joint limits: {len(joint_limits)} joints with limits")

    # Plan path
    print(f"\n[5/5] Planning path with {args.planner.upper()}...")
    print("  (This may take a few seconds...)")

    start_time = time.time()
    path = sim.plan_and_execute_path(
        goal=goal,
        planner_type=args.planner,
        joint_limits=joint_limits,
        collision_fn=None,  # Set to sim.check_collision to enable collision checking
        execute=False  # We'll execute manually for better visualization
    )
    plan_time = time.time() - start_time

    if path is None:
        print("\n✗ FAILED: No path found")
        sim.close()
        return

    # Calculate path statistics
    path_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
    print(f"\n✓ SUCCESS!")
    print(f"  Waypoints:   {len(path)}")
    print(f"  Path length: {path_length:.3f} rad")
    print(f"  Plan time:   {plan_time:.3f} sec")

    # Execute path with visualization
    if not args.no_viewer:
        print(f"\n{'='*70}")
        print("Executing path in MuJoCo viewer...")
        print("(Watch the viewer window for the robot motion)")
        print(f"{'='*70}\n")

        # Give user time to focus on viewer
        time.sleep(1.0)

        # Execute the path
        sim.execute_path(path, velocity_control=False, speed_factor=args.speed)

        print("\n✓ Execution complete!")
        print("\nKeeping viewer open. Press Ctrl+C to exit...")

        try:
            # Keep the viewer open
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        print("\n(Running in headless mode - no visualization)")

    sim.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
