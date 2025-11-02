#!/usr/bin/env python3
"""Test real robot control with simple motions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import time
from applied_planning.hardware import XArmLite6Adapter

print("=" * 60)
print("REAL ROBOT TEST - Lite6 will move!")
print("=" * 60)
print("\nThis script will:")
print("1. Connect to robot")
print("2. Clear any errors")
print("3. Move to home position")
print("4. Execute small test motions")
print("\n⚠️  Ensure workspace is clear!")

input("\nPress ENTER to continue or Ctrl+C to abort...")

try:
    # Connect to robot
    robot = XArmLite6Adapter(robot_ip='192.168.1.161')

    # Clear errors and reset
    robot.clear_error()
    time.sleep(0.5)

    # Get initial state
    print("\nInitial state:")
    info = robot.get_robot_info()
    print(f"  State: {info['state']}")
    print(f"  Error: {info['error_code']}")
    print(f"  Position: {robot.get_ee_position()}")

    # Move to home
    print("\nMoving to home position...")
    robot.reset()
    time.sleep(1)

    print("\nHome position reached")
    print(f"  EE position: {robot.get_ee_position()}")

    # Test velocity control - small motion on joint 1
    print("\nTest: Small rotation of joint 1...")
    print("Moving forward 10 degrees over 2 seconds")

    vel = np.array([np.deg2rad(5), 0, 0, 0, 0, 0])  # 5 deg/s on joint 1

    for i in range(40):  # 2 seconds at 50Hz
        robot.step({'qvel': vel})
        if i % 10 == 0:
            state = robot.get_state()
            print(f"  t={i*0.02:.1f}s: Joint 1 = {np.rad2deg(state['qpos'][0]):.1f}°")

    print("\nMoving back to home...")
    robot.reset()

    print("\n✓ Test complete!")
    print(f"Final position: {robot.get_ee_position()}")

    # Cleanup
    robot.close()
    print("\n✓ Disconnected safely")

except KeyboardInterrupt:
    print("\n\n⚠️  Aborted by user")
    if 'robot' in locals():
        robot.emergency_stop()
        robot.close()
except Exception as e:
    print(f"\n✗ Error: {e}")
    if 'robot' in locals():
        robot.emergency_stop()
        robot.close()
    raise
