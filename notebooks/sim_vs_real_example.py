#!/usr/bin/env python3
"""Example showing simulation and real robot using same control code."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import time

# Toggle between simulation and real robot
USE_REAL_ROBOT = False  # Set to True for real hardware

if USE_REAL_ROBOT:
    # Real hardware
    from applied_planning.hardware import XArmLite6Adapter
    robot = XArmLite6Adapter(robot_ip='192.168.1.161')
    print("Using REAL robot")
else:
    # Simulation
    from applied_planning.sim.adapters import MujocoLite6Adapter
    model_path = str(Path(__file__).parent.parent / 'src/applied_planning/sim/assets/ufactory_lite6/lite6.xml')
    robot = MujocoLite6Adapter(model_path, viewer=False, render_mode='offscreen')
    print("Using SIMULATION")

try:
    # ------ IDENTICAL CODE FOR BOTH SIM AND REAL ------

    # Go to home position
    print("\n1. Moving to home position...")
    robot.reset()
    print(f"   Position: {robot.get_ee_position()}")

    # Test 1: Move joint 1 only
    print("\n2. Testing joint 1 motion (2 seconds)...")
    for i in range(200):  # 2 seconds at 100Hz
        robot.step({'qvel': [0.1, 0, 0, 0, 0, 0]})
        if i % 50 == 0:
            state = robot.get_state()
            print(f"   Joint 1: {np.rad2deg(state['qpos'][0]):.1f}°")

    # Reset between tests
    print("\n3. Returning to home...")
    robot.reset()
    time.sleep(0.5)

    # Test 2: Move joints 2 and 3 together
    print("\n4. Testing joints 2 and 3 motion (2 seconds)...")
    for i in range(200):  # 2 seconds
        robot.step({'qvel': [0, 0.1, 0.1, 0, 0, 0]})
        if i % 50 == 0:
            state = robot.get_state()
            print(f"   Joint 2: {np.rad2deg(state['qpos'][1]):.1f}°, Joint 3: {np.rad2deg(state['qpos'][2]):.1f}°")

    # Final position
    print("\n5. Final state:")
    state = robot.get_state()
    print(f"   All joints (deg): {np.rad2deg(state['qpos'])}")
    print(f"   EE position (m): {robot.get_ee_position()}")

    print("\n✓ Test complete!")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\n✗ Error: {e}")
    if USE_REAL_ROBOT:
        robot.emergency_stop()
finally:
    robot.close()
    print("Disconnected")
