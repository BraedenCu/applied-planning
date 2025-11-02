import os
import time
from pathlib import Path

# Get project root (one level up from notebooks)
project_root = Path(os.getcwd()).parent

# Use the gripper model (narrow gripper for better cube grasping)
# Options: lite6.xml (no gripper), lite6_gripper_wide.xml, lite6_gripper_narrow.xml
model_path = project_root / "src/applied_planning/sim/assets/ufactory_lite6/lite6_gripper_narrow.xml"

from applied_planning.sim.adapters.mujoco_backend import MujocoLite6Adapter
import mujoco

# Create simulator with 3 cubes
sim = MujocoLite6Adapter(
    model_path=str(model_path),
    num_cubes=3,
    cube_placement_radius=0.3,
    viewer=True,
    render_mode="passive",
    ee_site_name="end_effector"  # Use gripper's end effector site
)

# Reset and get cube positions
obs = sim.reset()
cube_positions = sim.get_cube_positions()
print(f"Cube positions: {cube_positions}")

# Print cube poses
for i in range(3):
    pos, quat = sim.get_cube_pose(i)
    print(f"Cube {i}: pos={pos}, quat={quat}")

# Get gripper actuator info
print("\n" + "="*60)
print("Model Info:")
print(f"Number of actuators: {sim.model.nu}")
print(f"Number of joints: {sim.model.njnt}")
print(f"Actuator names: {[sim.model.actuator(i).name for i in range(sim.model.nu)]}")
print("="*60)

print("\nViewer is open. Gripper will open/close in a loop.")
print("Press Ctrl+C to exit...\n")

# Keep the viewer open and demonstrate gripper control
try:
    step_count = 0
    gripper_state = "closing"  # Start by closing

    while True:
        # Control the gripper (actuator index 6 is the gripper motor, 0-5 are arm joints)
        if sim.model.nu > 6:  # Make sure gripper actuator exists
            # Oscillate gripper between open and closed
            if step_count % 500 == 0:
                if gripper_state == "closing":
                    print("Closing gripper...")
                    sim.data.ctrl[6] = -10  # Close (negative force)
                    gripper_state = "opening"
                else:
                    print("Opening gripper...")
                    sim.data.ctrl[6] = 10   # Open (positive force)
                    gripper_state = "closing"

        # Step the physics simulation
        mujoco.mj_step(sim.model, sim.data)

        # Sync the viewer
        if sim._viewer is not None:
            sim._viewer.sync()

        # Small delay to prevent CPU overload
        time.sleep(0.01)
        step_count += 1

except KeyboardInterrupt:
    print("\nClosing viewer...")
    sim.close()
    print("Done!")