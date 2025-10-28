# uFactory Lite 6 Control Notebooks

Interactive Jupyter notebook for controlling the uFactory Lite 6 robot arm with sliders and real-time feedback.

## Setup

### First-Time Setup with uv

If you haven't set up the project environment yet:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project root
cd /path/to/applied-planning

# Create virtual environment
uv venv --python 3.10

# Activate the environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install with notebook support
uv pip install -e .[notebook]

# Enable Jupyter widgets
jupyter nbextension enable --py widgetsnbextension
```

### Quick Start (Environment Already Set Up)

From the project root directory with your environment activated:

```bash
# Make sure environment is activated
source .venv/bin/activate

# Launch Jupyter
jupyter notebook
# or
jupyter lab
```

Then open `ufactory_lite6_control.ipynb` and update the `ROBOT_IP` variable (cell 2) to match your robot's IP address.

## Features

### Interactive Joint Control
- **6 individual sliders** for precise joint angle control
- **Real-time position feedback** showing current joint angles and Cartesian pose
- **Speed control** with safety limits (default: 5 deg/s, max: 30 deg/s)
- **Visual status display** with color-coded state information

### Fine-Tuning Mode
- Adjust individual joints by small increments (0.1° steps)
- Delta adjustment for precise positioning
- Ultra-low speed control (1-10 deg/s) for safety

### Position Management
- **Save positions** with custom names
- **Load saved positions** instantly
- **List all saved positions** for reference

### Trajectory Recording
- Record multi-point trajectories by capturing waypoints
- Playback trajectories with adjustable speed and delay
- Save/load trajectories to JSON files
- Compatible with simulation trajectory files

## Safety Features

⚠️ **Important Safety Notes**:

1. **Low Default Speeds**: Default speed is set to 5 deg/s (very slow)
2. **Speed Limits**: Maximum speed capped at 30 deg/s
3. **Joint Limits**: Software limits enforce safe joint ranges for Lite6
4. **Emergency Stop**: Red STOP button available in all control interfaces
5. **Confirmation Required**: All movements require explicit button clicks

### Joint Angle Limits (Lite6)

| Joint | Min    | Max   |
|-------|--------|-------|
| J1    | -360°  | 360°  |
| J2    | -118°  | 120°  |
| J3    | -225°  | 11°   |
| J4    | -360°  | 360°  |
| J5    | -97°   | 180°  |
| J6    | -360°  | 360°  |

## Workflow Examples

### Example 1: Testing Individual Joints

1. Run cells 1-4 to connect to the robot
2. In Section 6 (Interactive Joint Control):
   - Use sliders to set desired angles
   - Start with small movements (±10°)
   - Click "Move Robot" to execute
   - Click "Update from Robot" to sync sliders with current position

### Example 2: Fine-Tuning a Position

1. Move to approximate position using Section 6
2. Use Section 7 (Fine-Tuning Individual Joints):
   - Select the joint to adjust
   - Use the delta slider for small adjustments (±0.1° to ±10°)
   - Click "Apply Adjustment"
3. Save the final position using Section 8

### Example 3: Recording a Trajectory

1. Move robot to first waypoint
2. Click "Record Waypoint"
3. Move to next position
4. Click "Record Waypoint" again
5. Repeat for all desired positions
6. Click "Play" to test the trajectory
7. Save to file for later use

### Example 4: Working with Simulation Data

If you have trajectory files from simulation (with `"joints"` key):

1. Use Section 9 (Trajectory Recording & Playback)
2. Enter filename in the text box
3. Click "Load" to import the trajectory
4. Adjust playback speed and delay
5. Click "Play" to execute

## Troubleshooting

### Robot Not Responding
- Check IP address is correct
- Ensure robot is powered on and connected to network
- Run the "reset_robot(arm)" cell in Section 11

### Connection Failed
- Verify network connectivity: `ping <robot-ip>`
- Check firewall settings
- Ensure no other applications are connected to the robot

### Movement Errors
- Check for error codes in status display
- Clear errors using Section 11: `reset_robot(arm)`
- Verify joint angles are within limits
- Reduce speed if movements are too fast

### Sliders Not Updating
- Click "Update from Robot" button
- Restart the kernel and reconnect

## Quick Reference

### Common Operations

```python
# Get current status
display_status(arm)

# Move to home position
go_home(arm, speed=10.0)

# Save current position
save_current_position('my_position')

# Load saved position
load_position('my_position', speed=5.0)

# Record current position as waypoint
record_waypoint()

# Play recorded trajectory
play_trajectory(speed=5.0, delay=0.5)

# Reset robot (clear errors)
reset_robot(arm)

# Disconnect
disconnect_robot(arm)
```

## API Reference

The notebook uses the xArm Python SDK. Key methods:

- `arm.get_servo_angle()` - Read joint angles
- `arm.set_servo_angle(angle, speed, wait)` - Move joints
- `arm.get_position()` - Read Cartesian position
- `arm.motion_enable(enable)` - Enable/disable motors
- `arm.emergency_stop()` - Emergency stop
- `arm.clean_error()` - Clear error codes

Full documentation: https://github.com/xArm-Developer/xArm-Python-SDK

## Support

For issues or questions:
- SDK Documentation: https://github.com/xArm-Developer/xArm-Python-SDK
- xArm User Manual: Check manufacturer documentation
- Report notebook issues to your project repository
