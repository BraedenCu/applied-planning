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

### Quick Start

From the project root directory with your environment activated:

```bash
# Make sure environment is activated
source .venv/bin/activate

# Launch Jupyter
jupyter notebook
# or
jupyter lab
```

### Joint Angle Limits (Lite6)

| Joint | Min    | Max   |
|-------|--------|-------|
| J1    | -360°  | 360°  |
| J2    | -118°  | 120°  |
| J3    | -225°  | 11°   |
| J4    | -360°  | 360°  |
| J5    | -97°   | 180°  |
| J6    | -360°  | 360°  |

## API Reference

The notebook uses the xArm Python SDK. Key methods:

- `arm.get_servo_angle()` - Read joint angles
- `arm.set_servo_angle(angle, speed, wait)` - Move joints
- `arm.get_position()` - Read Cartesian position
- `arm.motion_enable(enable)` - Enable/disable motors
- `arm.emergency_stop()` - Emergency stop
- `arm.clean_error()` - Clear error codes

Full documentation: https://github.com/xArm-Developer/xArm-Python-SDK
