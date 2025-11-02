# Applied Planning & Optimization

6DOF Robot Arm Path Planning in MuJoCo

Contributors: Braeden, Shaurya, Avril

## Overview

Simulation and control framework for the uFactory Lite 6 robot arm with path planning capabilities. Supports both MuJoCo (macOS-friendly) and future Isaac Sim backends.

## Quick Start

```bash
# Install dependencies
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .

# Optional: Install MuJoCo
uv pip install mujoco

# Fetch robot assets
python scripts/fetch_menagerie_lite6.py

# Run path planning demo with real-time visualization
python scripts/demo_path_planning.py

# On macosx, append with mjpython for mujoco ui to properly load
mjpython scripts/demo_path_planning.py

# Or run with different options
python scripts/demo_path_planning.py --planner linear --speed 0.5
```

## Features

- **Path Planning**: RRT and linear interpolation planners for 6DOF arms
- **MuJoCo Simulation**: Full integration with collision detection and visualization
- **Modular Architecture**: Easy to swap simulators, planners, and controllers
- **Joint & Cartesian Planning**: Support for both joint-space and Cartesian-space planning

## Real-Time Visualization

To see the robot motion in **real-time in MuJoCo viewer**, run the demo script (not in Jupyter):

```bash
# Default: RRT planner with interactive viewer
python scripts/demo_path_planning.py

# Slower motion for better visualization
python scripts/demo_path_planning.py --speed 0.5

# Use linear interpolation planner (faster planning)
python scripts/demo_path_planning.py --planner linear

# Run headless (no viewer)
python scripts/demo_path_planning.py --no-viewer

# On a macbook, you will need to run with mjpython to actually see the mujoco viewer
```

**Note:** The interactive MuJoCo viewer only works when running as a Python script. It does **not** work in Jupyter notebooks on macOS.

## Usage Example

### Python Script

```python
from applied_planning.sim.adapters.mujoco_backend import MujocoLite6Adapter
import numpy as np

# Initialize simulator with viewer
sim = MujocoLite6Adapter("path/to/ufactory_lite6.xml", viewer=True)
sim.reset()

# Define goal configuration
goal = np.array([0.5, -0.3, 0.8, 0.0, 1.0, 0.0])

# Plan and execute path
path = sim.plan_and_execute_path(
    goal=goal,
    planner_type="rrt",  # or "linear"
    joint_limits=sim.get_joint_limits(),
    collision_fn=sim.check_collision,
    execute=True
)

if path:
    print(f"Path found with {len(path)} waypoints")
else:
    print("No path found")
```

### Jupyter Notebook

```python
# Note: Interactive viewer doesn't work in notebooks on macOS
# Use headless or offscreen rendering instead

from applied_planning.sim.adapters.mujoco_backend import MujocoLite6Adapter
import numpy as np

# Initialize in headless mode
sim = MujocoLite6Adapter(
    "path/to/ufactory_lite6.xml",
    viewer=False,
    render_mode="none"  # or "offscreen" for visualization
)
sim.reset()

# Plan path
goal = np.array([0.5, -0.3, 0.8, 0.0, 1.0, 0.0])
path = sim.plan_and_execute_path(
    goal=goal,
    planner_type="rrt",
    joint_limits=sim.get_joint_limits(),
    execute=False
)

# Optionally visualize with offscreen rendering
if sim.render_mode == "offscreen":
    import matplotlib.pyplot as plt
    img = sim.render_notebook()
    plt.imshow(img)
    plt.show()
```

## Architecture

**Modular design** with swappable components:

- **Sim Adapters**: MuJoCo (current), Isaac Sim (future)
- **Planners**: RRT, linear interpolation, Cartesian planning (IK-based)
- **Controllers**: Joint velocity/torque, Cartesian impedance
- **Environments**: Gymnasium API for RL integration

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Installation

```bash
# Install uv (macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv venv --python 3.10 && source .venv/bin/activate

# Install options
uv pip install -e .              # Base
uv pip install -e .[dev]         # With dev tools
uv pip install -e .[notebook]    # With Jupyter support
uv pip install -e .[dev,notebook] # All features

# Install MuJoCo (optional)
uv pip install mujoco
```

## Path Planning Methods

### RRT (Rapidly-exploring Random Tree)

- Samples random configurations in joint space
- Efficient for high-DOF robots with obstacles
- Probabilistically complete

### Linear Interpolation

- Fast, straight-line paths in joint space
- Good for obstacle-free environments
- Useful for quick testing

### Cartesian Planning

- Plans in task space (end-effector pose)
- Requires inverse kinematics solver
- Ideal for precise positioning tasks

## Documentation

- [MuJoCo uFactory Lite6](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_lite6)
- [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
- [uFactory Lite6 Manual](https://cdn.robotshop.com/media/U/Ufa/RB-Ufa-32/pdf/ufactory-lite-6-user-manual.pdf)
