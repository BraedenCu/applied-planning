"""
Quick test script for the Lite6 Cube Pointing environment.

This is a simplified version for quick testing in notebooks or interactive Python.
"""

import os
from pathlib import Path
import numpy as np
import time

# Get project root
project_root = Path(os.getcwd()).parent if Path(os.getcwd()).name == "notebooks" else Path(os.getcwd())
model_path = project_root / "src/applied_planning/sim/assets/ufactory_lite6/lite6_gripper_narrow.xml"

from applied_planning.envs import Lite6CubePointingEnv

print("="*70)
print("Lite6 Cube Pointing Environment Test")
print("="*70)

# Create environment
print("\nCreating environment...")
env = Lite6CubePointingEnv(
    model_path=str(model_path),
    render_mode="human",  # Set to None for headless
    max_steps=200,
    success_threshold=0.05,  # 5cm
    reward_scale=10.0,
    cube_placement_radius=0.3,
    ee_site_name="end_effector"
)

print(f"✓ Environment created")
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space.shape}")

# Reset and show initial state
print("\n" + "="*70)
print("Episode 1: Random Actions")
print("="*70)

obs, info = env.reset()
print(f"\nInitial State:")
print(f"  Cube position: {info['cube_pos']}")
print(f"  EE position:   {info['ee_pos']}")
print(f"  Distance:      {info['distance']:.4f} m")

# Run episode with random actions
episode_reward = 0
for step in range(200):
    # Random action
    action = env.action_space.sample()

    # Step
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

    # Print every 20 steps
    if step % 20 == 0:
        print(f"  Step {step:3d}: distance={info['distance']:.4f} m, reward={reward:6.2f}")

    if terminated or truncated:
        print(f"\n✓ Episode finished at step {step+1}")
        break

    # Small delay for visualization
    time.sleep(0.02)

print(f"\nFinal State:")
print(f"  Cube position: {info['cube_pos']}")
print(f"  EE position:   {info['ee_pos']}")
print(f"  Distance:      {info['distance']:.4f} m")
print(f"  Total reward:  {episode_reward:.2f}")
print(f"  Success:       {info['is_success']}")

# Test observation structure
print("\n" + "="*70)
print("Observation Structure")
print("="*70)
print(f"Full observation: {obs}")
print(f"\nParsed:")
print(f"  Cube position (obs[0:3]):  {obs[0:3]}")
print(f"  Joint positions (obs[3:9]): {obs[3:9]}")
print(f"  EE position (obs[9:12]):    {obs[9:12]}")

# Try a second episode
print("\n" + "="*70)
print("Episode 2: New Random Cube Position")
print("="*70)

obs, info = env.reset()
print(f"\nNew Initial State:")
print(f"  Cube position: {info['cube_pos']}")
print(f"  EE position:   {info['ee_pos']}")
print(f"  Distance:      {info['distance']:.4f} m")

print("\nRunning 50 random steps...")
for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 10 == 0:
        print(f"  Step {step:2d}: distance={info['distance']:.4f} m")

    if terminated or truncated:
        break

    time.sleep(0.02)

print(f"\nFinal distance: {info['distance']:.4f} m")

# Clean up
env.close()
print("\n" + "="*70)
print("Test Complete!")
print("="*70)
print("\nNext steps:")
print("  1. Run scripts/demo_cube_pointing_env.py for full demo")
print("  2. Run notebooks/train_cube_pointing.py to train an RL agent")
print("  3. See docs/cube_pointing_rl.md for detailed documentation")
