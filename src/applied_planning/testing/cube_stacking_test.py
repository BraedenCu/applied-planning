#!/usr/bin/env python3
"""Test script for trained cube stacking model in MuJoCo viewer."""

import time
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from applied_planning.envs import Lite6CubeStackingEnv

# Setup paths
BASE_PATH = Path("/Users/braeden/Development/applied-planning/")
TRAINING_DIR = BASE_PATH / "src/applied_planning/training"
MODEL_PATH = BASE_PATH / "src/applied_planning/sim/assets/ufactory_lite6/lite6_gripper_narrow.xml"

print("="*60)
print("Testing Trained Cube Stacking Model in MuJoCo Viewer")
print("="*60)

# Try to load the best model, fallback to final model
try:
    model_file = TRAINING_DIR / "logs/cube_stacking/best_model/best_model.zip"
    model = SAC.load(str(model_file))
    print("✓ Loaded best model from evaluation")
except:
    try:
        model_file = TRAINING_DIR / "logs/cube_stacking/cube_stacking_final.zip"
        model = SAC.load(str(model_file))
        print("✓ Loaded final model")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease train the model first by running the cube_stacking.ipynb notebook.")
        exit(1)

# Create environment with rendering enabled
test_env = Lite6CubeStackingEnv(
    model_path=str(MODEL_PATH),
    render_mode="human",  # This will open MuJoCo viewer
    max_steps=300,
    success_threshold=0.08,
    horizontal_threshold=0.05,
    reward_scale=10.0,
    cube_placement_radius=0.3,
    ee_site_name="end_effector",
    collision_penalty=100.0,
    terminate_on_collision=False
)

print("\nRunning 5 test episodes...")
print("(MuJoCo viewer window should open)")
print("\nWatch the robot attempt to stack cube1 on top of cube2!\n")

# Track statistics
success_count = 0
total_rewards = []
collision_count = 0
final_vertical_distances = []
final_horizontal_distances = []

for episode in range(5):
    obs, info = test_env.reset()
    print(f"\n{'='*60}")
    print(f"Episode {episode+1}:")
    print(f"{'='*60}")
    print(f"  Cube 1 position: {info['cube1_pos']}")
    print(f"  Cube 2 position: {info['cube2_pos']}")
    print(f"  Initial vertical distance: {info['vertical_distance']:.4f} m")
    print(f"  Initial horizontal distance: {info['horizontal_distance']:.4f} m")

    episode_reward = 0
    episode_collisions = 0
    done = False

    for step in range(300):
        # Get action from trained model
        action, _states = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_reward += reward

        # Track collisions
        if info['has_collision']:
            episode_collisions += 1

        # Small delay so we can watch the movement
        time.sleep(0.02)

        if terminated or truncated:
            done = True
            break

    # Print episode results
    print(f"\n  Results:")
    print(f"    Steps taken: {step+1}")
    print(f"    Final vertical distance: {info['vertical_distance']:.4f} m")
    print(f"    Final horizontal distance: {info['horizontal_distance']:.4f} m")
    print(f"    Cube1 above Cube2: {info['cube1_pos'][2] > info['cube2_pos'][2]}")
    print(f"    Success (stacked): {info['is_success']}")
    print(f"    Self-collisions: {episode_collisions}")
    print(f"    Total reward: {episode_reward:.2f}")

    # Update statistics
    if info['is_success']:
        success_count += 1
        print(f"    ✓ SUCCESSFULLY STACKED!")

    collision_count += episode_collisions
    total_rewards.append(episode_reward)
    final_vertical_distances.append(info['vertical_distance'])
    final_horizontal_distances.append(info['horizontal_distance'])

test_env.close()

# Print summary
print("\n" + "="*60)
print("Test Summary:")
print("="*60)
print(f"Success rate: {success_count}/5 ({success_count/5*100:.1f}%)")
print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
print(f"Average final vertical distance: {np.mean(final_vertical_distances):.4f} m")
print(f"Average final horizontal distance: {np.mean(final_horizontal_distances):.4f} m")
print(f"Total self-collisions: {collision_count}")
print("="*60)

print("\n")
if success_count > 0:
    print(f"🎉 The robot successfully stacked cubes in {success_count}/5 episodes!")
else:
    print("The robot needs more training to successfully stack cubes.")
    print("Consider training for more timesteps or adjusting hyperparameters.")
