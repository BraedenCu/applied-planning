#!/usr/bin/env python3
"""
Demo script for the Lite6 Cube Pointing RL environment.

This script demonstrates:
1. How to create and use the Lite6CubePointingEnv
2. Random action rollout to visualize the environment
3. Observation and reward structure

For actual RL training, use with Stable Baselines3 or similar library.

Usage:
    python scripts/demo_cube_pointing_env.py
    python scripts/demo_cube_pointing_env.py --episodes 5
    python scripts/demo_cube_pointing_env.py --headless
"""

import argparse
import time
import numpy as np
from pathlib import Path

# Add src to path if running directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applied_planning.envs import Lite6CubePointingEnv


def run_random_policy(env, num_episodes=3, render=True):
    """Run random policy to demonstrate the environment.

    Args:
        env: The cube pointing environment
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    print("=" * 70)
    print("Running Random Policy")
    print("=" * 70)

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        obs, info = env.reset()
        print(f"Initial cube position: {info['cube_pos']}")
        print(f"Initial EE position:   {info['ee_pos']}")
        print(f"Initial distance:      {info['distance']:.4f} m")

        episode_reward = 0
        step_count = 0
        done = False

        while not done:
            # Take random action
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            done = terminated or truncated

            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"  Step {step_count}: distance={info['distance']:.4f} m, reward={reward:.2f}")

            # Render if enabled
            if render:
                env.render()
                time.sleep(0.01)

        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total steps:       {step_count}")
        print(f"  Total reward:      {episode_reward:.2f}")
        print(f"  Final distance:    {info['distance']:.4f} m")
        print(f"  Success:           {info['is_success']}")
        print(f"  Final cube pos:    {info['cube_pos']}")
        print(f"  Final EE pos:      {info['ee_pos']}")

    print("\n" + "=" * 70)
    print("Random Policy Demo Complete")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Demo cube pointing RL environment")
    parser.add_argument(
        "--model",
        type=str,
        default="src/applied_planning/sim/assets/ufactory_lite6/lite6_gripper_narrow.xml",
        help="Path to MuJoCo XML model"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode"
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
        print("  python scripts/demo_cube_pointing_env.py")
        return

    print("=" * 70)
    print("Lite6 Cube Pointing Environment Demo")
    print("=" * 70)
    print(f"Model:      {model_path}")
    print(f"Episodes:   {args.episodes}")
    print(f"Max steps:  {args.max_steps}")
    print(f"Render:     {'No (headless)' if args.headless else 'Yes'}")
    print("=" * 70)

    # Create environment
    print("\nInitializing environment...")
    env = Lite6CubePointingEnv(
        model_path=str(model_path),
        render_mode=None if args.headless else "human",
        max_steps=args.max_steps,
        success_threshold=0.05,  # 5cm
        reward_scale=10.0,
        cube_placement_radius=0.3,
        ee_site_name="end_effector"
    )

    print("✓ Environment initialized")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space:      {env.action_space}")

    # Run random policy
    try:
        run_random_policy(env, num_episodes=args.episodes, render=not args.headless)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        env.close()
        print("\nEnvironment closed.")


if __name__ == "__main__":
    main()
