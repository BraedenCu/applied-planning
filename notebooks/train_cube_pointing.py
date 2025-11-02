"""
Training script for Lite6 Cube Pointing with Stable Baselines3.

This script demonstrates how to train an RL agent to point the robot's
end-effector at a randomly placed cube.

Requirements:
    pip install stable-baselines3

Usage:
    python notebooks/train_cube_pointing.py
"""

import os
from pathlib import Path
import numpy as np

# Get project root (one level up from notebooks)
project_root = Path(os.getcwd()).parent if Path(os.getcwd()).name == "notebooks" else Path(os.getcwd())
model_path = project_root / "src/applied_planning/sim/assets/ufactory_lite6/lite6_gripper_narrow.xml"

# Import environment
from applied_planning.envs import Lite6CubePointingEnv

# Create environment
print("Creating environment...")
env = Lite6CubePointingEnv(
    model_path=str(model_path),
    render_mode=None,  # Set to "human" to watch training (will be slow!)
    max_steps=200,
    success_threshold=0.05,  # 5cm
    reward_scale=10.0,
    cube_placement_radius=0.3,
    ee_site_name="end_effector"
)

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Test environment with random actions
print("\n" + "="*60)
print("Testing environment with random policy...")
print("="*60)

obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Cube position: {info['cube_pos']}")
print(f"EE position: {info['ee_pos']}")
print(f"Initial distance: {info['distance']:.4f} m")

# Run a few random steps
for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1}: distance={info['distance']:.4f} m, reward={reward:.2f}")
    if terminated or truncated:
        break

print("\n✓ Environment test complete!")

# Optional: Train with Stable Baselines3
try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    print("\n" + "="*60)
    print("Stable Baselines3 detected - setting up training...")
    print("="*60)

    # Wrap environment for SB3
    def make_env():
        return Lite6CubePointingEnv(
            model_path=str(model_path),
            render_mode=None,
            max_steps=200,
            success_threshold=0.05,
            reward_scale=10.0,
            cube_placement_radius=0.3,
            ee_site_name="end_effector"
        )

    # Create vectorized environment
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./logs/checkpoints",
        name_prefix="cube_pointing_model"
    )

    # Create RL model - using SAC (works well for continuous control)
    print("\nCreating SAC model...")
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log="./logs/tensorboard"
    )

    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    print(f"Algorithm: SAC")
    print(f"Total timesteps: 100,000")
    print(f"Eval frequency: 5,000 steps")
    print(f"Checkpoint frequency: 10,000 steps")
    print("="*60)

    # Train the model
    print("\nStarting training...")
    print("(This will take a while. Press Ctrl+C to stop early)\n")

    try:
        model.learn(
            total_timesteps=100000,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )

        # Save final model
        model.save("./logs/cube_pointing_final")
        print("\n✓ Training complete! Model saved to ./logs/cube_pointing_final")

        # Test trained model
        print("\n" + "="*60)
        print("Testing trained model...")
        print("="*60)

        test_env = Lite6CubePointingEnv(
            model_path=str(model_path),
            render_mode="human",  # Render to see the trained agent
            max_steps=200,
            success_threshold=0.05,
            reward_scale=10.0,
            cube_placement_radius=0.3,
            ee_site_name="end_effector"
        )

        for episode in range(3):
            obs, info = test_env.reset()
            print(f"\nEpisode {episode+1} - Initial distance: {info['distance']:.4f} m")

            episode_reward = 0
            for step in range(200):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    print(f"  Finished in {step+1} steps")
                    print(f"  Final distance: {info['distance']:.4f} m")
                    print(f"  Success: {info['is_success']}")
                    print(f"  Total reward: {episode_reward:.2f}")
                    break

        test_env.close()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        model.save("./logs/cube_pointing_interrupted")
        print("Model saved to ./logs/cube_pointing_interrupted")

except ImportError:
    print("\n" + "="*60)
    print("Stable Baselines3 not installed")
    print("="*60)
    print("To train an RL agent, install with:")
    print("  pip install stable-baselines3")
    print("\nThen run this script again.")
    print("="*60)

finally:
    env.close()
    print("\nDone!")
