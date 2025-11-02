"""
Advanced visualization of robot pointing trials with matplotlib plots.

This script creates detailed visualizations including:
1. Distance over time plot
2. Reward over time plot
3. 3D trajectory visualization
4. Success metrics

Usage:
    python notebooks/visualize_pointing_with_plots.py
    python notebooks/visualize_pointing_with_plots.py --policy logs/cube_pointing_final.zip
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Get project root
project_root = Path(os.getcwd()).parent if Path(os.getcwd()).name == "notebooks" else Path(os.getcwd())
model_path = project_root / "src/applied_planning/sim/assets/ufactory_lite6/lite6_gripper_narrow.xml"

from applied_planning.envs import Lite6CubePointingEnv


def run_trial_and_collect_data(env, policy="random", max_steps=200):
    """Run trial and collect detailed data for visualization."""
    obs, info = env.reset()

    # Collect data
    data = {
        'distances': [info['distance']],
        'rewards': [],
        'ee_positions': [info['ee_pos'].copy()],
        'cube_position': info['cube_pos'].copy(),
        'joint_positions': [obs[3:9].copy()],
        'success': False,
        'steps': 0
    }

    episode_reward = 0

    for step in range(max_steps):
        # Get action
        if policy == "random":
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=True)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        # Collect data
        data['distances'].append(info['distance'])
        data['rewards'].append(reward)
        data['ee_positions'].append(info['ee_pos'].copy())
        data['joint_positions'].append(obs[3:9].copy())
        episode_reward += reward

        if terminated or truncated:
            data['success'] = info['is_success']
            data['steps'] = step + 1
            break

    data['total_reward'] = episode_reward
    return data


def plot_trial_results(data, save_path=None):
    """Create comprehensive visualization of trial results.

    Args:
        data: Dictionary containing trial data
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 10))

    # 1. Distance over time
    ax1 = plt.subplot(2, 3, 1)
    steps = np.arange(len(data['distances']))
    ax1.plot(steps, np.array(data['distances']) * 100, 'b-', linewidth=2, label='Distance')
    ax1.axhline(y=5.0, color='g', linestyle='--', linewidth=2, label='Success threshold (5cm)')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Distance (cm)', fontsize=12)
    ax1.set_title('Distance to Cube Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add min/max annotations
    min_dist = np.min(data['distances']) * 100
    min_step = np.argmin(data['distances'])
    ax1.annotate(f'Min: {min_dist:.2f}cm',
                xy=(min_step, min_dist),
                xytext=(min_step + 10, min_dist + 2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    # 2. Reward over time
    ax2 = plt.subplot(2, 3, 2)
    reward_steps = np.arange(len(data['rewards']))
    ax2.plot(reward_steps, data['rewards'], 'r-', linewidth=1.5, alpha=0.7, label='Step reward')
    ax2.plot(reward_steps, np.cumsum(data['rewards']), 'g-', linewidth=2, label='Cumulative reward')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. 3D trajectory
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    ee_positions = np.array(data['ee_positions'])
    cube_pos = data['cube_position']

    # Plot EE trajectory
    ax3.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
            'b-', linewidth=2, alpha=0.6, label='EE trajectory')

    # Plot start and end positions
    ax3.scatter(*ee_positions[0], color='green', s=200, marker='o',
               edgecolors='black', linewidth=2, label='Start', zorder=5)
    ax3.scatter(*ee_positions[-1], color='blue', s=200, marker='o',
               edgecolors='black', linewidth=2, label='End', zorder=5)

    # Plot cube
    ax3.scatter(*cube_pos, color='red', s=300, marker='s',
               edgecolors='black', linewidth=2, label='Cube', zorder=5)

    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.set_zlabel('Z (m)', fontsize=10)
    ax3.set_title('3D End-Effector Trajectory', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Distance improvement
    ax4 = plt.subplot(2, 3, 4)
    distances_cm = np.array(data['distances']) * 100
    improvements = distances_cm[:-1] - distances_cm[1:]  # Positive = improvement
    improvement_steps = np.arange(len(improvements))

    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax4.bar(improvement_steps, improvements, color=colors, alpha=0.6)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Step', fontsize=12)
    ax4.set_ylabel('Distance Change (cm)', fontsize=12)
    ax4.set_title('Step-by-Step Distance Improvement', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Joint positions over time
    ax5 = plt.subplot(2, 3, 5)
    joint_positions = np.array(data['joint_positions'])
    for i in range(6):
        ax5.plot(joint_positions[:, i], label=f'Joint {i+1}', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Step', fontsize=12)
    ax5.set_ylabel('Joint Angle (rad)', fontsize=12)
    ax5.set_title('Joint Positions Over Time', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best', fontsize=8)

    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate statistics
    initial_dist = data['distances'][0] * 100
    final_dist = data['distances'][-1] * 100
    min_dist = np.min(data['distances']) * 100
    total_reward = data['total_reward']
    avg_reward = total_reward / max(1, data['steps'])

    # Create summary text
    summary_text = f"""
TRIAL SUMMARY
{'='*40}

Status: {'✓ SUCCESS' if data['success'] else '✗ FAILED'}
Steps: {data['steps']}

DISTANCES (cm)
  Initial:  {initial_dist:6.2f}
  Final:    {final_dist:6.2f}
  Minimum:  {min_dist:6.2f}
  Improved: {initial_dist - final_dist:6.2f} ({((initial_dist-final_dist)/initial_dist*100):.1f}%)

REWARDS
  Total:    {total_reward:8.2f}
  Average:  {avg_reward:8.2f}

CUBE POSITION
  X: {cube_pos[0]:6.3f} m
  Y: {cube_pos[1]:6.3f} m
  Z: {cube_pos[2]:6.3f} m

FINAL EE POSITION
  X: {ee_positions[-1, 0]:6.3f} m
  Y: {ee_positions[-1, 1]:6.3f} m
  Z: {ee_positions[-1, 2]:6.3f} m
    """

    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def compare_policies(env, policies, policy_names, num_trials=10):
    """Compare multiple policies and visualize results.

    Args:
        env: Environment instance
        policies: List of policies (or "random")
        policy_names: List of policy names for display
        num_trials: Number of trials per policy
    """
    results = {name: [] for name in policy_names}

    print("Running comparison trials...")
    for policy, name in zip(policies, policy_names):
        print(f"\nTesting policy: {name}")
        for trial in range(num_trials):
            data = run_trial_and_collect_data(env, policy=policy, max_steps=200)
            results[name].append(data)
            print(f"  Trial {trial+1}/{num_trials}: distance={data['distances'][-1]*100:.2f}cm, success={data['success']}")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Success rates
    ax = axes[0, 0]
    success_rates = [np.mean([d['success'] for d in results[name]]) * 100
                    for name in policy_names]
    ax.bar(policy_names, success_rates, color=['blue', 'green', 'orange'][:len(policy_names)])
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Average final distances
    ax = axes[0, 1]
    avg_distances = [np.mean([d['distances'][-1] * 100 for d in results[name]])
                    for name in policy_names]
    ax.bar(policy_names, avg_distances, color=['blue', 'green', 'orange'][:len(policy_names)])
    ax.set_ylabel('Average Final Distance (cm)', fontsize=12)
    ax.set_title('Final Distance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Average steps
    ax = axes[1, 0]
    avg_steps = [np.mean([d['steps'] for d in results[name]])
                for name in policy_names]
    ax.bar(policy_names, avg_steps, color=['blue', 'green', 'orange'][:len(policy_names)])
    ax.set_ylabel('Average Steps', fontsize=12)
    ax.set_title('Steps to Completion', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Average total reward
    ax = axes[1, 1]
    avg_rewards = [np.mean([d['total_reward'] for d in results[name]])
                  for name in policy_names]
    ax.bar(policy_names, avg_rewards, color=['blue', 'green', 'orange'][:len(policy_names)])
    ax.set_ylabel('Average Total Reward', fontsize=12)
    ax.set_title('Total Reward Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('policy_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to policy_comparison.png")
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize pointing with plots")
    parser.add_argument("--policy", type=str, default="random",
                       help="Path to trained model or 'random'")
    parser.add_argument("--trials", type=int, default=1,
                       help="Number of trials to visualize")
    parser.add_argument("--compare", action="store_true",
                       help="Compare random vs trained policy")
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save plots")
    args = parser.parse_args()

    # Create environment (headless for plotting)
    env = Lite6CubePointingEnv(
        model_path=str(model_path),
        render_mode=None,
        max_steps=200,
        success_threshold=0.05,
        cube_placement_radius=0.3,
        ee_site_name="end_effector"
    )

    # Load policy if specified
    policy = "random"
    if args.policy != "random":
        try:
            from stable_baselines3 import SAC
            policy = SAC.load(args.policy)
            print(f"✓ Loaded trained policy from {args.policy}")
        except Exception as e:
            print(f"Error loading policy: {e}")
            print("Using random policy")

    if args.compare and args.policy != "random":
        # Compare policies
        compare_policies(env, [policy, "random"], ["Trained", "Random"], num_trials=10)
    else:
        # Single trial visualization
        for trial in range(args.trials):
            print(f"\nRunning trial {trial + 1}/{args.trials}...")
            data = run_trial_and_collect_data(env, policy=policy, max_steps=200)

            save_path = args.save
            if args.trials > 1 and save_path:
                save_path = f"{Path(save_path).stem}_trial{trial+1}.png"

            plot_trial_results(data, save_path=save_path)

    env.close()


if __name__ == "__main__":
    main()
