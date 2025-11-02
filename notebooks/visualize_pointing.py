"""
Visualize robot pointing at cube with visual feedback.

This script shows:
1. The robot and cube in MuJoCo viewer
2. Distance information printed to console
3. Step-by-step progress
4. Final success/failure result

For advanced visualization with plots, see visualize_pointing_with_plots.py
"""

import os
from pathlib import Path
import numpy as np
import time

# Get project root
project_root = Path(os.getcwd()).parent if Path(os.getcwd()).name == "notebooks" else Path(os.getcwd())
model_path = project_root / "src/applied_planning/sim/assets/ufactory_lite6/lite6_gripper_narrow.xml"

from applied_planning.envs import Lite6CubePointingEnv


def run_pointing_trial(env, policy="random", max_steps=200, verbose=True):
    """Run a single pointing trial and visualize the result.

    Args:
        env: The cube pointing environment
        policy: "random" for random actions, or a trained model
        max_steps: Maximum steps to run
        verbose: Print detailed step information

    Returns:
        Dict with trial results
    """
    # Reset environment
    obs, info = env.reset()

    # Print initial state
    print("\n" + "="*70)
    print("POINTING TRIAL START")
    print("="*70)
    print(f"Cube position:        [{info['cube_pos'][0]:6.3f}, {info['cube_pos'][1]:6.3f}, {info['cube_pos'][2]:6.3f}]")
    print(f"Initial EE position:  [{info['ee_pos'][0]:6.3f}, {info['ee_pos'][1]:6.3f}, {info['ee_pos'][2]:6.3f}]")
    print(f"Initial distance:     {info['distance']:.4f} m ({info['distance']*100:.2f} cm)")
    print("="*70)

    # Track metrics
    distances = [info['distance']]
    rewards = []
    positions = {'ee': [info['ee_pos']], 'cube': info['cube_pos']}

    # Run trial
    episode_reward = 0
    min_distance = info['distance']

    for step in range(max_steps):
        # Get action from policy
        if policy == "random":
            action = env.action_space.sample()
        else:
            # Assume policy is a trained model
            action, _ = policy.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Track metrics
        episode_reward += reward
        distances.append(info['distance'])
        positions['ee'].append(info['ee_pos'])
        rewards.append(reward)

        # Update min distance
        if info['distance'] < min_distance:
            min_distance = info['distance']

        # Print progress
        if verbose and (step % 20 == 0 or step < 5):
            arrow = "↓" if len(distances) > 1 and distances[-1] < distances[-2] else "↑"
            print(f"Step {step:3d}: distance={info['distance']:.4f} m ({info['distance']*100:5.2f} cm) {arrow}  reward={reward:7.2f}")

        # Check termination
        if terminated or truncated:
            break

        # Small delay for visualization
        time.sleep(0.02)

    # Print final results
    print("\n" + "="*70)
    print("TRIAL RESULTS")
    print("="*70)
    print(f"Final EE position:    [{info['ee_pos'][0]:6.3f}, {info['ee_pos'][1]:6.3f}, {info['ee_pos'][2]:6.3f}]")
    print(f"Cube position:        [{info['cube_pos'][0]:6.3f}, {info['cube_pos'][1]:6.3f}, {info['cube_pos'][2]:6.3f}]")
    print(f"Final distance:       {info['distance']:.4f} m ({info['distance']*100:.2f} cm)")
    print(f"Min distance reached: {min_distance:.4f} m ({min_distance*100:.2f} cm)")
    print(f"Total steps:          {step + 1}")
    print(f"Total reward:         {episode_reward:.2f}")
    print(f"Average reward:       {episode_reward/(step+1):.2f}")

    # Success evaluation
    if info['is_success']:
        print(f"\n✓ SUCCESS! End-effector within {env.success_threshold*100:.1f}cm of cube")
    else:
        print(f"\n✗ FAILED - Distance {info['distance']*100:.2f}cm exceeds threshold {env.success_threshold*100:.1f}cm")

    # Distance improvement
    improvement = (distances[0] - info['distance']) * 100
    improvement_pct = (improvement / (distances[0] * 100)) * 100
    if improvement > 0:
        print(f"   Improved by {improvement:.2f}cm ({improvement_pct:.1f}%)")
    else:
        print(f"   Worsened by {-improvement:.2f}cm ({-improvement_pct:.1f}%)")

    print("="*70)

    # Return results
    return {
        'success': info['is_success'],
        'final_distance': info['distance'],
        'min_distance': min_distance,
        'initial_distance': distances[0],
        'steps': step + 1,
        'total_reward': episode_reward,
        'distances': distances,
        'rewards': rewards,
        'ee_positions': positions['ee'],
        'cube_position': positions['cube']
    }


def run_multiple_trials(env, policy="random", num_trials=5, verbose=False):
    """Run multiple pointing trials and show statistics.

    Args:
        env: The cube pointing environment
        policy: "random" for random actions, or a trained model
        num_trials: Number of trials to run
        verbose: Print detailed step information for each trial

    Returns:
        List of trial results
    """
    print("\n" + "="*70)
    print(f"RUNNING {num_trials} POINTING TRIALS")
    print("="*70)

    results = []
    for trial_num in range(num_trials):
        print(f"\n{'─'*70}")
        print(f"Trial {trial_num + 1}/{num_trials}")
        print(f"{'─'*70}")

        result = run_pointing_trial(env, policy=policy, verbose=verbose)
        results.append(result)

        # Brief pause between trials
        if trial_num < num_trials - 1:
            time.sleep(1.0)

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    successes = sum(1 for r in results if r['success'])
    success_rate = (successes / num_trials) * 100

    avg_final_dist = np.mean([r['final_distance'] for r in results])
    avg_min_dist = np.mean([r['min_distance'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['total_reward'] for r in results])

    print(f"Success rate:         {successes}/{num_trials} ({success_rate:.1f}%)")
    print(f"Average final dist:   {avg_final_dist:.4f} m ({avg_final_dist*100:.2f} cm)")
    print(f"Average min dist:     {avg_min_dist:.4f} m ({avg_min_dist*100:.2f} cm)")
    print(f"Average steps:        {avg_steps:.1f}")
    print(f"Average total reward: {avg_reward:.2f}")
    print("="*70)

    return results


def main():
    """Main function to run visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize robot pointing at cube")
    parser.add_argument("--model", type=str,
                       default="src/applied_planning/sim/assets/ufactory_lite6/lite6_gripper_narrow.xml",
                       help="Path to MuJoCo model")
    parser.add_argument("--trials", type=int, default=1,
                       help="Number of trials to run")
    parser.add_argument("--policy", type=str, default="random",
                       help="Policy to use: 'random' or path to trained model")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per trial")
    parser.add_argument("--headless", action="store_true",
                       help="Run without viewer")
    parser.add_argument("--quiet", action="store_true",
                       help="Less verbose output")
    args = parser.parse_args()

    # Resolve model path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Load policy if specified
    policy = "random"
    if args.policy != "random":
        try:
            from stable_baselines3 import SAC
            policy = SAC.load(args.policy)
            print(f"Loaded trained policy from {args.policy}")
        except Exception as e:
            print(f"Error loading policy: {e}")
            print("Using random policy instead")
            policy = "random"

    # Create environment
    print("Creating environment...")
    env = Lite6CubePointingEnv(
        model_path=str(model_path),
        render_mode=None if args.headless else "human",
        max_steps=args.max_steps,
        success_threshold=0.05,
        reward_scale=10.0,
        cube_placement_radius=0.3,
        ee_site_name="end_effector"
    )

    print(f"✓ Environment created")

    try:
        if args.trials == 1:
            # Single trial with verbose output
            run_pointing_trial(env, policy=policy, max_steps=args.max_steps,
                             verbose=not args.quiet)
        else:
            # Multiple trials with summary
            run_multiple_trials(env, policy=policy, num_trials=args.trials,
                              verbose=not args.quiet)

        # Keep viewer open
        if not args.headless:
            print("\nViewer open. Press Ctrl+C to exit...")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nExiting...")

    finally:
        env.close()
        print("\nDone!")


if __name__ == "__main__":
    main()
