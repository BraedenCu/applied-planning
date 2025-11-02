# Load and test the trained model in MuJoCo viewer
import time
from stable_baselines3 import SAC
from applied_planning.envs import Lite6CubePointingEnv
import numpy as np

BASE_PATH = "/Users/braeden/Development/applied-planning/"
LOGS_DIR = BASE_PATH + "logs/"
MODEL_DIR = LOGS_DIR + "best_model/"
BASE_PATH = "/Users/braeden/Development/applied-planning/"
model_path = BASE_PATH + "src/applied_planning/sim/assets/ufactory_lite6/lite6_gripper_narrow.xml"

print("="*60)
print("Testing Trained Model in MuJoCo Viewer")
print("="*60)

model = SAC.load("/Users/braeden/Development/applied-planning/logs/best_model/best_model.zip")
print("✓ Loaded best model from evaluation")

# Create environment with rendering enabled
test_env = Lite6CubePointingEnv(
    model_path=str(model_path),
    render_mode="human",  # This will open MuJoCo viewer
    max_steps=200,
    success_threshold=0.05,
    reward_scale=10.0,
    cube_placement_radius=0.3,
    ee_site_name="end_effector"
)

print("\nRunning 5 test episodes...")
print("(MuJoCo viewer window should open)\n")

success_count = 0
total_rewards = []

for episode in range(5):
    obs, info = test_env.reset()
    print(f"\nEpisode {episode+1}:")
    print(f"  Cube position: {info['cube_pos']}")
    print(f"  Initial distance: {info['distance']:.4f} m")
    
    episode_reward = 0
    done = False
    
    for step in range(200):
        # Get action from trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_reward += reward
        
        # Small delay so we can watch the movement
        time.sleep(0.02)
        
        if terminated or truncated:
            done = True
            break
    
    # Print episode results
    print(f"  Steps taken: {step+1}")
    print(f"  Final distance: {info['distance']:.4f} m")
    print(f"  Success: {info['is_success']}")
    print(f"  Total reward: {episode_reward:.2f}")
    
    if info['is_success']:
        success_count += 1
    total_rewards.append(episode_reward)

test_env.close()

print("\n" + "="*60)
print("Test Summary:")
print("="*60)
print(f"Success rate: {success_count}/5 ({success_count/5*100:.1f}%)")
print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
print("="*60)