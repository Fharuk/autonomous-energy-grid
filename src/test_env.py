import os
from env import GridEnvironment

# 1. Get the absolute path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'processed', 'grid_data_clean.csv')

print(f"Looking for data at: {data_path}")

try:
    # 2. Initialize
    env = GridEnvironment(data_path=data_path)
    obs, _ = env.reset()
    print("[SUCCESS] Initial State Loaded:", obs)

    # 3. Run 10 Random Steps
    print("\nRunning Simulation...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}: Action={action[0]:.2f}, Cost=${info['cost']:.2f}, Battery={info['battery_soc']:.2f}, Reward={reward:.2f}")

    print("\n[DONE] Environment is valid. Ready for training.")

except Exception as e:
    print(f"\n[ERROR] Something went wrong: {e}")