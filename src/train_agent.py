import os
from stable_baselines3 import PPO
from env import GridEnvironment

# 1. SETUP PATHS
# We fix the path dynamically so Windows doesn't get confused
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'processed', 'grid_data_clean.csv')
models_dir = os.path.join(current_dir, '..', 'models')

# Create models directory if it doesn't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

print("[INFO] Setting up Environment...")
# 2. INITIALIZE ENVIRONMENT
env = GridEnvironment(data_path=data_path)

# 3. INITIALIZE AGENT (The Brain)
# MlpPolicy = Multi-Layer Perceptron (Basic Neural Network)
# verbose=1 = Show us the training logs in the terminal
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

print("[INFO] Starting Training... (This will take a few minutes)")
# 4. TRAIN THE AGENT
# We train for 30,000 steps just to see if it learns.
# In a real project, you would do 1,000,000+
model.learn(total_timesteps=200000)

# 5. SAVE THE TRAINED MODEL
save_path = os.path.join(models_dir, "ppo_grid_agent")
model.save(save_path)

print(f"[SUCCESS] Model saved to {save_path}.zip")