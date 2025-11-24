import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class GridEnvironment(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    The Agent controls a battery to minimize energy costs.
    """
    
    def __init__(self, data_path='data/processed/grid_data_clean.csv'):
        super(GridEnvironment, self).__init__()
        
        # 1. LOAD DATA
        # We load the clean CSV you just created
        self.df = pd.read_csv(data_path)
        self.max_steps = len(self.df)
        
        # 2. DEFINE SYSTEM LIMITS (The "Physics")
        self.BATTERY_CAPACITY_KWH = 500.0  # Max size of battery
        self.MAX_CHARGE_RATE_KW = 100.0    # Max speed to charge/discharge
        self.initial_battery_soc = 0.5     # Start at 50% charge
        
        # 3. DEFINE ACTION SPACE (Output of AI)
        # The AI outputs ONE number between -1 and +1
        # -1.0 = Max Discharge (Sell/Use Energy)
        # +1.0 = Max Charge (Store Energy)
        #  0.0 = Do Nothing
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 4. DEFINE OBSERVATION SPACE (Input to AI)
        # The AI sees: [Battery_SOC (%), Current_Load, Current_Solar, Current_Price, Hour_of_Day]
        # We normalize these values generally between 0 and 1 for better learning
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(5,), 
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.battery_soc = self.initial_battery_soc # State of Charge (0.0 to 1.0)
        
    def reset(self, seed=None, options=None):
        """
        Resets the environment to a random day to start a new training episode.
        """
        super().reset(seed=seed)
        
        # Pick a random start time (but leave enough space for a full episode)
        self.current_step = np.random.randint(0, self.max_steps - 96) # 96 steps = 24 hours
        self.battery_soc = self.initial_battery_soc
        
        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        """
        Helper to package the current state for the AI.
        """
        # Get data from DataFrame
        current_data = self.df.iloc[self.current_step]
        
        obs = np.array([
            self.battery_soc,                              # Battery Level (0-1)
            current_data['load_kw'] / 200.0,               # Normalize Load (Assume max ~200)
            current_data['solar_kw'] / 200.0,              # Normalize Solar
            current_data['price_per_kwh'],                 # Price (already small, 0.12 - 0.50)
            (self.current_step % 96) / 96.0                # Time of Day (0.0 to 1.0)
        ], dtype=np.float32)
        return obs

    def step(self, action):
        """
        The Physics Engine: Calculates what happens after the AI decides to Charge/Discharge.
        """
        # 1. Unpack Action
        # Action is [-1, 1]. Convert to actual kW.
        battery_action_kw = float(action[0]) * self.MAX_CHARGE_RATE_KW
        
        # 2. Apply Battery Physics (Constraints)
        energy_change_kwh = battery_action_kw * 0.25 
        
        # Update Battery State
        new_soc = self.battery_soc + (energy_change_kwh / self.BATTERY_CAPACITY_KWH)
        
        # CLIPPING: You cannot charge past 100% or discharge below 0%
        
        new_soc = np.clip(new_soc, 0.0, 1.0)
        
        # Calculate ACTUAL energy flow (in case we hit a limit)
        actual_energy_change = (new_soc - self.battery_soc) * self.BATTERY_CAPACITY_KWH
        actual_power_kw = actual_energy_change / 0.25
        
        self.battery_soc = new_soc
        
        # 3. Calculate Grid Interaction
        # Net Load = House_Load - Solar_Panel + Battery_Action
        # Positive = We must BUY from grid. Negative = We have EXCESS.
        current_data = self.df.iloc[self.current_step]
        net_grid_load = current_data['load_kw'] - current_data['solar_kw'] + actual_power_kw
        
        # 4. Calculate Cost (Reward)
        # If Net Load > 0: We buy from grid at current price
        # If Net Load < 0: We assume we give it to grid for free (or small feed-in tariff)
        # For this project: We only pay. We want to minimize Payment.
        
        grid_cost = 0.0
        if net_grid_load > 0:
            grid_cost = net_grid_load * current_data['price_per_kwh'] * 0.25 # Price is per kWh
            
        # REWARD FUNCTION (Crucial!)
        # RL maximizes Reward. We want to minimize Cost.
        # So Reward = -Cost.
        # We add a small penalty for using the battery aggressively to prevent jitter.
        reward = -grid_cost - (0.01 * (action[0]**2))
        
        # 5. Advance Time
        self.current_step += 1
        terminated = False
        truncated = False
        
        # End episode after 24 hours (96 steps) or if data runs out
        if self.current_step >= self.max_steps - 1 or (self.current_step % 96 == 0):
            terminated = True
            
        observation = self._get_observation()
        info = {"cost": grid_cost, "battery_soc": self.battery_soc}
        
        return observation, reward, terminated, truncated, info