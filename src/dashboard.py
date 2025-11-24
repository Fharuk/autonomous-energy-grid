import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from env import GridEnvironment

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Energy Grid", layout="wide", page_icon="⚡")

st.title("⚡ AI Autonomous Grid Manager")
st.markdown("""
### Real-Time Optimization: Deep Reinforcement Learning
This dashboard compares the **AI Agent's performance** against a traditional grid (No Battery Storage).
""")

# --- PATH SETUP (Windows Safe) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'processed', 'grid_data_clean.csv')
model_path = os.path.join(current_dir, '..', 'models', 'ppo_grid_agent.zip')

# --- LOAD RESOURCES ---
# We cache the model loading because it's heavy.
# We DO NOT cache the environment because we want a fresh simulation every time.
@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return PPO.load(model_path), "✅ AI Model Loaded"
    else:
        return None, "⚠️ Model Not Found (Running Random Actions)"

model, status = load_model()

# Sidebar Controls
st.sidebar.header("Simulation Settings")
st.sidebar.info(status)
days_to_sim = st.sidebar.slider("Days to Simulate", 1, 30, 7)
steps_to_sim = days_to_sim * 96  # 15-min intervals * 4 * 24h

# --- MAIN SIMULATION BLOCK ---
if st.button("▶️ Run Simulation", type="primary"):
    
    # 1. Initialize Fresh Environment
    env = GridEnvironment(data_path=data_path)
    obs, _ = env.reset()
    
    # Storage for plotting
    results = {
        "time_step": [],
        "load_kw": [],
        "solar_kw": [],
        "battery_soc": [],
        "grid_cost_ai": [],
        "price": []
    }
    
    progress_bar = st.progress(0)
    
    # 2. Run the Loop
    for i in range(steps_to_sim):
        # AI Decision
        if model:
            action, _ = model.predict(obs)
        else:
            action = env.action_space.sample() # Random fallback
            
        # Physics Step
        obs, reward, done, _, info = env.step(action)
        
        # Store Data (Get raw data from DataFrame for visualization)
        current_data = env.df.iloc[env.current_step]
        
        results["time_step"].append(i)
        results["load_kw"].append(current_data['load_kw'])
        results["solar_kw"].append(current_data['solar_kw'])
        results["battery_soc"].append(info['battery_soc'] * 100) # Convert to %
        results["grid_cost_ai"].append(info['cost']) # The cost the AI incurred
        results["price"].append(current_data['price_per_kwh'])
        
        progress_bar.progress((i + 1) / steps_to_sim)
        
        if done:
            break

    # --- 3. CALCULATE METRICS (The "Money Shot") ---
    df_res = pd.DataFrame(results)
    
    # Calculate Baseline: What if we had NO battery?
    # Logic: Net_Load = Load - Solar. 
    # If Net_Load > 0, we pay price. If Net_Load < 0, we pay 0.
    df_res['net_load_no_batt'] = df_res['load_kw'] - df_res['solar_kw']
    df_res['cost_no_batt'] = df_res['net_load_no_batt'].apply(lambda x: max(0, x)) * df_res['price'] * 0.25
    
    total_cost_ai = df_res['grid_cost_ai'].sum()
    total_cost_no_ai = df_res['cost_no_batt'].sum()
    
    savings = total_cost_no_ai - total_cost_ai
    savings_pct = (savings / total_cost_no_ai) * 100 if total_cost_no_ai > 0 else 0
    
    # --- DISPLAY METRICS ---
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    col1.metric("🚫 Traditional Cost (No Battery)", f"${total_cost_no_ai:,.2f}")
    col2.metric("🤖 AI Managed Cost", f"${total_cost_ai:,.2f}")
    col3.metric("💰 Net Savings", f"${savings:,.2f}", f"{savings_pct:.1f}%")
    
    if savings > 0:
        st.success(f"**Success:** The AI successfully utilized the battery to reduce costs by {savings_pct:.1f}%.")
    else:
        st.error(f"**Underperformance:** The AI lost money (${abs(savings):.2f}). This is common in early training. It needs more 'Timesteps' to learn arbitrage.")

    # --- PLOTTING ---
    st.subheader("🔋 Grid Activity & Battery Response")
    
    # Plot 1: Power Flow
    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    ax1.plot(df_res['load_kw'], label='House Load (kW)', color='blue', alpha=0.3)
    ax1.plot(df_res['solar_kw'], label='Solar Gen (kW)', color='orange', alpha=0.5)
    ax1.set_ylabel("Power (kW)")
    ax1.set_xlabel("Time Steps (15-min intervals)")
    ax1.legend(loc='upper left')
    
    # Create a second y-axis for Battery %
    ax2 = ax1.twinx()
    ax2.plot(df_res['battery_soc'], label='Battery Level (%)', color='green', linewidth=2.5)
    ax2.set_ylabel("Battery Charge (%)")
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right')
    
    st.pyplot(fig)
    
    # Plot 2: Cost Analysis
    st.subheader("💵 Cost Analysis: Avoiding the Peak")
    fig2, ax3 = plt.subplots(figsize=(15, 4))
    
    # Plot Price as a filled area (The Danger Zone)
    ax3.fill_between(range(len(df_res)), df_res['price'], color='red', alpha=0.1, label='Energy Price ($)')
    ax3.set_ylabel("Price ($/kWh)", color='red')
    
    # Plot Cost Accumulation
    ax4 = ax3.twinx()
    ax4.bar(range(len(df_res)), df_res['grid_cost_ai'], color='black', alpha=0.3, label='AI Cost Incurred')
    ax4.set_ylabel("Actual Cost ($)")
    
    st.pyplot(fig2)