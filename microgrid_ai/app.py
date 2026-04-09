import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import modular components
from data_preprocessing import MicroGridDataProcessor
from forecasting_model import SolarForecaster
from microgrid_simulation import MicroGridSimulator
from controller import MicroGridController

# Set Streamlit Page Config
st.set_page_config(page_title="MicroGrid AI Controller: Intelligent Renewable Integration and Real-Time Energy Management System", layout="wide")

st.title("🔋 MicroGrid AI Controller: Intelligent Renewable Integration and Real-Time Energy Management System")

st.markdown("""
### Real-Time Energy Forecasting & Optimization System
This dashboard simulates a microgrid with solar and wind renewable generation, household load, and battery storage controlled by an AI-driven management system.

#### 🔌 Grid/Microgrid Integration Features
- **Inverter-Based Renewables**: Simulates grid-forming inverters for low-inertia systems
- **Intelligent Control**: AI optimization minimizes grid dependency and costs
- **Stability Analysis**: Test system response to various renewable penetration levels
""")

# Grid Integration Mode
st.sidebar.header("🔌 Grid Integration Mode")
grid_mode = st.sidebar.selectbox("Inverter Mode", ["Grid-Following", "Grid-Forming"], index=1)
st.sidebar.markdown("""
**Grid-Following**: Inverters inject current, follow grid voltage/frequency
**Grid-Forming**: Inverters act as voltage sources, provide inertia in low-inertia systems
""")

if grid_mode == "Grid-Forming":
    st.sidebar.success("✅ Grid-Forming mode active: Enhanced stability for high renewable penetration")
else:
    st.sidebar.warning("⚠️ Grid-Following mode: May require synchronous generators for stability")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Simulation Settings")
days_to_sim = st.sidebar.slider("Simulation Duration (Days)", 1, 30, 7)
battery_cap = st.sidebar.number_input("Battery Capacity (MWh)", 0.1, 4000.0, 20.0)
grid_price = st.sidebar.slider("Grid Electricity Price ($/kWh)", 0.05, 0.50, 0.15)

st.sidebar.header("⚙️ Renewable Capacity Settings (MW)")
solar_capacity = st.sidebar.slider("Solar Capacity (MW)", 0.1, 800.0, 1.0)
wind_capacity = st.sidebar.slider("Wind Capacity (MW)", 0.1, 800.0, 0.5)
load_demand = st.sidebar.slider("Load (MW)", 0.1, 1000.0, 0.5)

# --- Initialization ---
@st.cache_resource
def load_and_train():
    processor = MicroGridDataProcessor()
    df = processor.generate_synthetic_data(days=60) # Larger dataset for training
    
    X, y, scaler = processor.prepare_forecasting_data(df, 'solar_gen')
    forecaster = SolarForecaster()
    # Training for a few epochs for demo purposes
    forecaster.train(X, y, epochs=5)
    
    X_wind, y_wind, scaler_wind = processor.prepare_forecasting_data(df, 'wind_gen')
    forecaster_wind = SolarForecaster()
    forecaster_wind.train(X_wind, y_wind, epochs=5)
    
    return processor, forecaster, scaler, forecaster_wind, scaler_wind

processor, forecaster, scaler, forecaster_wind, scaler_wind = load_and_train()

# --- Simulation Execution ---
if st.button("🚀 Run Simulation"):
    # 1. Generate fresh data for simulation period
    sim_data = processor.generate_synthetic_data(days=days_to_sim)
    sim_data['solar_gen'] *= (solar_capacity / 0.5)  # Scale to MW capacity
    sim_data['wind_gen'] *= (wind_capacity / 0.25)   # Scale to MW capacity
    sim_data['load_demand'] *= (load_demand / 0.2)   # Scale to MW load
    
    # 2. Run Simulation Step-by-Step
    simulator = MicroGridSimulator(battery_capacity=battery_cap * 1000)
    controller = MicroGridController(cost_per_kwh=grid_price)
    
    # To simulate forecasting, we use the forecaster at each step
    # For speed in UI, we'll batch predict
    X_sim, _, _ = processor.prepare_forecasting_data(sim_data, 'solar_gen')
    # Note: prepare_forecasting_data truncates the first window_size elements
    
    results = []
    # Loop through the data points where we have enough history for forecasting
    window_size = 24
    for i in range(len(sim_data) - window_size):
        ts = sim_data.iloc[i + window_size]['timestamp']
        solar_actual = sim_data.iloc[i + window_size]['solar_gen']
        wind_actual = sim_data.iloc[i + window_size]['wind_gen']
        renewable_actual = solar_actual + wind_actual
        load_actual = sim_data.iloc[i + window_size]['load_demand']
        
        # Make a forecast for the current step (using previous 24h)
        input_seq = sim_data.iloc[i:i + window_size]['solar_gen'].values.reshape(1, window_size, 1)
        # Scaled prediction
        input_seq_scaled = scaler.transform(input_seq.reshape(-1, 1)).reshape(1, window_size, 1)
        solar_pred_scaled = forecaster.predict(input_seq_scaled)
        solar_pred = scaler.inverse_transform(solar_pred_scaled.reshape(-1, 1))[0][0]
        
        input_seq_wind = sim_data.iloc[i:i + window_size]['wind_gen'].values.reshape(1, window_size, 1)
        input_seq_wind_scaled = scaler_wind.transform(input_seq_wind.reshape(-1, 1)).reshape(1, window_size, 1)
        wind_pred_scaled = forecaster_wind.predict(input_seq_wind_scaled)
        wind_pred = scaler_wind.inverse_transform(wind_pred_scaled.reshape(-1, 1))[0][0]
        
        # Decide action based on current state (Actuals used for real-time control)
        decision = controller.decide_action(renewable_actual, load_actual, simulator.battery)
        
        # Record results
        results.append({
            'Timestamp': ts,
            'Solar Actual (kW)': solar_actual,
            'Wind Actual (kW)': wind_actual,
            'Renewable Actual (kW)': renewable_actual,
            'Solar Forecast (kW)': solar_pred,
            'Wind Forecast (kW)': wind_pred,
            'Load Demand (kW)': load_actual,
            'Battery Action (kW)': decision['battery_action_kw'],
            'Grid Draw (kW)': decision['grid_draw_kw'],
            'Grid Export (kW)': decision['grid_export_kw'],
            'Battery SOC (%)': decision['soc_after'],
            'Cost ($)': decision['current_cost']
        })

    results_df = pd.DataFrame(results)

    # --- Metrics Comparison ---
    grid_only_cost = (results_df['Load Demand (kW)'] * grid_price).sum()
    with_controller_cost = results_df['Cost ($)'].sum()
    savings = grid_only_cost - with_controller_cost
    savings_pct = (savings / grid_only_cost) * 100 if grid_only_cost > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Solar Gen", f"{results_df['Solar Actual (kW)'].sum():.2f} kWh")
    col2.metric("Total Grid Cost (AI)", f"${with_controller_cost:.2f}", delta=f"-${savings:.2f}")
    col3.metric("Grid Only Cost", f"${grid_only_cost:.2f}")
    col4.metric("Cost Savings", f"{savings_pct:.1f}%", delta_color="normal")

    # --- Visualizations ---
    
    # 1. Energy Flow Chart
    st.subheader("📊 Energy Balance & Flow")
    fig_flow = go.Figure()
    fig_flow.add_trace(go.Scatter(x=results_df['Timestamp'], y=results_df['Solar Actual (kW)'], name="Solar Generation", fill='tozeroy'))
    fig_flow.add_trace(go.Scatter(x=results_df['Timestamp'], y=results_df['Wind Actual (kW)'], name="Wind Generation", fill='tonexty'))
    fig_flow.add_trace(go.Scatter(x=results_df['Timestamp'], y=results_df['Load Demand (kW)'], name="Load Demand"))
    fig_flow.add_trace(go.Bar(x=results_df['Timestamp'], y=results_df['Grid Draw (kW)'], name="Grid Draw", marker_color='red'))
    fig_flow.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig_flow, use_container_width=True)

    # 2. Battery SOC & Forecasting Accuracy
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🔋 Battery State of Charge (SOC)")
        fig_soc = px.line(results_df, x='Timestamp', y='Battery SOC (%)', color_discrete_sequence=['#00FF00'])
        fig_soc.update_layout(template="plotly_dark")
        st.plotly_chart(fig_soc, use_container_width=True)
        
    with col_b:
        st.subheader("📈 Forecasting Performance")
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(x=results_df['Timestamp'], y=results_df['Solar Actual (kW)'], name="Actual"))
        fig_fore.add_trace(go.Scatter(x=results_df['Timestamp'], y=results_df['Solar Forecast (kW)'], name="Forecast", line=dict(dash='dash')))
        fig_fore.update_layout(template="plotly_dark")
        st.plotly_chart(fig_fore, use_container_width=True)

    # 3. Recommendations
    total_export = results_df['Grid Export (kW)'].sum()
    total_draw = results_df['Grid Draw (kW)'].sum()
    renewable_total = results_df['Renewable Actual (kW)'].sum()
    load_total = results_df['Load Demand (kW)'].sum()
    penetration = renewable_total / (renewable_total + load_total) * 100 if (renewable_total + load_total) > 0 else 0
    
    rec_text = ""
    if grid_mode == "Grid-Forming":
        rec_text += "🔌 **Grid-Forming Mode**: Inverter provides virtual inertia, suitable for high renewable penetration.\n"
        if penetration > 50:
            rec_text += "✅ High renewable penetration detected. Grid-forming inverters help maintain stability.\n"
    else:
        rec_text += "🔌 **Grid-Following Mode**: Inverters follow grid voltage/frequency and may need synchronous support in low-inertia systems.\n"
        if penetration > 30:
            rec_text += "⚠️ High renewable penetration may cause instability in grid-following mode.\n"
    
    if total_export > 100:
        rec_text += f"💡 **AI Recommendation**: Increase battery capacity to capture more excess renewable energy ({total_export:.1f} kWh exported). This improves stability and reduces waste.\n"
    elif total_draw > renewable_total * 0.5 and renewable_total < load_total:
        rec_text += f"⚠️ Renewable output is low relative to load. Grid draw is high ({total_draw:.1f} kWh). Consider increasing renewable capacity or storage.\n"
    else:
        rec_text += "✅ System is reasonably balanced for the current generation and load.\n"
    
    if total_export > 100 and total_draw > 100:
        rec_text += "⚠️ The system is exporting excess energy while also drawing from the grid. Check battery sizing and generation mix."
    elif total_draw > 1000:
        rec_text += "⚠️ High grid draw indicates load is much larger than renewable supply. Consider adding more renewables or storage."
    
    st.info(rec_text)

    # 4. 3D Energy Flow Simulation
    st.subheader("🌐 3D Energy Flow Simulation")
    results_df['Hours'] = (results_df['Timestamp'] - results_df['Timestamp'].min()).dt.total_seconds() / 3600
    
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(
        x=results_df['Hours'],
        y=results_df['Renewable Actual (kW)'],
        z=results_df['Battery SOC (%)'],
        mode='lines+markers',
        name='Energy Flow',
        line=dict(color='cyan', width=2),
        marker=dict(size=3, color='yellow')
    ))
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Time (Hours)',
            yaxis_title='Renewable Generation (kW)',
            zaxis_title='Battery SOC (%)'
        ),
        template="plotly_dark"
    )
    st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.warning("Click 'Run Simulation' to start the energy management process.")
