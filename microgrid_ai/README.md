# MicroGrid AI Controller: Intelligent Energy Management System

An AI-powered microgrid system that performs solar energy forecasting, integrates multiple energy sources (solar, battery, grid), and makes real-time intelligent energy management decisions through a dynamic simulation.

## 🚀 Features
- **Solar Forecasting:** Uses an LSTM (Long Short-Term Memory) neural network to predict future solar generation.
- **Dynamic Simulation:** Real-time time-stepped simulation of solar, load, and battery state.
- **Intelligent Controller:** Rule-based logic to minimize grid cost and maximize renewable usage.
- **Interactive Dashboard:** Streamlit-based UI for real-time visualization and scenario analysis.

## 📁 Project Structure
- `app.py`: Main Streamlit application and UI.
- `data_preprocessing.py`: Data generation, cleaning, and windowing for LSTM.
- `forecasting_model.py`: LSTM model definition and training logic.
- `microgrid_simulation.py`: Battery dynamics and environment simulation.
- `controller.py`: Energy management strategy and decision logic.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd microgrid_ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Recommended Datasets (Kaggle)
For research-grade results, replace the synthetic data generator with these datasets:
1. [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
2. [Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

## 💡 How it Works
1. **Forecasting:** The system looks at the previous 24 hours of solar data to predict the next hour's generation.
2. **Decision Making:** 
   - If `Solar > Load`: Excess energy charges the battery. If the battery is full, it exports to the grid.
   - If `Load > Solar`: The system first discharges the battery. If the battery is empty, it draws from the grid.
3. **Visualization:** Real-time charts show the energy balance, battery SOC, and forecasting accuracy.

## 🚀 Deployment
This app is ready for deployment on **Streamlit Cloud** or **Hugging Face Spaces**.
- For Streamlit Cloud: Connect your GitHub repo and point to `app.py`.
- Ensure `requirements.txt` is in the root directory.
