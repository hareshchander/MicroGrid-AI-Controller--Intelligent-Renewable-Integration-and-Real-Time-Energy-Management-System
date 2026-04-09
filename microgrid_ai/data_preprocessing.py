import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

class MicroGridDataProcessor:
    def __init__(self):
        self.solar_scaler = MinMaxScaler()
        self.load_scaler = MinMaxScaler()
        self.weather_scaler = MinMaxScaler()

    def generate_synthetic_data(self, days=30):
        """Generates synthetic solar, load, and weather data for testing."""
        date_rng = pd.date_range(start='2024-01-01', periods=days*24, freq='h')
        df = pd.DataFrame(date_rng, columns=['timestamp'])
        
        # Synthetic Solar Generation (Sinusoidal with day/night)
        df['hour'] = df['timestamp'].dt.hour
        df['solar_gen'] = np.maximum(0, np.sin((df['hour'] - 6) * np.pi / 12) * 500) + np.random.normal(0, 20, len(df))
        df['solar_gen'] = df['solar_gen'].clip(lower=0)
        
        # Synthetic Wind Generation (Variable, higher at certain hours)
        df['wind_gen'] = np.maximum(0, 200 * np.exp(-((df['hour'] - 12)**2) / 50) + np.random.normal(0, 30, len(df)))
        df['wind_gen'] = df['wind_gen'].clip(lower=0)
        
        # Synthetic Load Consumption (Two peaks: morning and evening)
        df['load_demand'] = 100 + 50 * np.sin((df['hour'] - 7) * np.pi / 12)**2 + \
                            150 * np.sin((df['hour'] - 18) * np.pi / 12)**2 + \
                            np.random.normal(0, 15, len(df))
        df['load_demand'] = df['load_demand'].clip(lower=50)

        # Synthetic Weather (Temperature and Irradiance)
        df['temperature'] = 20 + 10 * np.sin((df['hour'] - 10) * np.pi / 24) + np.random.normal(0, 2, len(df))
        df['irradiance'] = df['solar_gen'] * 2 # Simplified relationship

        return df

    def prepare_forecasting_data(self, df, target_col, window_size=24):
        """Prepares sliding window data for LSTM forecasting."""
        data = df[target_col].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - window_size):
            X.append(scaled_data[i:i + window_size])
            y.append(scaled_data[i + window_size])
            
        return np.array(X), np.array(y), scaler

    def align_datasets(self, solar_df, load_df, weather_df):
        """Aligns multiple datasets by timestamp."""
        # Ensure all 'timestamp' columns are datetime
        for df in [solar_df, load_df, weather_df]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        combined_df = pd.merge(solar_df, load_df, on='timestamp', how='inner')
        combined_df = pd.merge(combined_df, weather_df, on='timestamp', how='inner')
        return combined_df

if __name__ == "__main__":
    processor = MicroGridDataProcessor()
    df = processor.generate_synthetic_data(days=5)
    print("Synthetic Data Head:\n", df.head())
