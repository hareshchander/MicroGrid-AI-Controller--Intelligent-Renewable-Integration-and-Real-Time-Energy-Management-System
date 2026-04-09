import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

class SolarForecaster:
    def __init__(self, window_size=24):
        self.window_size = window_size
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """Trains the model."""
        # Flatten X_train for sklearn
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_flat, y_train)
        return self.model

    def predict(self, X):
        """Makes predictions for a given input sequence."""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)

    def evaluate(self, y_true, y_pred):
        """Evaluates the model performance."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse

if __name__ == "__main__":
    from data_preprocessing import MicroGridDataProcessor
    
    # Simple test run
    processor = MicroGridDataProcessor()
    df = processor.generate_synthetic_data(days=10)
    X, y, scaler = processor.prepare_forecasting_data(df, 'solar_gen')
    
    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    forecaster = SolarForecaster()
    forecaster.train(X_train, y_train, epochs=5)
    preds = forecaster.predict(X_test)
    
    mae, rmse = forecaster.evaluate(y_test, preds)
    print(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")
