#!/usr/bin/env python3
"""
XGBoost Price Forecasting Model

Trains an XGBoost model to forecast CAISO electricity prices using weather and temporal features.
Supports n-step ahead forecasting for battery arbitrage decision making.

Usage:
    python xgboost_price_forecaster.py
    python xgboost_price_forecaster.py --forecast-steps 12  # 1 hour ahead (12 x 5min)
    python xgboost_price_forecaster.py --forecast-steps 144 # 12 hours ahead
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PriceForecastModel:
    def __init__(self, forecast_steps=12, lookback_steps=24):
        """
        Initialize the price forecasting model.

        Args:
            forecast_steps (int): Number of 5-minute steps to forecast ahead
            lookback_steps (int): Number of past steps to use as features
        """
        self.forecast_steps = forecast_steps
        self.lookback_steps = lookback_steps
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'price_mwh'

    def create_lagged_features(self, df):
        """Create lagged features for time series forecasting."""

        print(f"Creating lagged features (lookback: {self.lookback_steps} steps)...")

        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp').copy()

        # Create lagged price features
        for lag in range(1, self.lookback_steps + 1):
            df[f'price_lag_{lag}'] = df[self.target_column].shift(lag)

        # Create rolling statistics for prices
        for window in [6, 12, 24]:  # 30min, 1hr, 2hr windows
            df[f'price_rolling_mean_{window}'] = df[self.target_column].rolling(window=window).mean()
            df[f'price_rolling_std_{window}'] = df[self.target_column].rolling(window=window).std()
            df[f'price_rolling_max_{window}'] = df[self.target_column].rolling(window=window).max()
            df[f'price_rolling_min_{window}'] = df[self.target_column].rolling(window=window).min()

        # Create lagged weather features (less frequent, every 6 steps = 30 min)
        weather_cols = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                       'pressure_msl', 'cloud_cover']

        for col in weather_cols:
            if col in df.columns:
                for lag in [6, 12, 24]:  # 30min, 1hr, 2hr lags
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        # Create temporal interaction features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Price change features
        df['price_change_1'] = df[self.target_column].diff(1)
        df['price_change_6'] = df[self.target_column].diff(6)
        df['price_change_12'] = df[self.target_column].diff(12)

        # Weather-price interaction features
        if 'temperature_2m' in df.columns:
            df['temp_hour_interaction'] = df['temperature_2m'] * df['hour']
            df['temp_weekend_interaction'] = df['temperature_2m'] * df['is_weekend'].astype(int)

        return df

    def prepare_features(self, df):
        """Prepare feature matrix and target vector."""

        # Create lagged features
        df_features = self.create_lagged_features(df)

        # Select feature columns (exclude target and metadata)
        exclude_cols = ['timestamp', self.target_column]
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]

        # Remove rows with NaN values (due to lagging)
        df_clean = df_features.dropna()

        print(f"Features created: {len(feature_cols)}")
        print(f"Samples after cleaning: {len(df_clean)} (removed {len(df_features) - len(df_clean)} rows with NaN)")

        X = df_clean[feature_cols]
        y = df_clean[self.target_column]
        timestamps = df_clean['timestamp']

        self.feature_columns = feature_cols

        return X, y, timestamps

    def train(self, df, test_size=0.2):
        """Train the XGBoost model with time series split."""

        print("=" * 60)
        print("XGBoost Price Forecasting Model Training")
        print("=" * 60)

        # Prepare features
        X, y, timestamps = self.prepare_features(df)

        # Time series split (respect temporal order)
        split_idx = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        train_timestamps = timestamps.iloc[:split_idx]
        test_timestamps = timestamps.iloc[split_idx:]

        print(f"Training set: {len(X_train)} samples ({train_timestamps.min()} to {train_timestamps.max()})")
        print(f"Test set: {len(X_test)} samples ({test_timestamps.min()} to {test_timestamps.max()})")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # XGBoost model with time series optimized parameters
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )

        print("\nTraining XGBoost model...")
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )

        # Predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        # Evaluation metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print(f"\n=== Model Performance ===")
        print(f"Train RMSE: ${train_rmse:.2f}/MWh")
        print(f"Test RMSE:  ${test_rmse:.2f}/MWh")
        print(f"Train MAE:  ${train_mae:.2f}/MWh")
        print(f"Test MAE:   ${test_mae:.2f}/MWh")
        print(f"Train R²:   {train_r2:.3f}")
        print(f"Test R²:    {test_r2:.3f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n=== Top 10 Most Important Features ===")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")

        # Store results for analysis
        self.train_results = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'train_pred': train_pred, 'test_pred': test_pred,
            'train_timestamps': train_timestamps, 'test_timestamps': test_timestamps,
            'feature_importance': feature_importance,
            'metrics': {
                'train_rmse': train_rmse, 'test_rmse': test_rmse,
                'train_mae': train_mae, 'test_mae': test_mae,
                'train_r2': train_r2, 'test_r2': test_r2
            }
        }

        return self.train_results

    def forecast_n_steps(self, df, n_steps=None):
        """
        Make n-step ahead forecasts.

        Args:
            df: DataFrame with features up to current time
            n_steps: Number of steps to forecast (default: self.forecast_steps)

        Returns:
            forecasts: Array of predicted prices
            timestamps: Array of forecast timestamps
        """

        if n_steps is None:
            n_steps = self.forecast_steps

        if self.model is None:
            raise ValueError("Model must be trained first. Call train() method.")

        print(f"\nGenerating {n_steps}-step ahead forecast...")

        # Get the latest data point
        df_sorted = df.sort_values('timestamp')
        latest_timestamp = df_sorted['timestamp'].iloc[-1]

        # Prepare features for the latest data
        X_latest, _, _ = self.prepare_features(df_sorted)
        X_latest_scaled = self.scaler.transform(X_latest.iloc[[-1]])  # Last row only

        forecasts = []
        forecast_timestamps = []

        # Start forecasting from latest point
        current_features = X_latest.iloc[-1:].copy()

        for step in range(n_steps):
            # Generate timestamp for this forecast
            forecast_time = latest_timestamp + pd.Timedelta(minutes=5 * (step + 1))
            forecast_timestamps.append(forecast_time)

            # Scale features and predict
            current_scaled = self.scaler.transform(current_features)
            prediction = self.model.predict(current_scaled)[0]
            forecasts.append(prediction)

            # Update features for next prediction (simplified approach)
            # In practice, you'd need to update lagged features with the new prediction
            # For now, we'll use the prediction as the new price_lag_1 and shift others

            if step < n_steps - 1:  # Don't update on last iteration
                # Create a copy for updating
                updated_features = current_features.copy()

                # Update price lag features
                for lag in range(min(self.lookback_steps, step + 1), 0, -1):
                    if f'price_lag_{lag}' in updated_features.columns:
                        if lag == 1:
                            updated_features[f'price_lag_{lag}'] = prediction
                        elif f'price_lag_{lag-1}' in updated_features.columns:
                            updated_features[f'price_lag_{lag}'] = updated_features[f'price_lag_{lag-1}'].iloc[0]

                # Update rolling statistics (simplified - just using recent prediction)
                for window in [6, 12, 24]:
                    if f'price_rolling_mean_{window}' in updated_features.columns:
                        # This is a simplification - in practice you'd maintain a rolling buffer
                        updated_features[f'price_rolling_mean_{window}'] = prediction

                # Update temporal features for next timestamp
                next_hour = forecast_time.hour
                next_dow = forecast_time.dayofweek

                if 'hour' in updated_features.columns:
                    updated_features['hour'] = next_hour
                if 'day_of_week' in updated_features.columns:
                    updated_features['day_of_week'] = next_dow
                if 'hour_sin' in updated_features.columns:
                    updated_features['hour_sin'] = np.sin(2 * np.pi * next_hour / 24)
                if 'hour_cos' in updated_features.columns:
                    updated_features['hour_cos'] = np.cos(2 * np.pi * next_hour / 24)
                if 'day_sin' in updated_features.columns:
                    updated_features['day_sin'] = np.sin(2 * np.pi * next_dow / 7)
                if 'day_cos' in updated_features.columns:
                    updated_features['day_cos'] = np.cos(2 * np.pi * next_dow / 7)

                current_features = updated_features

        forecasts = np.array(forecasts)
        forecast_timestamps = pd.to_datetime(forecast_timestamps)

        print(f"Forecast range: ${forecasts.min():.2f} to ${forecasts.max():.2f}/MWh")
        print(f"Forecast mean: ${forecasts.mean():.2f}/MWh")

        return forecasts, forecast_timestamps

    def save_model(self, filepath):
        """Save the trained model and scaler."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'forecast_steps': self.forecast_steps,
            'lookback_steps': self.lookback_steps,
            'target_column': self.target_column
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath):
        """Load a trained model and scaler."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.forecast_steps = model_data['forecast_steps']
        self.lookback_steps = model_data['lookback_steps']
        self.target_column = model_data['target_column']
        print(f"Model loaded from: {filepath}")

    def plot_results(self, save_plots=True):
        """Plot training results and forecasts."""

        if not hasattr(self, 'train_results'):
            print("No training results to plot. Train the model first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Training vs Test predictions
        axes[0, 0].scatter(self.train_results['y_train'], self.train_results['train_pred'],
                          alpha=0.5, label='Train', s=1)
        axes[0, 0].scatter(self.train_results['y_test'], self.train_results['test_pred'],
                          alpha=0.5, label='Test', s=1)
        axes[0, 0].plot([self.train_results['y_train'].min(), self.train_results['y_train'].max()],
                       [self.train_results['y_train'].min(), self.train_results['y_train'].max()],
                       'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price ($/MWh)')
        axes[0, 0].set_ylabel('Predicted Price ($/MWh)')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].legend()

        # 2. Time series plot of test predictions
        test_df = pd.DataFrame({
            'timestamp': self.train_results['test_timestamps'],
            'actual': self.train_results['y_test'],
            'predicted': self.train_results['test_pred']
        }).iloc[:288]  # First 24 hours of test set

        axes[0, 1].plot(test_df['timestamp'], test_df['actual'], label='Actual', linewidth=1)
        axes[0, 1].plot(test_df['timestamp'], test_df['predicted'], label='Predicted', linewidth=1)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Price ($/MWh)')
        axes[0, 1].set_title('Price Predictions Over Time (First 24h of Test)')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Feature importance
        top_features = self.train_results['feature_importance'].head(15)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'], fontsize=8)
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 15 Feature Importances')

        # 4. Residuals analysis
        test_residuals = self.train_results['y_test'] - self.train_results['test_pred']
        axes[1, 1].scatter(self.train_results['test_pred'], test_residuals, alpha=0.5, s=1)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Price ($/MWh)')
        axes[1, 1].set_ylabel('Residuals ($/MWh)')
        axes[1, 1].set_title('Residuals vs Predicted')

        plt.tight_layout()

        if save_plots:
            plt.savefig('data/xgboost_model_results.png', dpi=300, bbox_inches='tight')
            print("Plots saved to: data/xgboost_model_results.png")

        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost price forecasting model')
    parser.add_argument('--data-file', type=str,
                       default='data/merged_weather_prices_2025-08-01_2025-08-30.csv',
                       help='Path to merged dataset CSV file')
    parser.add_argument('--forecast-steps', type=int, default=12,
                       help='Number of 5-minute steps to forecast ahead (default: 12 = 1 hour)')
    parser.add_argument('--lookback-steps', type=int, default=24,
                       help='Number of past steps to use as features (default: 24 = 2 hours)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--save-model', type=str, default='models/xgboost_price_model.pkl',
                       help='Path to save trained model')

    args = parser.parse_args()

    # Create models directory
    Path('models').mkdir(exist_ok=True)

    # Load data
    print(f"Loading data from: {args.data_file}")
    df = pd.read_csv(args.data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Dataset: {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Initialize and train model
    model = PriceForecastModel(
        forecast_steps=args.forecast_steps,
        lookback_steps=args.lookback_steps
    )

    # Train model
    results = model.train(df, test_size=args.test_size)

    # Save model
    model.save_model(args.save_model)

    # Generate sample forecast
    print(f"\n=== Sample {args.forecast_steps}-Step Forecast ===")
    forecasts, forecast_times = model.forecast_n_steps(df, args.forecast_steps)

    forecast_df = pd.DataFrame({
        'timestamp': forecast_times,
        'predicted_price': forecasts
    })

    print(forecast_df.to_string(index=False))

    # Save forecast
    forecast_df.to_csv('data/sample_forecast.csv', index=False)
    print(f"\nSample forecast saved to: data/sample_forecast.csv")

    # Plot results
    model.plot_results(save_plots=True)

    print("\n" + "=" * 60)
    print("XGBoost Price Forecasting Model Complete!")
    print(f"Model saved to: {args.save_model}")
    print(f"Forecast horizon: {args.forecast_steps} steps ({args.forecast_steps * 5} minutes)")

if __name__ == "__main__":
    main()