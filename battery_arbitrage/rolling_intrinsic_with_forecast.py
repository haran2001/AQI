"""
Enhanced Battery Arbitrage with XGBoost Price Forecasting
Combines Rolling Intrinsic strategy with ML-based price predictions

This implementation replaces perfect foresight with XGBoost forecasts,
making the strategy more realistic for real-world deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the original classes
from rolling_internsic_battery_arbitrage import BatteryConfig, RollingIntrinsic


class PriceForecaster:
    """Wrapper for XGBoost price forecasting model"""

    def __init__(self, model_path: str = 'models/xgboost_price_model.pkl'):
        """Load the pre-trained XGBoost model"""
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.lookback_steps = None

        self.load_model()

    def load_model(self):
        """Load the trained XGBoost model and configuration"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Train the model first.")

        print(f"Loading XGBoost model from: {self.model_path}")
        self.model_data = joblib.load(self.model_path)

        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_columns = self.model_data['feature_columns']
        self.lookback_steps = self.model_data.get('lookback_steps', 24)
        self.target_column = self.model_data.get('target_column', 'price_mwh')

        print(f"Model loaded successfully. Features: {len(self.feature_columns)}, Lookback: {self.lookback_steps}")

    def prepare_features_for_forecast(self, historical_data: pd.DataFrame, current_idx: int):
        """
        Prepare feature vector for forecasting from historical data

        Args:
            historical_data: DataFrame with historical price and weather data
            current_idx: Current position index in the data

        Returns:
            Feature vector ready for prediction
        """
        # Get the necessary historical window
        lookback_start = max(0, current_idx - self.lookback_steps)
        historical_window = historical_data.iloc[lookback_start:current_idx + 1].copy()

        if len(historical_window) < self.lookback_steps:
            # Pad with earliest available data if not enough history
            padding_needed = self.lookback_steps - len(historical_window)
            padding = pd.concat([historical_window.iloc[[0]]] * padding_needed)
            historical_window = pd.concat([padding, historical_window])

        # Create features similar to training
        features = {}

        # Get the latest values
        latest_data = historical_window.iloc[-1]

        # Create lagged price features
        for lag in range(1, min(self.lookback_steps + 1, len(historical_window))):
            if lag <= len(historical_window):
                features[f'price_lag_{lag}'] = historical_window[self.target_column].iloc[-(lag+1)]

        # Create rolling statistics
        for window in [6, 12, 24]:
            if window <= len(historical_window):
                window_data = historical_window[self.target_column].tail(window)
                features[f'price_rolling_mean_{window}'] = window_data.mean()
                features[f'price_rolling_std_{window}'] = window_data.std()
                features[f'price_rolling_max_{window}'] = window_data.max()
                features[f'price_rolling_min_{window}'] = window_data.min()

        # Add weather features if available
        weather_cols = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                       'pressure_msl', 'cloud_cover']

        for col in weather_cols:
            if col in historical_window.columns:
                features[col] = latest_data[col]
                # Add lagged weather features
                for lag in [6, 12, 24]:
                    if lag <= len(historical_window):
                        features[f'{col}_lag_{lag}'] = historical_window[col].iloc[-(lag+1)]

        # Add temporal features
        current_timestamp = historical_window['timestamp'].iloc[-1]
        features['hour'] = current_timestamp.hour
        features['day_of_week'] = current_timestamp.dayofweek
        features['is_weekend'] = 1 if current_timestamp.dayofweek >= 5 else 0
        features['is_peak_hour'] = 1 if 6 <= current_timestamp.hour <= 22 else 0

        # Temporal encodings
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        # Price change features
        if len(historical_window) > 1:
            features['price_change_1'] = historical_window[self.target_column].diff(1).iloc[-1]
        if len(historical_window) > 6:
            features['price_change_6'] = historical_window[self.target_column].diff(6).iloc[-1]
        if len(historical_window) > 12:
            features['price_change_12'] = historical_window[self.target_column].diff(12).iloc[-1]

        # Additional binary features
        if 'price_negative' in self.feature_columns:
            features['price_negative'] = 1 if latest_data[self.target_column] < 0 else 0
        if 'price_high' in self.feature_columns:
            features['price_high'] = 1 if latest_data[self.target_column] > 100 else 0

        # Weather-price interactions
        if 'temperature_2m' in features and 'temp_hour_interaction' in self.feature_columns:
            features['temp_hour_interaction'] = features['temperature_2m'] * features['hour']
        if 'temperature_2m' in features and 'temp_weekend_interaction' in self.feature_columns:
            features['temp_weekend_interaction'] = features['temperature_2m'] * features['is_weekend']

        # Convert to DataFrame with proper column order
        feature_df = pd.DataFrame([features])

        # Ensure all required features are present (fill missing with 0)
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0

        # Select only the features used in training
        feature_df = feature_df[self.feature_columns]

        return feature_df

    def forecast_prices(self, historical_data: pd.DataFrame, current_idx: int,
                       horizon: int = 12) -> np.ndarray:
        """
        Generate price forecasts for the next horizon steps

        Args:
            historical_data: DataFrame with historical data up to current time
            current_idx: Current position in the data
            horizon: Number of 5-minute steps to forecast

        Returns:
            Array of forecasted prices
        """
        forecasts = []

        # Start with actual historical data
        working_data = historical_data.iloc[:current_idx + 1].copy()

        for step in range(horizon):
            # Prepare features for current state
            features = self.prepare_features_for_forecast(working_data, len(working_data) - 1)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            forecasts.append(prediction)

            # Add prediction to working data for next iteration
            next_timestamp = working_data['timestamp'].iloc[-1] + pd.Timedelta(minutes=5)

            # Create new row with prediction and carry forward weather data
            new_row = working_data.iloc[-1:].copy()
            new_row['timestamp'] = next_timestamp
            new_row[self.target_column] = prediction

            # Update temporal features
            new_row['hour'] = next_timestamp.hour
            new_row['day_of_week'] = next_timestamp.dayofweek
            new_row['is_weekend'] = 1 if next_timestamp.dayofweek >= 5 else 0
            new_row['is_peak_hour'] = 1 if 6 <= next_timestamp.hour <= 22 else 0

            working_data = pd.concat([working_data, new_row], ignore_index=True)

        return np.array(forecasts)


class RollingIntrinsicWithForecast(RollingIntrinsic):
    """
    Enhanced Rolling Intrinsic that uses XGBoost price forecasts
    instead of perfect foresight
    """

    def __init__(self, config: BatteryConfig, forecaster: PriceForecaster):
        """
        Initialize with battery config and price forecaster

        Args:
            config: Battery configuration
            forecaster: Trained price forecasting model
        """
        super().__init__(config)
        self.forecaster = forecaster
        self.forecast_accuracy_log = []

    def solve_dp_with_forecast(self, historical_data: pd.DataFrame,
                               current_idx: int, initial_soc: float,
                               horizon: int = 12) -> Tuple[List[float], List[float], np.ndarray]:
        """
        Solve DP using forecasted prices instead of actual future prices

        Args:
            historical_data: Historical price and weather data
            current_idx: Current position in the data
            initial_soc: Initial state of charge
            horizon: Planning horizon (number of 5-minute steps)

        Returns:
            actions: Optimal actions for each time step
            soc_trajectory: State of charge trajectory
            forecasted_prices: The prices that were forecasted
        """
        # Generate price forecast
        forecasted_prices = self.forecaster.forecast_prices(
            historical_data, current_idx, horizon
        )

        # Use the parent class DP solver with forecasted prices
        actions, soc_trajectory = self.solve_dp(
            forecasted_prices, initial_soc, horizon
        )

        return actions, soc_trajectory, forecasted_prices


class EnhancedBatteryBacktest:
    """
    Backtesting framework that compares perfect foresight vs forecast-based strategies
    """

    def __init__(self, battery_config: BatteryConfig):
        self.battery_config = battery_config
        self.forecaster = PriceForecaster()
        self.strategy_perfect = RollingIntrinsic(battery_config)
        self.strategy_forecast = RollingIntrinsicWithForecast(battery_config, self.forecaster)

    def run_comparison_backtest(self, data: pd.DataFrame,
                               rolling_window: int = 12,
                               reoptimize_freq: int = 1,
                               compare_strategies: bool = True) -> Dict:
        """
        Run backtest comparing perfect foresight vs forecast-based strategies

        Args:
            data: Historical data with prices and weather
            rolling_window: Planning horizon
            reoptimize_freq: How often to reoptimize
            compare_strategies: Whether to run both strategies for comparison

        Returns:
            Dictionary with results for both strategies
        """
        results = {}

        # Ensure price column name matches
        if 'rt_price_kwh' in data.columns and 'price_mwh' not in data.columns:
            data['price_mwh'] = data['rt_price_kwh'] * 1000  # Convert to $/MWh

        print("=" * 70)
        print("Running Enhanced Battery Arbitrage Backtest")
        print("=" * 70)

        # Run forecast-based strategy
        print("\n1. Running FORECAST-BASED Strategy...")
        results['forecast'] = self._run_single_strategy(
            data, self.strategy_forecast, rolling_window,
            reoptimize_freq, use_forecast=True
        )

        if compare_strategies:
            # Run perfect foresight strategy
            print("\n2. Running PERFECT FORESIGHT Strategy (Benchmark)...")
            results['perfect'] = self._run_single_strategy(
                data, self.strategy_perfect, rolling_window,
                reoptimize_freq, use_forecast=False
            )

            # Compare results
            self._print_comparison(results)

        return results

    def _run_single_strategy(self, data: pd.DataFrame, strategy,
                           rolling_window: int, reoptimize_freq: int,
                           use_forecast: bool) -> pd.DataFrame:
        """Run a single strategy backtest"""

        results = []
        current_soc = self.battery_config.capacity_kwh / 2  # Start at 50%
        forecast_errors = []

        # Determine start index (need enough history for forecasting)
        start_idx = max(strategy.forecaster.lookback_steps if use_forecast else 0, 100)

        for i in range(start_idx, len(data) - rolling_window, reoptimize_freq):

            if use_forecast:
                # Use forecast-based strategy
                actions, soc_trajectory, forecasted_prices = strategy.solve_dp_with_forecast(
                    data, i, current_soc, rolling_window
                )

                # Log forecast accuracy
                actual_prices = data['price_mwh'].iloc[i+1:i+1+rolling_window].values
                if len(actual_prices) == len(forecasted_prices):
                    forecast_error = np.mean(np.abs(actual_prices - forecasted_prices))
                    forecast_errors.append(forecast_error)
            else:
                # Use perfect foresight
                if 'price_mwh' in data.columns:
                    price_window = data['price_mwh'].iloc[i:i+rolling_window].values
                else:
                    price_window = data['rt_price_kwh'].iloc[i:i+rolling_window].values * 1000

                actions, soc_trajectory = strategy.solve_dp(
                    price_window, current_soc, rolling_window
                )

            # Execute actions for reoptimize_freq steps
            for j in range(min(reoptimize_freq, len(actions))):
                if i + j >= len(data):
                    break

                action = actions[j]
                if 'price_mwh' in data.columns:
                    price = data['price_mwh'].iloc[i + j] / 1000  # Convert to $/kWh
                else:
                    price = data['rt_price_kwh'].iloc[i + j]

                # Calculate revenue
                if action > 0:  # Charging
                    energy_kwh = action * strategy.dt
                    revenue = -(price * energy_kwh * (1 + strategy.total_cost))
                elif action < 0:  # Discharging
                    energy_kwh = -action * strategy.dt
                    revenue = price * energy_kwh * (1 - strategy.total_cost)
                else:
                    revenue = 0
                    energy_kwh = 0

                new_soc = strategy.update_soc(current_soc, action)

                results.append({
                    'timestamp': data['timestamp'].iloc[i + j] if 'timestamp' in data.columns else data.index[i + j],
                    'price': price,
                    'action_kw': action,
                    'energy_kwh': energy_kwh,
                    'soc_before': current_soc,
                    'soc_after': new_soc,
                    'revenue': revenue,
                    'strategy': 'forecast' if use_forecast else 'perfect'
                })

                current_soc = new_soc

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df['cumulative_revenue'] = results_df['revenue'].cumsum()

            # Add forecast accuracy if applicable
            if use_forecast and forecast_errors:
                avg_forecast_error = np.mean(forecast_errors)
                print(f"   Average Forecast Error: ${avg_forecast_error:.2f}/MWh")

        return results_df

    def _print_comparison(self, results: Dict):
        """Print comparison between strategies"""

        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON RESULTS")
        print("=" * 70)

        metrics_comparison = {}

        for strategy_name, df in results.items():
            if df is not None and not df.empty:
                metrics = {
                    'Total Revenue': df['revenue'].sum(),
                    'Avg Daily Revenue': df.groupby(df['timestamp'].dt.date)['revenue'].sum().mean(),
                    'Total Charged (kWh)': df[df['energy_kwh'] > 0]['energy_kwh'].sum(),
                    'Total Discharged (kWh)': -df[df['energy_kwh'] < 0]['energy_kwh'].sum(),
                    'Num Actions': len(df[df['action_kw'] != 0]),
                    'Avg SoC (kWh)': df['soc_after'].mean()
                }
                metrics_comparison[strategy_name] = metrics

        # Create comparison table
        if 'perfect' in metrics_comparison and 'forecast' in metrics_comparison:
            print(f"\n{'Metric':<25} {'Perfect Foresight':>20} {'Forecast-Based':>20} {'Difference':>15}")
            print("-" * 80)

            for metric in metrics_comparison['perfect'].keys():
                perfect_val = metrics_comparison['perfect'][metric]
                forecast_val = metrics_comparison['forecast'][metric]
                diff = forecast_val - perfect_val
                diff_pct = (diff / perfect_val * 100) if perfect_val != 0 else 0

                if 'Revenue' in metric:
                    print(f"{metric:<25} ${perfect_val:>18.2f} ${forecast_val:>18.2f} ${diff:>13.2f} ({diff_pct:+.1f}%)")
                elif 'kWh' in metric:
                    print(f"{metric:<25} {perfect_val:>18.2f} {forecast_val:>18.2f} {diff:>13.2f} ({diff_pct:+.1f}%)")
                else:
                    print(f"{metric:<25} {perfect_val:>18.0f} {forecast_val:>18.0f} {diff:>13.0f} ({diff_pct:+.1f}%)")

            # Calculate efficiency ratio
            forecast_revenue = metrics_comparison['forecast']['Total Revenue']
            perfect_revenue = metrics_comparison['perfect']['Total Revenue']
            efficiency_ratio = (forecast_revenue / perfect_revenue * 100) if perfect_revenue > 0 else 0

            print("\n" + "=" * 70)
            print(f"Forecast Strategy Efficiency: {efficiency_ratio:.1f}% of Perfect Foresight")
            print(f"Revenue Loss from Forecast Errors: ${perfect_revenue - forecast_revenue:.2f}")
            print("=" * 70)


def main():
    """Example usage of enhanced battery arbitrage with forecasting"""

    # Battery configuration
    battery_config = BatteryConfig(
        capacity_kwh=500.0,
        max_power_kw=100.0,
        efficiency_charge=0.95,
        efficiency_discharge=0.95,
        degradation_cost_per_kwh=0.004,
        trading_cost_per_kwh=0.00009
    )

    # Load data
    print("Loading historical data...")

    # Try to load the merged dataset with weather features
    data_path = 'data/merged_weather_prices_2025-08-01_2025-08-30.csv'
    if not Path(data_path).exists():
        # Fall back to simple price data
        data_path = 'casio_data.txt'
        print(f"Merged data not found. Using: {data_path}")

    data = pd.read_csv(data_path)
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])

    print(f"Loaded {len(data)} data points")

    # Run enhanced backtest
    backtester = EnhancedBatteryBacktest(battery_config)

    results = backtester.run_comparison_backtest(
        data,
        rolling_window=12,    # 1 hour horizon
        reoptimize_freq=1,    # Reoptimize every 5 minutes
        compare_strategies=True  # Compare with perfect foresight
    )

    # Save results
    for strategy_name, df in results.items():
        if df is not None and not df.empty:
            filename = f'arbitrage_results_{strategy_name}.csv'
            df.to_csv(filename, index=False)
            print(f"\n{strategy_name.capitalize()} strategy results saved to: {filename}")


if __name__ == "__main__":
    main()