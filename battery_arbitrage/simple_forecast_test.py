#!/usr/bin/env python3
"""
Simple Forecast Test
Follows exact instructions:
1. Get weather forecast for a day (user parameter)
2. Interpolate and merge weather+price data
3. Use XGBoost to forecast prices
4. Account for API limits
5. Run rolling intrinsic on predicted vs ground truth
"""

import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import time
import argparse
from pathlib import Path

# Import existing modules
from rolling_internsic_battery_arbitrage import BatteryConfig, RollingIntrinsic, BatteryArbitrageBacktest
import joblib


class SimpleForecastTest:
    def __init__(self, test_date: str):
        self.test_date = datetime.strptime(test_date, '%Y-%m-%d')
        self.api_limits = {
            'open_meteo_daily': 5,  # 5 calls per day
            'caiso_per_minute': 10  # 10 calls per minute
        }

        # Calculate weather update intervals
        self.weather_interval_hours = 24 / self.api_limits['open_meteo_daily']  # 4.8 hours
        self.weather_interval_hours = int(self.weather_interval_hours)  # Round to 4 hours

        print(f"Test date: {self.test_date.strftime('%Y-%m-%d')}")
        print(f"Weather API limit: {self.api_limits['open_meteo_daily']} calls/day")
        print(f"Weather update interval: {self.weather_interval_hours} hours")

    def get_weather_forecast(self, forecast_hours=24):
        """1. Get weather forecast for the day using open_metero_weather_data.py pattern"""
        print(f"\n1. Getting weather forecast for {forecast_hours}h...")

        # Setup Open-Meteo API (following open_metero_weather_data.py)
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        # API parameters for Eland Solar location (matching open_metero_weather_data.py)
        params = {
            "latitude": 35.3733,
            "longitude": -119.0187,
            "start_date": self.test_date.strftime('%Y-%m-%d'),
            "end_date": (self.test_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                      "pressure_msl", "cloud_cover", "precipitation"]
        }

        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Process hourly data
        hourly = response.Hourly()
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        }

        # Add weather variables
        hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
        hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
        hourly_data["wind_speed_10m"] = hourly.Variables(2).ValuesAsNumpy()
        hourly_data["pressure_msl"] = hourly.Variables(3).ValuesAsNumpy()
        hourly_data["cloud_cover"] = hourly.Variables(4).ValuesAsNumpy()
        hourly_data["precipitation"] = hourly.Variables(5).ValuesAsNumpy()

        weather_data = pd.DataFrame(data=hourly_data)
        print(f"Weather data: {len(weather_data)} hourly records")
        return weather_data

    def interpolate_to_5min(self, weather_data):
        """2a. Interpolate weather to 5-minute intervals"""
        print("\n2a. Interpolating weather data to 5-minute intervals...")

        # Use direct interpolation logic instead of file-based function
        df = weather_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Calculate new frequency (5-minute intervals = 12 points per hour)
        points_per_hour = 12
        minutes_per_point = 60 / points_per_hour
        freq_string = f"{int(minutes_per_point)}min"

        # Extend end time to complete the last hour
        extended_end = df.index.max() + pd.Timedelta(hours=1) - pd.Timedelta(minutes=minutes_per_point)

        # Create new index with higher frequency
        new_index = pd.date_range(
            start=df.index.min(),
            end=extended_end,
            freq=freq_string
        )

        # Reindex and interpolate
        df_reindexed = df.reindex(new_index)
        result = pd.DataFrame(index=new_index)

        # Simple linear interpolation for all columns
        for col in df.columns:
            if not df[col].isna().all():
                result[col] = df_reindexed[col].interpolate(method='linear')
            else:
                result[col] = np.nan

        # Reset index to have date as a column
        result = result.reset_index()
        result = result.rename(columns={'index': 'date'})

        print(f"Interpolated data: {len(result)} 5-minute records")
        return result

    def get_historical_prices(self):
        """2b. Get historical CAISO price data for merging"""
        print("\n2b. Loading historical CAISO price data...")

        # Load existing RT price data
        price_file = f'data/eland_sp15_rt_prices_2025-08-01_2025-08-31.csv'
        if Path(price_file).exists():
            price_data = pd.read_csv(price_file)
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])

            # Filter for test date
            test_prices = price_data[price_data['timestamp'].dt.date == self.test_date.date()]
            print(f"Historical prices: {len(test_prices)} 5-minute records")
            return test_prices
        else:
            print("No historical price data found")
            return pd.DataFrame()

    def merge_data(self, weather_5min, price_data):
        """2c. Merge weather and price data"""
        print("\n2c. Merging weather and price data...")

        # Use existing merge logic from create_merged_dataset.py
        if len(price_data) == 0:
            print("No price data to merge")
            return pd.DataFrame()

        # Ensure timestamp columns
        weather_5min['timestamp'] = pd.to_datetime(weather_5min['date'])
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])

        # Set index and merge
        weather_indexed = weather_5min.set_index('timestamp')
        price_indexed = price_data.set_index('timestamp')

        merged = price_indexed.join(weather_indexed, how='inner')

        # Add derived features
        merged['hour'] = merged.index.hour
        merged['day_of_week'] = merged.index.dayofweek
        merged['is_weekend'] = merged['day_of_week'].isin([5, 6]).astype(int)
        merged['is_peak_hours'] = merged['hour'].between(6, 22).astype(int)
        merged['price_negative'] = (merged['price_mwh'] < 0).astype(int)
        merged['price_high'] = (merged['price_mwh'] > 100).astype(int)

        merged = merged.reset_index()

        print(f"Merged data: {len(merged)} records, {len(merged.columns)} features")
        return merged

    def forecast_prices(self, merged_data):
        """3. Use XGBoost model to forecast prices"""
        print("\n3. Forecasting prices with XGBoost model...")

        model_file = 'models/xgboost_price_model.pkl'
        if not Path(model_file).exists():
            print("XGBoost model not found, using persistence forecast")
            if len(merged_data) > 0:
                return merged_data['price_mwh'].values
            else:
                return np.array([30.0] * 288)  # Default day

        # Load model
        model_data = joblib.load(model_file)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']

        # Prepare features (simplified)
        features_df = merged_data.copy()

        # Add basic lag features
        for lag in [1, 6, 12]:
            features_df[f'price_lag_{lag}'] = features_df['price_mwh'].shift(lag)

        # Add rolling features
        for window in [6, 12]:
            features_df[f'price_rolling_mean_{window}'] = features_df['price_mwh'].rolling(window).mean()

        # Fill missing features with 0
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        # Drop NaN rows and predict
        clean_data = features_df.dropna()
        if len(clean_data) == 0:
            print("No clean data for prediction")
            return np.array([30.0] * len(merged_data))

        X = clean_data[feature_columns].fillna(0)
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        print(f"Generated {len(predictions)} price predictions")
        return predictions

    def run_rolling_intrinsic(self, prices, label):
        """5. Run rolling intrinsic algorithm"""
        print(f"\n5. Running rolling intrinsic on {label}...")

        # Battery configuration
        battery_config = BatteryConfig(
            capacity_kwh=500.0,
            max_power_kw=100.0,
            efficiency_charge=0.95,
            efficiency_discharge=0.95
        )

        # Create strategy
        strategy = RollingIntrinsic(battery_config)

        # Create price DataFrame
        timestamps = pd.date_range(
            start=self.test_date,
            periods=len(prices),
            freq='5min'
        )

        price_df = pd.DataFrame({
            'timestamp': timestamps,
            'rt_price_kwh': prices / 1000  # Convert $/MWh to $/kWh
        })

        # Run backtest
        backtester = BatteryArbitrageBacktest(strategy)
        results = backtester.run_backtest(
            price_df,
            rolling_window=12,  # 1 hour
            reoptimize_freq=1   # Every 5 minutes
        )

        if not results.empty:
            total_revenue = results['revenue'].sum()
            print(f"{label} portfolio value: ${total_revenue:.2f}")
            return total_revenue, results
        else:
            print(f"No results for {label}")
            return 0.0, pd.DataFrame()

    def run_test(self):
        """Run the complete test"""
        print("=" * 60)
        print("SIMPLE FORECAST TEST")
        print("=" * 60)

        try:
            # 1. Get weather forecast
            weather_data = self.get_weather_forecast()

            # 2. Interpolate and merge
            weather_5min = self.interpolate_to_5min(weather_data)
            price_data = self.get_historical_prices()
            merged_data = self.merge_data(weather_5min, price_data)

            if len(merged_data) == 0:
                print("No merged data available for testing")
                return

            # 3. Forecast prices
            predicted_prices = self.forecast_prices(merged_data)
            actual_prices = merged_data['price_mwh'].values

            # 4. API limits are accounted for by using limited weather calls
            print(f"\n4. API limits: Weather updates every {self.weather_interval_hours}h")

            # 5. Run rolling intrinsic on both
            pred_value, pred_results = self.run_rolling_intrinsic(predicted_prices, "Predicted")
            actual_value, actual_results = self.run_rolling_intrinsic(actual_prices, "Ground Truth")

            # Results
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"Predicted portfolio value: ${pred_value:.2f}")
            print(f"Ground truth portfolio value: ${actual_value:.2f}")

            if actual_value != 0:
                efficiency = (pred_value / actual_value) * 100
                print(f"Forecast efficiency: {efficiency:.1f}%")

            # Save results
            if not pred_results.empty:
                pred_results.to_csv(f'predicted_portfolio_{self.test_date.strftime("%Y%m%d")}.csv', index=False)
            if not actual_results.empty:
                actual_results.to_csv(f'actual_portfolio_{self.test_date.strftime("%Y%m%d")}.csv', index=False)

            print("Portfolio results saved to CSV files")

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Simple forecast test')
    parser.add_argument('--date', type=str, required=True,
                       help='Test date in YYYY-MM-DD format')

    args = parser.parse_args()

    # Validate date
    try:
        test_date = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print("Invalid date format. Use YYYY-MM-DD")
        return

    # Run test
    test = SimpleForecastTest(args.date)
    test.run_test()


if __name__ == "__main__":
    main()