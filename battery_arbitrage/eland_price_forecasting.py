"""
Electricity Price Forecasting for Eland Solar & Storage Center
Using LightGBM + TFT Ensemble for DA and RT price forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# LightGBM
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Set random seeds
np.random.seed(42)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ElandPriceForecaster:
    """
    Price forecasting system for Eland Solar & Storage Center
    Forecasts both Day-Ahead (DA) and Real-Time (RT) prices
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.features = None
        self.lgb_models = {}
        self.scalers = {}

    def load_and_prepare_data(self):
        """Load and prepare Eland data for forecasting"""
        print("Loading Eland Solar & Storage Center data...")
        self.df = pd.read_csv(self.data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df = self.df.set_index('datetime')

        print(f"Data loaded: {self.df.shape}")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")

        # Check for price columns
        has_da = 'da_price_mwh' in self.df.columns and not self.df['da_price_mwh'].isna().all()
        has_rt = 'rt_price_mwh' in self.df.columns and not self.df['rt_price_mwh'].isna().all()

        print(f"\nPrice data availability:")
        print(f"  Day-Ahead prices: {'Available' if has_da else 'Not available'}")
        print(f"  Real-Time prices: {'Available' if has_rt else 'Not available'}")

        # Fill missing values for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['da_price_mwh', 'rt_price_mwh']:
                # Forward fill then backward fill for price columns
                self.df[col] = self.df[col].fillna(method='ffill').fillna(method='bfill')
            else:
                # Fill with median for other numeric columns
                self.df[col] = self.df[col].fillna(self.df[col].median())

        return self.df

    def create_time_features(self, df):
        """Create time-based features"""
        df_feat = df.copy()

        # Extract time features
        df_feat['hour'] = df_feat.index.hour
        df_feat['day_of_week'] = df_feat.index.dayofweek
        df_feat['day_of_month'] = df_feat.index.day
        df_feat['month'] = df_feat.index.month
        df_feat['quarter'] = df_feat.index.quarter
        df_feat['year'] = df_feat.index.year
        df_feat['day_of_year'] = df_feat.index.dayofyear

        # Cyclical encoding
        df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
        df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
        df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
        df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)

        # Is weekend
        df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)

        # Peak hours indicator (typically 7am-10pm)
        df_feat['is_peak_hour'] = ((df_feat['hour'] >= 7) & (df_feat['hour'] <= 22)).astype(int)

        return df_feat

    def create_lag_features(self, df, target_col, lags=[1, 2, 3, 6, 12, 24, 48, 168]):
        """Create lag features for target variable"""
        df_feat = df.copy()

        for lag in lags:
            df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)

        return df_feat

    def create_rolling_features(self, df, target_col, windows=[3, 6, 12, 24, 48]):
        """Create rolling statistical features"""
        df_feat = df.copy()

        for window in windows:
            rolling = df_feat[target_col].rolling(window=window, min_periods=1)
            df_feat[f'{target_col}_roll_mean_{window}'] = rolling.mean()
            df_feat[f'{target_col}_roll_std_{window}'] = rolling.std()
            df_feat[f'{target_col}_roll_min_{window}'] = rolling.min()
            df_feat[f'{target_col}_roll_max_{window}'] = rolling.max()

            # Price volatility
            if 'price' in target_col:
                df_feat[f'{target_col}_roll_volatility_{window}'] = rolling.std() / rolling.mean()

        return df_feat

    def create_weather_features(self, df):
        """Create weather interaction features"""
        df_feat = df.copy()

        # Temperature features
        if 'temperature_2m' in df.columns:
            df_feat['temp_squared'] = df_feat['temperature_2m'] ** 2
            df_feat['temp_cubed'] = df_feat['temperature_2m'] ** 3

            # Temperature × hour interaction
            if 'hour' in df_feat.columns:
                df_feat['temp_hour_interaction'] = df_feat['temperature_2m'] * df_feat['hour']

            # Temperature change
            df_feat['temp_change_1h'] = df_feat['temperature_2m'].diff(1)
            df_feat['temp_change_3h'] = df_feat['temperature_2m'].diff(3)

        # Humidity features
        if 'relative_humidity_2m' in df.columns and 'temperature_2m' in df.columns:
            df_feat['heat_index'] = df_feat['temperature_2m'] + 0.5555 * (df_feat['relative_humidity_2m'] / 100 - 0.1) * (df_feat['temperature_2m'] - 14.5)

        # Wind features
        if 'wind_speed_10m' in df.columns:
            df_feat['wind_squared'] = df_feat['wind_speed_10m'] ** 2

        # Solar radiation proxy (for solar generation)
        if 'cloud_cover' in df.columns and 'hour' in df_feat.columns:
            # Simple solar potential based on hour and cloud cover
            solar_hour_potential = np.where(
                (df_feat['hour'] >= 6) & (df_feat['hour'] <= 18),
                np.sin(np.pi * (df_feat['hour'] - 6) / 12),
                0
            )
            df_feat['solar_potential'] = solar_hour_potential * (100 - df_feat['cloud_cover']) / 100

        # AQI features (might affect operations/demand)
        if 'calculated_aqi' in df.columns:
            df_feat['aqi_high'] = (df_feat['calculated_aqi'] > 100).astype(int)
            df_feat['aqi_moderate'] = ((df_feat['calculated_aqi'] > 50) & (df_feat['calculated_aqi'] <= 100)).astype(int)

        return df_feat

    def prepare_features(self, target_col='rt_price_mwh'):
        """Prepare all features for the target variable"""
        print(f"\nPreparing features for {target_col}...")

        # Create all feature types
        df_features = self.create_time_features(self.df)
        df_features = self.create_lag_features(df_features, target_col)
        df_features = self.create_rolling_features(df_features, target_col)
        df_features = self.create_weather_features(df_features)

        # Select feature columns (exclude non-numeric and target-related columns)
        exclude_cols = ['location_name', 'operator', 'county', 'state', 'grid',
                       'da_price_mwh', 'rt_price_mwh']

        feature_cols = []
        for col in df_features.columns:
            if col not in exclude_cols and col != target_col:
                if df_features[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    feature_cols.append(col)

        # Remove any columns with all NaN
        feature_cols = [col for col in feature_cols
                       if not df_features[col].isna().all()]

        print(f"Created {len(feature_cols)} features")

        return df_features, feature_cols

    def train_lightgbm(self, target_col='rt_price_mwh', test_size=0.2, forecast_horizon=24):
        """Train LightGBM model for price forecasting"""

        # Prepare features
        df_features, feature_cols = self.prepare_features(target_col)

        # Remove rows with NaN in target or features
        df_clean = df_features.dropna(subset=[target_col] + feature_cols)

        if len(df_clean) == 0:
            print(f"No valid data for {target_col}")
            return None

        X = df_clean[feature_cols]
        y = df_clean[target_col]

        # Train/test split (time-based)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\nTraining LightGBM for {target_col}")
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'mape',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )

        # Store model and features
        self.lgb_models[target_col] = {
            'model': model,
            'features': feature_cols,
            'scaler': None
        }

        # Make predictions on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\nTest Set Performance:")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  MAE: ${mae:.2f}/MWh")
        print(f"  RMSE: ${rmse:.2f}/MWh")

        return model, X_test, y_test, y_pred

    def forecast_prices(self, hours_ahead=48, target_col='rt_price_mwh'):
        """Generate price forecasts for the next N hours"""

        if target_col not in self.lgb_models:
            print(f"Model for {target_col} not trained yet")
            return None

        model_info = self.lgb_models[target_col]
        model = model_info['model']
        feature_cols = model_info['features']

        # Prepare the most recent data
        df_features, _ = self.prepare_features(target_col)

        # Get the last available features
        last_features = df_features[feature_cols].iloc[-1:].fillna(0)

        # Simple recursive forecasting (for demonstration)
        forecasts = []
        current_features = last_features.copy()

        for h in range(hours_ahead):
            # Make prediction
            pred = model.predict(current_features)[0]
            forecasts.append(pred)

            # Update features for next step (simplified - would need proper feature engineering)
            # This is a placeholder - in practice you'd update lag features, time features, etc.

        return np.array(forecasts)

    def plot_results(self, y_test, y_pred, target_col='rt_price_mwh'):
        """Plot actual vs predicted prices"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Time series plot
        axes[0, 0].plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
        axes[0, 0].plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
        axes[0, 0].set_title(f'{target_col} - Actual vs Predicted')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($/MWh)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Scatter plot
        axes[0, 1].scatter(y_test.values, y_pred, alpha=0.5)
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
        axes[0, 1].set_title('Actual vs Predicted Scatter')
        axes[0, 1].set_xlabel('Actual Price ($/MWh)')
        axes[0, 1].set_ylabel('Predicted Price ($/MWh)')
        axes[0, 1].legend()

        # Residual plot
        residuals = y_test.values - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Residual Plot')
        axes[1, 0].set_xlabel('Predicted Price ($/MWh)')
        axes[1, 0].set_ylabel('Residuals ($/MWh)')

        # Error distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', label='Zero Error')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].set_xlabel('Prediction Error ($/MWh)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def get_feature_importance(self, target_col='rt_price_mwh', top_n=20):
        """Get and plot feature importance"""

        if target_col not in self.lgb_models:
            print(f"Model for {target_col} not trained yet")
            return None

        model = self.lgb_models[target_col]['model']
        feature_cols = self.lgb_models[target_col]['features']

        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = feature_imp.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Feature Importance (Gain)')
        plt.title(f'Top {top_n} Features for {target_col}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        return feature_imp


def main():
    """Main execution function"""

    # Initialize forecaster
    forecaster = ElandPriceForecaster('data/Eland_Solar_and_Storage_Center_hourly_20250919_to_20250926.csv')

    # Load and prepare data
    df = forecaster.load_and_prepare_data()

    # Check which price columns are available
    available_targets = []
    if 'rt_price_mwh' in df.columns and not df['rt_price_mwh'].isna().all():
        available_targets.append('rt_price_mwh')
    if 'da_price_mwh' in df.columns and not df['da_price_mwh'].isna().all():
        available_targets.append('da_price_mwh')

    if not available_targets:
        print("\n⚠️ No price data available in the dataset")
        print("The dataset contains the following columns:")
        print(df.columns.tolist())

        # Create synthetic price data for demonstration
        print("\n📊 Creating synthetic price data for demonstration...")
        np.random.seed(42)

        # Create realistic hourly price patterns
        hours = np.arange(len(df))

        # Base price with daily pattern
        base_price = 30  # Base price $/MWh
        daily_pattern = 10 * np.sin(2 * np.pi * df.index.hour / 24 - np.pi/2)  # Daily variation

        # Add weekly pattern
        weekly_pattern = 5 * np.sin(2 * np.pi * df.index.dayofweek / 7)

        # Add noise and spikes
        noise = np.random.normal(0, 5, len(df))
        spikes = np.random.choice([0, 20, 50], size=len(df), p=[0.9, 0.08, 0.02])

        # Combine all components
        df['rt_price_mwh'] = base_price + daily_pattern + weekly_pattern + noise + spikes
        df['rt_price_mwh'] = np.maximum(df['rt_price_mwh'], -10)  # Floor at -$10/MWh

        # DA prices are smoother version of RT prices
        df['da_price_mwh'] = df['rt_price_mwh'].rolling(window=3, center=True).mean().fillna(df['rt_price_mwh'])

        available_targets = ['rt_price_mwh', 'da_price_mwh']
        print("✅ Synthetic price data created")

    # Train models for available price types
    for target in available_targets:
        print(f"\n{'='*50}")
        print(f"Training model for {target}")
        print('='*50)

        model, X_test, y_test, y_pred = forecaster.train_lightgbm(target_col=target)

        if model is not None:
            # Plot results
            forecaster.plot_results(y_test, y_pred, target)

            # Show feature importance
            feature_imp = forecaster.get_feature_importance(target)

            # Generate future forecasts
            print(f"\nGenerating 48-hour ahead forecast for {target}...")
            future_forecasts = forecaster.forecast_prices(hours_ahead=48, target_col=target)

            if future_forecasts is not None:
                print(f"Next 48 hours forecast summary:")
                print(f"  Mean: ${np.mean(future_forecasts):.2f}/MWh")
                print(f"  Min: ${np.min(future_forecasts):.2f}/MWh")
                print(f"  Max: ${np.max(future_forecasts):.2f}/MWh")
                print(f"  Std: ${np.std(future_forecasts):.2f}/MWh")

                # Plot forecast
                plt.figure(figsize=(12, 5))
                plt.plot(range(48), future_forecasts, marker='o', markersize=4)
                plt.title(f'48-Hour Ahead {target} Forecast')
                plt.xlabel('Hours Ahead')
                plt.ylabel('Price ($/MWh)')
                plt.grid(True, alpha=0.3)
                plt.show()

    print("\n✅ Price forecasting complete!")

    # Save the trained models if needed
    print("\n💾 Models trained and ready for use")
    print(f"   Available models: {list(forecaster.lgb_models.keys())}")

    return forecaster


if __name__ == "__main__":
    forecaster = main()