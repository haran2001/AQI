#!/usr/bin/env python3
"""
Create Merged Dataset

Merges CAISO real-time price data with interpolated weather data for analysis.
Both datasets now have matching 5-minute intervals and timestamps.

Usage:
    python create_merged_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_merged_dataset():
    """
    Merge weather and price data with aligned timestamps.
    """

    print("=" * 60)
    print("Creating Merged Dataset: Weather + CAISO RT Prices")
    print("=" * 60)

    # Load datasets
    print("Loading datasets...")
    weather_file = 'data/2025-08-01_2025-08-30_open_metero_weather_data_5min.csv'
    price_file = 'data/eland_sp15_rt_prices_2025-08-01_2025-08-31.csv'

    weather_df = pd.read_csv(weather_file)
    price_df = pd.read_csv(price_file)

    # Parse timestamps
    weather_df['timestamp'] = pd.to_datetime(weather_df['date'])
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])

    print(f"Weather data: {len(weather_df)} records")
    print(f"Price data: {len(price_df)} records")

    # Drop the original date column from weather data
    weather_df = weather_df.drop('date', axis=1)

    # Set timestamp as index for both
    weather_df = weather_df.set_index('timestamp')
    price_df = price_df.set_index('timestamp')

    print(f"\nDate ranges:")
    print(f"Weather: {weather_df.index.min()} to {weather_df.index.max()}")
    print(f"Prices:  {price_df.index.min()} to {price_df.index.max()}")

    # Merge on timestamp index
    print("\nMerging datasets...")
    merged_df = price_df.join(weather_df, how='inner', rsuffix='_weather')

    print(f"Merged dataset: {len(merged_df)} records")

    # Verify no missing values in key columns
    key_cols = ['price_mwh', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
    missing_counts = merged_df[key_cols].isnull().sum()

    print(f"\nMissing values check:")
    for col, missing in missing_counts.items():
        status = "✓" if missing == 0 else "✗"
        print(f"{status} {col}: {missing} missing")

    # Add derived features for analysis
    print("\nAdding derived features...")

    # Time-based features
    merged_df['hour'] = merged_df.index.hour
    merged_df['day_of_week'] = merged_df.index.dayofweek
    merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6])

    # Price features
    merged_df['price_negative'] = merged_df['price_mwh'] < 0
    merged_df['price_high'] = merged_df['price_mwh'] > 100  # High price threshold

    # Weather features
    merged_df['temp_celsius'] = merged_df['temperature_2m']
    merged_df['temp_fahrenheit'] = (merged_df['temperature_2m'] * 9/5) + 32
    merged_df['wind_speed_mph'] = merged_df['wind_speed_10m'] * 2.237  # m/s to mph

    # Peak/off-peak periods (rough approximation)
    merged_df['is_peak_hours'] = merged_df['hour'].between(16, 21)  # 4-9 PM

    # Reset index to have timestamp as column
    merged_df = merged_df.reset_index()

    # Reorder columns for better readability
    col_order = [
        'timestamp', 'price_mwh', 'hour', 'day_of_week', 'is_weekend', 'is_peak_hours',
        'price_negative', 'price_high',
        'temperature_2m', 'temp_fahrenheit', 'relative_humidity_2m',
        'wind_speed_10m', 'wind_speed_mph', 'wind_direction_10m',
        'pressure_msl', 'cloud_cover'
    ]

    # Add remaining weather columns
    weather_cols = [col for col in merged_df.columns if col not in col_order]
    final_col_order = col_order + weather_cols

    # Only include columns that exist
    final_col_order = [col for col in final_col_order if col in merged_df.columns]
    merged_df = merged_df[final_col_order]

    # Save merged dataset
    output_file = 'data/merged_weather_prices_2025-08-01_2025-08-30.csv'
    merged_df.to_csv(output_file, index=False)

    print(f"\n✓ Merged dataset saved to: {output_file}")
    print(f"  Shape: {merged_df.shape}")

    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Date range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
    print(f"  Price range: ${merged_df['price_mwh'].min():.2f} to ${merged_df['price_mwh'].max():.2f}/MWh")
    print(f"  Avg price: ${merged_df['price_mwh'].mean():.2f}/MWh")
    print(f"  Temperature range: {merged_df['temperature_2m'].min():.1f}°C to {merged_df['temperature_2m'].max():.1f}°C")
    print(f"  Negative price periods: {merged_df['price_negative'].sum()} ({merged_df['price_negative'].mean()*100:.1f}%)")
    print(f"  High price periods (>$100): {merged_df['price_high'].sum()} ({merged_df['price_high'].mean()*100:.1f}%)")

    return merged_df

if __name__ == "__main__":
    merged_data = create_merged_dataset()
    print("\n" + "=" * 60)
    print("Merged dataset ready for analysis!")