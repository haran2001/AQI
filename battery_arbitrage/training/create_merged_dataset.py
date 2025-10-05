#!/usr/bin/env python3
"""
Create Merged Dataset

Merges CAISO real-time price data with interpolated weather data for analysis.
Both datasets now have matching 5-minute intervals and timestamps.

Usage:
    python create_merged_dataset.py --config config/training_config.yaml
    python create_merged_dataset.py --config config/training_config.yaml --price-file data/prices.csv
"""

import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_merged_dataset(
    weather_file: str,
    price_file: str,
    output_file: str,
    basic_features_config: Dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Merge weather and price data with aligned timestamps.

    Args:
        weather_file: Path to interpolated weather CSV file
        price_file: Path to real-time price CSV file
        output_file: Path to output merged CSV file
        basic_features_config: Configuration for basic feature engineering
        verbose: Whether to print progress messages

    Returns:
        Merged DataFrame with basic features
    """
    if verbose:
        print("=" * 60)
        print("Creating Merged Dataset: Weather + CAISO RT Prices")
        print("=" * 60)

    # Load datasets
    if verbose:
        print("\nLoading datasets...")

    weather_df = pd.read_csv(weather_file)
    price_df = pd.read_csv(price_file)

    # Parse timestamps
    weather_df['timestamp'] = pd.to_datetime(weather_df['date'])
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])

    if verbose:
        print(f"Weather data: {len(weather_df)} records")
        print(f"Price data: {len(price_df)} records")

    # Drop the original date column from weather data
    weather_df = weather_df.drop('date', axis=1)

    # Set timestamp as index for both
    weather_df = weather_df.set_index('timestamp')
    price_df = price_df.set_index('timestamp')

    if verbose:
        print(f"\nDate ranges:")
        print(f"Weather: {weather_df.index.min()} to {weather_df.index.max()}")
        print(f"Prices:  {price_df.index.min()} to {price_df.index.max()}")

    # Merge on timestamp index
    if verbose:
        print("\nMerging datasets...")
    merged_df = price_df.join(weather_df, how='inner', rsuffix='_weather')

    if verbose:
        print(f"Merged dataset: {len(merged_df)} records")

    # Verify no missing values in key columns
    key_cols = ['price_mwh', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
    missing_counts = merged_df[key_cols].isnull().sum()

    if verbose:
        print(f"\nMissing values check:")
        for col, missing in missing_counts.items():
            status = "✓" if missing == 0 else "✗"
            print(f"{status} {col}: {missing} missing")

    # Add derived features for analysis
    if verbose:
        print("\nAdding derived features...")

    # Extract configuration values
    temporal_config = basic_features_config.get('temporal', {})
    peak_hours_config = basic_features_config.get('peak_hours', {})
    weather_conversions = basic_features_config.get('weather_conversions', {})
    price_indicators = basic_features_config.get('price_indicators', {})
    cyclical_config = basic_features_config.get('cyclical', {})

    # Time-based features (if enabled)
    if temporal_config.get('enabled', True):
        merged_df['hour'] = merged_df.index.hour
        merged_df['day_of_week'] = merged_df.index.dayofweek
        merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6])

        # Peak hours configuration
        peak_start = peak_hours_config.get('start_hour', 16)
        peak_end = peak_hours_config.get('end_hour', 21)
        merged_df['is_peak_hours'] = merged_df['hour'].between(peak_start, peak_end)

    # Cyclical encoding (if enabled)
    if cyclical_config.get('enabled', True):
        merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df['hour'] / 24)
        merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df['hour'] / 24)
        merged_df['day_sin'] = np.sin(2 * np.pi * merged_df['day_of_week'] / 7)
        merged_df['day_cos'] = np.cos(2 * np.pi * merged_df['day_of_week'] / 7)

    # Price indicator features (if enabled)
    if price_indicators.get('enabled', True):
        price_neg_threshold = price_indicators.get('price_negative_threshold', 0)
        price_high_threshold = price_indicators.get('price_high_threshold', 100)
        merged_df['price_negative'] = merged_df['price_mwh'] < price_neg_threshold
        merged_df['price_high'] = merged_df['price_mwh'] > price_high_threshold

    # Weather conversions (if enabled)
    if weather_conversions.get('enabled', True):
        merged_df['temp_celsius'] = merged_df['temperature_2m']
        merged_df['temp_fahrenheit'] = (merged_df['temperature_2m'] * 9/5) + 32
        merged_df['wind_speed_mph'] = merged_df['wind_speed_10m'] * 2.237  # m/s to mph

    # Reset index to have timestamp as column
    merged_df = merged_df.reset_index()

    # Reorder columns for better readability
    col_order = [
        'timestamp', 'price_mwh', 'hour', 'day_of_week', 'is_weekend', 'is_peak_hours',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
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
    merged_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\n✓ Merged dataset saved to: {output_file}")
        print(f"  Shape: {merged_df.shape}")

        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Date range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
        print(f"  Price range: ${merged_df['price_mwh'].min():.2f} to ${merged_df['price_mwh'].max():.2f}/MWh")
        print(f"  Avg price: ${merged_df['price_mwh'].mean():.2f}/MWh")
        print(f"  Temperature range: {merged_df['temperature_2m'].min():.1f}°C to {merged_df['temperature_2m'].max():.1f}°C")

        if 'price_negative' in merged_df.columns:
            print(f"  Negative price periods: {merged_df['price_negative'].sum()} ({merged_df['price_negative'].mean()*100:.1f}%)")
        if 'price_high' in merged_df.columns:
            print(f"  High price periods (>$100): {merged_df['price_high'].sum()} ({merged_df['price_high'].mean()*100:.1f}%)")

    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description='Merge CAISO price data with interpolated weather data'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--weather-file',
        type=str,
        default=None,
        help='Path to interpolated weather CSV file (overrides config)'
    )
    parser.add_argument(
        '--price-file',
        type=str,
        default=None,
        help='Path to RT price CSV file (overrides config)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to output merged CSV file (overrides config)'
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 60)
    print("Create Merged Dataset")
    print("=" * 60)
    print(f"\nLoading configuration from: {args.config}")

    config = load_config(args.config)

    # Extract config values
    date_config = config.get('date_range', {})
    start_date = date_config.get('start_date')
    end_date = date_config.get('end_date')

    paths_config = config.get('paths', {})
    processed_data_dir = paths_config.get('processed_dir', 'data/processed')
    raw_data_dir = paths_config.get('raw_dir', 'data/raw')

    basic_features_config = config.get('features', {}).get('basic', {})

    # Determine weather file
    if args.weather_file:
        weather_file = args.weather_file
    else:
        # Construct default weather filename from config
        interp_settings = config.get('data_collection', {}).get('weather', {}).get('interpolation', {})
        points_per_hour = interp_settings.get('points_per_hour', 12)
        minutes = 60 // points_per_hour
        weather_filename = f"{start_date}_{end_date}_weather_interpolated_{minutes}min.csv"
        weather_file = str(Path(raw_data_dir) / weather_filename)

    # Determine price file
    if args.price_file:
        price_file = args.price_file
    else:
        # Construct default price filename from config
        # Use simple caiso naming convention: caiso_rt_prices_{start}_{end}.csv
        price_filename = f"caiso_rt_prices_{start_date}_{end_date}.csv"
        price_file = str(Path(raw_data_dir) / price_filename)

    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        output_filename = f"merged_weather_prices_{start_date}_{end_date}.csv"
        output_file = str(Path(processed_data_dir) / output_filename)

    print(f"\nConfiguration:")
    print(f"  Weather file: {weather_file}")
    print(f"  Price file: {price_file}")
    print(f"  Output file: {output_file}")

    # Validate input files exist
    if not Path(weather_file).exists():
        raise FileNotFoundError(f"Weather file not found: {weather_file}")
    if not Path(price_file).exists():
        raise FileNotFoundError(f"Price file not found: {price_file}")

    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Run merge
    print("\n" + "=" * 60)
    merged_data = create_merged_dataset(
        weather_file=weather_file,
        price_file=price_file,
        output_file=output_file,
        basic_features_config=basic_features_config,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Merged dataset ready for analysis!")
    print("=" * 60)


if __name__ == "__main__":
    main()