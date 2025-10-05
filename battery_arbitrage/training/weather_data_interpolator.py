#!/usr/bin/env python3
"""
Weather Data Interpolator

Interpolates hourly weather data to higher frequency intervals (e.g., 5-minute intervals).
Uses various interpolation methods appropriate for different weather variables.

Usage:
    python weather_data_interpolator.py --config config/training_config.yaml
    python weather_data_interpolator.py --config config/training_config.yaml --input-file data/weather.csv
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def interpolate_circular(values, num_points):
    """
    Interpolate circular data (e.g., wind direction in degrees).
    Handles the 360-0 degree boundary correctly.
    """
    # Convert to radians
    radians = np.deg2rad(values)

    # Convert to Cartesian coordinates
    x = np.cos(radians)
    y = np.sin(radians)

    # Create interpolation points
    old_indices = np.arange(len(values))
    new_indices = np.linspace(0, len(values) - 1, num_points)

    # Interpolate x and y separately
    f_x = interpolate.interp1d(old_indices, x, kind='linear', fill_value='extrapolate')
    f_y = interpolate.interp1d(old_indices, y, kind='linear', fill_value='extrapolate')

    new_x = f_x(new_indices)
    new_y = f_y(new_indices)

    # Convert back to angles
    new_radians = np.arctan2(new_y, new_x)
    new_degrees = np.rad2deg(new_radians)

    # Ensure positive angles (0-360)
    new_degrees = (new_degrees + 360) % 360

    return new_degrees


def interpolate_weather_data(
    input_file: str,
    output_file: str,
    interpolation_config: Dict,
    points_per_hour: int = 12,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Interpolate hourly weather data to higher frequency.

    Args:
        input_file: Path to input CSV file with hourly data
        output_file: Path to output CSV file for interpolated data
        interpolation_config: Dictionary mapping interpolation methods to variable lists
        points_per_hour: Number of data points per hour (12 for 5-min, 4 for 15-min, etc.)
        verbose: Whether to print progress messages

    Returns:
        DataFrame with interpolated data
    """
    if verbose:
        print(f"Loading weather data from: {input_file}")

    df = pd.read_csv(input_file)

    # Parse the date column
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if verbose:
        print(f"Original data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Calculate new frequency
    minutes_per_point = 60 / points_per_hour
    freq_string = f"{int(minutes_per_point)}min" if minutes_per_point.is_integer() else f"{minutes_per_point:.1f}min"

    # Extend end time to complete the last hour with all intervals
    # This ensures we get all 5-min intervals up to 23:55 instead of stopping at 23:00
    extended_end = df.index.max() + pd.Timedelta(hours=1) - pd.Timedelta(minutes=minutes_per_point)

    # Create new index with higher frequency
    new_index = pd.date_range(
        start=df.index.min(),
        end=extended_end,
        freq=freq_string
    )

    if verbose:
        print(f"\nInterpolating to {points_per_hour} points per hour ({freq_string} intervals)")
        print(f"New data will have {len(new_index)} rows (vs {len(df)} original)")

    # First, reindex the dataframe to the new frequency
    # This will create NaN values between existing hourly data points
    df_reindexed = df.reindex(new_index)

    # Initialize result dataframe
    result = pd.DataFrame(index=new_index)

    # Process each column based on its interpolation method
    for col in df.columns:
        if verbose:
            print(f"Processing {col}...", end=" ")

        # Skip if column has all NaN values
        if df[col].isna().all():
            result[col] = np.nan
            if verbose:
                print("skipped (all NaN)")
            continue

        # Determine interpolation method
        if col in interpolation_config.get('cubic', []):
            # Cubic interpolation for smooth temperature curves
            # Use pandas interpolate on the reindexed data
            result[col] = df_reindexed[col].interpolate(method='cubic')
            if verbose:
                print("cubic")

        elif col in interpolation_config.get('linear', []):
            # Linear interpolation
            result[col] = df_reindexed[col].interpolate(method='linear')
            if verbose:
                print("linear")

        elif col in interpolation_config.get('wind_direction', []):
            # Circular interpolation for wind direction
            # Need special handling for circular data
            valid_indices = ~df[col].isna()
            if valid_indices.sum() > 1:
                # Get original timestamps and values
                orig_times = pd.to_numeric(df.index[valid_indices]) / 1e9  # Convert to seconds
                orig_values = df[col][valid_indices].values

                # New timestamps
                new_times = pd.to_numeric(new_index) / 1e9

                # Convert to radians and then to x,y coordinates
                radians = np.deg2rad(orig_values)
                x_orig = np.cos(radians)
                y_orig = np.sin(radians)

                # Interpolate x and y separately
                f_x = interpolate.interp1d(orig_times, x_orig, kind='linear',
                                         fill_value='extrapolate', bounds_error=False)
                f_y = interpolate.interp1d(orig_times, y_orig, kind='linear',
                                         fill_value='extrapolate', bounds_error=False)

                x_new = f_x(new_times)
                y_new = f_y(new_times)

                # Convert back to angles
                new_radians = np.arctan2(y_new, x_new)
                result[col] = np.rad2deg(new_radians) % 360
            else:
                result[col] = df_reindexed[col].fillna(method='ffill')
            if verbose:
                print("circular")

        elif col in interpolation_config.get('wind_speed', []):
            # Linear interpolation for wind speed
            result[col] = df_reindexed[col].interpolate(method='linear')
            if verbose:
                print("linear")

        elif col in interpolation_config.get('nearest', []):
            # Forward fill for discrete variables (precipitation, weather codes)
            result[col] = df_reindexed[col].fillna(method='ffill')
            if verbose:
                print("nearest/forward-fill")

        else:
            # Default to linear for unknown columns
            result[col] = df_reindexed[col].interpolate(method='linear')
            if verbose:
                print("linear (default)")

    # Reset index to have date as a column
    result = result.reset_index()
    result = result.rename(columns={'index': 'date'})

    # Save to file
    result.to_csv(output_file, index=False)

    if verbose:
        print(f"\n✓ Interpolated data saved to: {output_file}")
        print(f"  Shape: {result.shape}")

        # Print sample statistics
        print("\nSample statistics (first 5 columns):")
        for col in result.columns[:6]:
            if col != 'date' and pd.api.types.is_numeric_dtype(result[col]):
                print(f"  {col}: mean={result[col].mean():.2f}, std={result[col].std():.2f}")

    return result


def generate_output_filename(
    input_file: str,
    output_dir: str,
    start_date: str,
    end_date: str,
    points_per_hour: int
) -> str:
    """
    Generate output filename based on config parameters.

    Args:
        input_file: Path to input file (used if no output_dir specified)
        output_dir: Output directory from config
        start_date: Start date string
        end_date: End date string
        points_per_hour: Number of points per hour

    Returns:
        Output file path
    """
    minutes = 60 // points_per_hour
    filename = f"{start_date}_{end_date}_weather_interpolated_{minutes}min.csv"

    if output_dir:
        output_path = Path(output_dir) / filename
    else:
        # Use same directory as input file
        input_path = Path(input_file)
        output_path = input_path.parent / filename

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Interpolate hourly weather data to higher frequency'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Path to input CSV file (overrides config)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to output CSV file (overrides config)'
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 60)
    print("Weather Data Interpolator")
    print("=" * 60)
    print(f"\nLoading configuration from: {args.config}")

    config = load_config(args.config)

    # Extract config values
    date_config = config.get('date_range', {})
    start_date = date_config.get('start_date')
    end_date = date_config.get('end_date')

    weather_config = config.get('data_collection', {}).get('weather', {})
    interp_settings = weather_config.get('interpolation', {})
    interpolation_config = interp_settings.get('methods', {})
    points_per_hour = interp_settings.get('points_per_hour', 12)

    raw_data_dir = config.get('paths', {}).get('raw_dir', 'data/raw')
    output_dir = raw_data_dir  # Save interpolated data to raw data folder

    # Determine input file
    if args.input_file:
        input_file = args.input_file
    else:
        # Construct default input filename from config
        input_filename = f"weather_hourly_{start_date}_{end_date}.csv"
        input_file = str(Path(raw_data_dir) / input_filename)

    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = generate_output_filename(
            input_file=input_file,
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            points_per_hour=points_per_hour
        )

    print(f"\nConfiguration:")
    print(f"  Input file: {input_file}")
    print(f"  Output file: {output_file}")
    print(f"  Points per hour: {points_per_hour} ({60 // points_per_hour}-minute intervals)")
    print(f"  Interpolation methods loaded: {list(interpolation_config.keys())}")

    # Validate input file exists
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Run interpolation
    print("\n" + "=" * 60)
    interpolate_weather_data(
        input_file=input_file,
        output_file=output_file,
        interpolation_config=interpolation_config,
        points_per_hour=points_per_hour,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Interpolation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()