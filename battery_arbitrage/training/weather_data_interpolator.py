#!/usr/bin/env python3
"""
Weather Data Interpolator

Interpolates hourly weather data to higher frequency intervals (e.g., 5-minute intervals).
Uses various interpolation methods appropriate for different weather variables.

Usage:
    python weather_data_interpolator.py
    python weather_data_interpolator.py --points-per-hour 12  # 5-minute intervals
    python weather_data_interpolator.py --points-per-hour 4   # 15-minute intervals
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import argparse
from pathlib import Path

# Configuration for interpolation methods by variable type
INTERPOLATION_CONFIG = {
    # Continuous variables - use cubic or linear interpolation
    'cubic': [
        'temperature_2m', 'apparent_temperature', 'dew_point_2m',
        'temperature_80m', 'temperature_120m', 'temperature_180m',
        'soil_temperature_0cm', 'soil_temperature_6cm',
        'soil_temperature_18cm', 'soil_temperature_54cm'
    ],

    # Smooth continuous variables - linear interpolation
    'linear': [
        'relative_humidity_2m', 'pressure_msl', 'surface_pressure',
        'visibility', 'evapotranspiration', 'et0_fao_evapotranspiration',
        'vapour_pressure_deficit', 'soil_moisture_0_to_1cm',
        'soil_moisture_1_to_3cm', 'soil_moisture_3_to_9cm',
        'soil_moisture_9_to_27cm', 'soil_moisture_27_to_81cm',
        'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high'
    ],

    # Wind data - special handling for direction (circular) and speed
    'wind_speed': [
        'wind_speed_10m', 'wind_speed_80m',
        'wind_speed_120m', 'wind_speed_180m', 'wind_gusts_10m'
    ],

    'wind_direction': [
        'wind_direction_10m', 'wind_direction_80m',
        'wind_direction_120m', 'wind_direction_180m'
    ],

    # Precipitation - use nearest neighbor (no interpolation for discrete events)
    'nearest': [
        'snow_depth', 'snowfall', 'showers', 'rain',
        'precipitation', 'precipitation_probability', 'weather_code'
    ]
}


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


def interpolate_weather_data(input_file, output_file, points_per_hour=12):
    """
    Interpolate hourly weather data to higher frequency.

    Args:
        input_file: Path to input CSV file with hourly data
        output_file: Path to output CSV file for interpolated data
        points_per_hour: Number of data points per hour (12 for 5-min, 4 for 15-min, etc.)
    """

    print(f"Loading weather data from: {input_file}")
    df = pd.read_csv(input_file)

    # Parse the date column
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

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

    print(f"\nInterpolating to {points_per_hour} points per hour ({freq_string} intervals)")
    print(f"New data will have {len(new_index)} rows (vs {len(df)} original)")

    # First, reindex the dataframe to the new frequency
    # This will create NaN values between existing hourly data points
    df_reindexed = df.reindex(new_index)

    # Initialize result dataframe
    result = pd.DataFrame(index=new_index)

    # Process each column based on its interpolation method
    for col in df.columns:
        print(f"Processing {col}...", end=" ")

        # Skip if column has all NaN values
        if df[col].isna().all():
            result[col] = np.nan
            print("skipped (all NaN)")
            continue

        # Determine interpolation method
        if col in INTERPOLATION_CONFIG['cubic']:
            # Cubic interpolation for smooth temperature curves
            # Use pandas interpolate on the reindexed data
            result[col] = df_reindexed[col].interpolate(method='cubic')
            print("cubic")

        elif col in INTERPOLATION_CONFIG['linear']:
            # Linear interpolation
            result[col] = df_reindexed[col].interpolate(method='linear')
            print("linear")

        elif col in INTERPOLATION_CONFIG['wind_direction']:
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
            print("circular")

        elif col in INTERPOLATION_CONFIG['wind_speed']:
            # Linear interpolation for wind speed
            result[col] = df_reindexed[col].interpolate(method='linear')
            print("linear")

        elif col in INTERPOLATION_CONFIG['nearest']:
            # Forward fill for discrete variables (precipitation, weather codes)
            result[col] = df_reindexed[col].fillna(method='ffill')
            print("nearest/forward-fill")

        else:
            # Default to linear for unknown columns
            result[col] = df_reindexed[col].interpolate(method='linear')
            print("linear (default)")

    # Reset index to have date as a column
    result = result.reset_index()
    result = result.rename(columns={'index': 'date'})

    # Save to file
    result.to_csv(output_file, index=False)
    print(f"\n✓ Interpolated data saved to: {output_file}")
    print(f"  Shape: {result.shape}")

    # Print sample statistics
    print("\nSample statistics (first 5 columns):")
    for col in result.columns[:6]:
        if col != 'date' and pd.api.types.is_numeric_dtype(result[col]):
            print(f"  {col}: mean={result[col].mean():.2f}, std={result[col].std():.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Interpolate hourly weather data to higher frequency')
    parser.add_argument(
        '--input-file',
        type=str,
        default='data/2025-08-01_2025-08-30_open_metero_weather_data.csv',
        help='Path to input CSV file with hourly weather data'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to output CSV file (default: adds _5min to input filename)'
    )
    parser.add_argument(
        '--points-per-hour',
        type=int,
        default=12,
        help='Number of data points per hour (12=5min, 4=15min, 2=30min)'
    )

    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output_file is None:
        input_path = Path(args.input_file)
        minutes = 60 // args.points_per_hour
        output_name = input_path.stem + f'_{minutes}min' + input_path.suffix
        args.output_file = str(input_path.parent / output_name)

    print("=" * 60)
    print("Weather Data Interpolator")
    print("=" * 60)

    # Run interpolation
    interpolate_weather_data(
        input_file=args.input_file,
        output_file=args.output_file,
        points_per_hour=args.points_per_hour
    )

    print("\nInterpolation complete!")


if __name__ == "__main__":
    main()