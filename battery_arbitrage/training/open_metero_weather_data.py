#!/usr/bin/env python3
"""
Open-Meteo Weather Data Fetcher for Eland Solar & Storage Center

Fetches historical hourly weather data from Open-Meteo API for battery arbitrage modeling.
This version is config-driven - all parameters loaded from training_config.yaml

Usage:
    python open_metero_weather_data.py --config config/training_config.yaml
    python open_metero_weather_data.py --config config/training_config.yaml --start-date 2025-08-01 --end-date 2025-08-31

Output:
    - CSV file with hourly weather data (42 variables)
"""

import os
import yaml
import argparse
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime
from pathlib import Path
from typing import Dict, List


# ----------------------------
# Configuration Loading
# ----------------------------

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ----------------------------
# Weather Data Fetcher
# ----------------------------

def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_variables: List[str],
    api_url: str,
    max_retries: int = 5,
    backoff_factor: float = 0.2,
    cache_expire_seconds: int = 3600
) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo API

    Args:
        latitude: Site latitude
        longitude: Site longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        hourly_variables: List of weather variables to fetch
        api_url: Open-Meteo API endpoint URL
        max_retries: Maximum retry attempts
        backoff_factor: Exponential backoff factor for retries
        cache_expire_seconds: Cache expiration time in seconds

    Returns:
        DataFrame with hourly weather data
    """

    # Setup Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=cache_expire_seconds)
    retry_session = retry(cache_session, retries=max_retries, backoff_factor=backoff_factor)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # API request parameters
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_variables,
    }

    print(f"Fetching weather data from Open-Meteo API...")
    print(f"  Location: {latitude}°N, {longitude}°E")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Variables: {len(hourly_variables)} weather parameters")

    # Make API request
    responses = openmeteo.weather_api(api_url, params=params)

    # Process first location (single-site mode)
    response = responses[0]

    print(f"\n✓ Data retrieved successfully:")
    print(f"  Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"  Elevation: {response.Elevation()} m asl")
    print(f"  Timezone offset: {response.UtcOffsetSeconds()}s from GMT")

    # Process hourly data
    hourly = response.Hourly()

    # Create timestamp range
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    # Extract all weather variables
    # The order of variables needs to be the same as requested
    print(f"\nExtracting {len(hourly_variables)} weather variables:")
    for i, var_name in enumerate(hourly_variables):
        try:
            hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()
            print(f"  [{i+1:2d}/{len(hourly_variables)}] {var_name}")
        except Exception as e:
            print(f"  [ERROR] Failed to extract {var_name}: {e}")
            raise

    # Create DataFrame
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    print(f"\n✓ Weather data processed:")
    print(f"  Records: {len(hourly_dataframe)}")
    print(f"  Columns: {len(hourly_dataframe.columns)} (date + {len(hourly_variables)} variables)")
    print(f"  Date range: {hourly_dataframe['date'].min()} to {hourly_dataframe['date'].max()}")

    # Data quality check
    missing_counts = hourly_dataframe.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\n⚠ Warning: Missing values detected:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"    {col}: {count} missing values")
    else:
        print(f"  ✓ No missing values")

    return hourly_dataframe


# ----------------------------
# Main Execution
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Fetch weather data from Open-Meteo API')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD) - overrides config')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD) - overrides config')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory - overrides config')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Extract configuration values
    site_info = config['site']
    weather_config = config['data_collection']['weather']
    date_range = config['date_range']
    paths = config['paths']

    # Use command-line args if provided, otherwise use config
    start_date = args.start_date if args.start_date else date_range['start_date']
    end_date = args.end_date if args.end_date else date_range['end_date']
    output_dir = args.output_dir if args.output_dir else paths['raw_dir']

    # Display configuration
    print("=" * 70)
    print("Open-Meteo Weather Data Fetcher (Config-Driven)")
    print("=" * 70)
    print(f"Site: {site_info['name']}")
    print(f"Owner: {site_info['owner']}")
    print(f"Location: {site_info['county']}, {site_info['state']}")
    print(f"Coordinates: {site_info['latitude']}°N, {site_info['longitude']}°E")
    print(f"Period: {start_date} to {end_date}")
    print(f"Provider: {weather_config['provider']}")
    print(f"Variables: {len(weather_config['hourly_variables'])}")
    print(f"Cache expiration: {weather_config['cache_expire_seconds']}s")
    print("-" * 70)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Fetch weather data
        print("\n" + "=" * 70)
        print("Fetching Weather Data")
        print("=" * 70)

        weather_df = fetch_weather_data(
            latitude=site_info['latitude'],
            longitude=site_info['longitude'],
            start_date=start_date,
            end_date=end_date,
            hourly_variables=weather_config['hourly_variables'],
            api_url=weather_config['api_url'],
            max_retries=weather_config['max_retries'],
            backoff_factor=weather_config['backoff_factor'],
            cache_expire_seconds=weather_config['cache_expire_seconds']
        )

        # Save to CSV
        output_filename = f"{output_dir}/weather_hourly_{start_date}_{end_date}.csv"
        weather_df.to_csv(output_filename, index=False)

        print("\n" + "=" * 70)
        print("Weather Data Saved")
        print("=" * 70)
        print(f"✓ File: {output_filename}")
        print(f"  Records: {len(weather_df)}")
        print(f"  Size: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")

        # Display sample statistics
        print(f"\n✓ Sample Statistics:")
        numeric_cols = weather_df.select_dtypes(include=['float64', 'int64']).columns[:5]
        for col in numeric_cols:
            mean_val = weather_df[col].mean()
            std_val = weather_df[col].std()
            min_val = weather_df[col].min()
            max_val = weather_df[col].max()
            print(f"  {col:30s}: mean={mean_val:8.2f}, std={std_val:7.2f}, min={min_val:8.2f}, max={max_val:8.2f}")

        print("\n" + "=" * 70)
        print("Data fetch completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError fetching weather data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
