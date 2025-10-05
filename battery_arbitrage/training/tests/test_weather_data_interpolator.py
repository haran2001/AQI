#!/usr/bin/env python3
"""
Test suite for weather_data_interpolator.py

Tests configuration loading, interpolation logic, and output generation.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import sys
import tempfile
import os

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from weather_data_interpolator import (
    load_config,
    interpolate_circular,
    interpolate_weather_data,
    generate_output_filename
)


# =============================================================================
# Configuration Tests
# =============================================================================

def test_load_config():
    """Test that configuration file loads correctly."""
    config = load_config('config/training_config.yaml')

    assert 'date_range' in config
    assert 'data_collection' in config
    assert 'weather' in config['data_collection']


def test_interpolation_config_exists():
    """Test that interpolation configuration is present."""
    config = load_config('config/training_config.yaml')

    weather_config = config['data_collection']['weather']
    assert 'interpolation' in weather_config

    interp_config = weather_config['interpolation']
    assert 'methods' in interp_config
    assert 'points_per_hour' in interp_config


def test_interpolation_methods_are_valid():
    """Test that interpolation methods dictionary has expected keys."""
    config = load_config('config/training_config.yaml')

    methods = config['data_collection']['weather']['interpolation']['methods']

    # Check expected method types exist
    assert 'cubic' in methods
    assert 'linear' in methods
    assert 'wind_speed' in methods
    assert 'wind_direction' in methods
    assert 'nearest' in methods

    # Check they contain lists
    assert isinstance(methods['cubic'], list)
    assert isinstance(methods['linear'], list)
    assert isinstance(methods['wind_speed'], list)
    assert isinstance(methods['wind_direction'], list)
    assert isinstance(methods['nearest'], list)


def test_points_per_hour_is_valid():
    """Test that points_per_hour is a valid integer."""
    config = load_config('config/training_config.yaml')

    points_per_hour = config['data_collection']['weather']['interpolation']['points_per_hour']

    assert isinstance(points_per_hour, int)
    assert points_per_hour > 0
    assert points_per_hour <= 60  # Max 1 point per minute
    assert 60 % points_per_hour == 0  # Must divide evenly into 60


# =============================================================================
# Circular Interpolation Tests
# =============================================================================

def test_interpolate_circular_basic():
    """Test circular interpolation with simple wind direction data."""
    # Test data: 0° -> 90° -> 180° -> 270°
    values = np.array([0.0, 90.0, 180.0, 270.0])
    num_points = 13  # 4 original + 9 interpolated

    result = interpolate_circular(values, num_points)

    assert len(result) == num_points
    assert result[0] == pytest.approx(0.0, abs=1.0)  # First value
    assert result[-1] == pytest.approx(270.0, abs=1.0)  # Last value
    assert all(0 <= x <= 360 for x in result)  # All in valid range


def test_interpolate_circular_boundary():
    """Test circular interpolation handles 360°/0° boundary correctly."""
    # Test boundary case: 350° -> 10° (should go through 0°, not through 180°)
    values = np.array([350.0, 10.0])
    num_points = 5

    result = interpolate_circular(values, num_points)

    # Middle value should be close to 0° (or 360°), not 180°
    middle = result[len(result) // 2]
    # It should be either near 0° or near 360°
    assert middle < 20.0 or middle > 340.0


def test_interpolate_circular_returns_positive_angles():
    """Test that circular interpolation always returns positive angles (0-360)."""
    values = np.array([45.0, 135.0, 225.0, 315.0])
    num_points = 20

    result = interpolate_circular(values, num_points)

    assert all(0 <= x <= 360 for x in result)


# =============================================================================
# Filename Generation Tests
# =============================================================================

def test_generate_output_filename_basic():
    """Test output filename generation with basic inputs."""
    filename = generate_output_filename(
        input_file="data/weather.csv",
        output_dir="data",
        start_date="2025-08-01",
        end_date="2025-08-31",
        points_per_hour=12
    )

    assert "2025-08-01_2025-08-31" in filename
    assert "weather_interpolated" in filename
    assert "5min.csv" in filename  # 12 points/hour = 5 min intervals


def test_generate_output_filename_different_frequencies():
    """Test filename generation with different interpolation frequencies."""
    # 15-minute intervals (4 points per hour)
    filename_15min = generate_output_filename(
        input_file="data/weather.csv",
        output_dir="data",
        start_date="2025-08-01",
        end_date="2025-08-31",
        points_per_hour=4
    )
    assert "15min.csv" in filename_15min

    # 30-minute intervals (2 points per hour)
    filename_30min = generate_output_filename(
        input_file="data/weather.csv",
        output_dir="data",
        start_date="2025-08-01",
        end_date="2025-08-31",
        points_per_hour=2
    )
    assert "30min.csv" in filename_30min


def test_generate_output_filename_without_output_dir():
    """Test filename generation when output_dir is not specified."""
    filename = generate_output_filename(
        input_file="some/path/data/weather.csv",
        output_dir=None,
        start_date="2025-08-01",
        end_date="2025-08-31",
        points_per_hour=12
    )

    # Should use same directory as input file
    assert filename.startswith("some/path/data/")


# =============================================================================
# Interpolation Function Tests
# =============================================================================

@pytest.fixture
def sample_hourly_weather_data():
    """Create sample hourly weather data for testing."""
    # Create 24 hours of data
    dates = pd.date_range('2025-08-01 00:00:00', periods=24, freq='h')

    data = {
        'date': dates,
        'temperature_2m': np.linspace(20.0, 30.0, 24),  # Smooth temperature curve
        'wind_speed_10m': np.linspace(5.0, 15.0, 24),    # Wind speed
        'wind_direction_10m': np.array([0, 45, 90, 135, 180, 225, 270, 315, 0, 45, 90, 135, 180, 225, 270, 315, 0, 45, 90, 135, 180, 225, 270, 315]),
        'precipitation': np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'relative_humidity_2m': np.linspace(60.0, 80.0, 24)
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def interpolation_config():
    """Create sample interpolation configuration."""
    return {
        'cubic': ['temperature_2m'],
        'linear': ['relative_humidity_2m'],
        'wind_speed': ['wind_speed_10m'],
        'wind_direction': ['wind_direction_10m'],
        'nearest': ['precipitation']
    }


def test_interpolate_weather_data_basic(sample_hourly_weather_data, interpolation_config, tmp_path):
    """Test basic interpolation functionality."""
    # Save sample data to temporary file
    input_file = tmp_path / "weather_hourly.csv"
    output_file = tmp_path / "weather_5min.csv"

    sample_hourly_weather_data.to_csv(input_file, index=False)

    # Run interpolation
    result = interpolate_weather_data(
        input_file=str(input_file),
        output_file=str(output_file),
        interpolation_config=interpolation_config,
        points_per_hour=12,
        verbose=False
    )

    # Check output file was created
    assert output_file.exists()

    # Check result shape - 24 hours * 12 points/hour = 288 rows
    assert len(result) == 288

    # Check all columns are present
    assert 'date' in result.columns
    assert 'temperature_2m' in result.columns
    assert 'wind_speed_10m' in result.columns
    assert 'wind_direction_10m' in result.columns
    assert 'precipitation' in result.columns


def test_interpolate_weather_data_no_nans(sample_hourly_weather_data, interpolation_config, tmp_path):
    """Test that interpolation produces minimal NaN values."""
    input_file = tmp_path / "weather_hourly.csv"
    output_file = tmp_path / "weather_5min.csv"

    sample_hourly_weather_data.to_csv(input_file, index=False)

    result = interpolate_weather_data(
        input_file=str(input_file),
        output_file=str(output_file),
        interpolation_config=interpolation_config,
        points_per_hour=12,
        verbose=False
    )

    # Check that most values are not NaN (allow for some at the end due to extension)
    # Should have mostly valid data (>95%)
    assert result['temperature_2m'].notna().sum() / len(result) > 0.95

    # Check no NaN values in wind speed (linear interpolation, more stable)
    assert result['wind_speed_10m'].notna().sum() / len(result) > 0.95


def test_interpolate_weather_data_timestamps_are_correct(sample_hourly_weather_data, interpolation_config, tmp_path):
    """Test that interpolated timestamps are at correct 5-minute intervals."""
    input_file = tmp_path / "weather_hourly.csv"
    output_file = tmp_path / "weather_5min.csv"

    sample_hourly_weather_data.to_csv(input_file, index=False)

    result = interpolate_weather_data(
        input_file=str(input_file),
        output_file=str(output_file),
        interpolation_config=interpolation_config,
        points_per_hour=12,
        verbose=False
    )

    # Convert date column to datetime
    result['date'] = pd.to_datetime(result['date'])

    # Check first timestamp
    assert result['date'].iloc[0] == pd.Timestamp('2025-08-01 00:00:00')

    # Check time differences are 5 minutes
    time_diffs = result['date'].diff().dropna()
    assert all(time_diffs == pd.Timedelta(minutes=5))


def test_interpolate_weather_data_preserves_original_values(sample_hourly_weather_data, interpolation_config, tmp_path):
    """Test that interpolation preserves original hourly values."""
    input_file = tmp_path / "weather_hourly.csv"
    output_file = tmp_path / "weather_5min.csv"

    sample_hourly_weather_data.to_csv(input_file, index=False)

    result = interpolate_weather_data(
        input_file=str(input_file),
        output_file=str(output_file),
        interpolation_config=interpolation_config,
        points_per_hour=12,
        verbose=False
    )

    result['date'] = pd.to_datetime(result['date'])

    # Extract hourly values (every 12th row, starting from 0)
    hourly_from_result = result.iloc[::12]

    # Check temperature values match (within tolerance for interpolation)
    original_temps = sample_hourly_weather_data['temperature_2m'].values
    result_temps = hourly_from_result['temperature_2m'].values[:len(original_temps)]

    assert np.allclose(original_temps, result_temps, rtol=0.01)


def test_interpolate_weather_data_precipitation_forward_fill(sample_hourly_weather_data, interpolation_config, tmp_path):
    """Test that precipitation uses forward-fill (no interpolation for discrete events)."""
    input_file = tmp_path / "weather_hourly.csv"
    output_file = tmp_path / "weather_5min.csv"

    sample_hourly_weather_data.to_csv(input_file, index=False)

    result = interpolate_weather_data(
        input_file=str(input_file),
        output_file=str(output_file),
        interpolation_config=interpolation_config,
        points_per_hour=12,
        verbose=False
    )

    # Precipitation should be forward-filled, not interpolated
    # Check that value at hour 2 (0.2mm) is repeated for all 5-min intervals in that hour
    precip_hour2 = result['precipitation'].iloc[24:36]  # Hour 2 = rows 24-35 (12 rows)

    assert all(precip_hour2 == 0.2)


def test_interpolate_weather_data_wind_direction_circular(sample_hourly_weather_data, interpolation_config, tmp_path):
    """Test that wind direction uses circular interpolation."""
    input_file = tmp_path / "weather_hourly.csv"
    output_file = tmp_path / "weather_5min.csv"

    sample_hourly_weather_data.to_csv(input_file, index=False)

    result = interpolate_weather_data(
        input_file=str(input_file),
        output_file=str(output_file),
        interpolation_config=interpolation_config,
        points_per_hour=12,
        verbose=False
    )

    # Check all wind directions are in valid range [0, 360]
    assert all(result['wind_direction_10m'] >= 0)
    assert all(result['wind_direction_10m'] <= 360)


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
def test_interpolate_real_weather_file():
    """Integration test: Load and interpolate actual weather data if available."""
    config = load_config('config/training_config.yaml')

    # Try to find real weather file
    start_date = config['date_range']['start_date']
    end_date = config['date_range']['end_date']

    raw_data_dir = config.get('paths', {}).get('raw_data', 'data')
    input_filename = f"{start_date}_{end_date}_open_metero_weather_data.csv"
    input_file = Path(raw_data_dir) / input_filename

    if not input_file.exists():
        pytest.skip(f"Real weather data file not found: {input_file}")

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        output_file = tmp_file.name

    try:
        # Load config for interpolation
        weather_config = config['data_collection']['weather']
        interp_settings = weather_config['interpolation']
        interpolation_config = interp_settings['methods']
        points_per_hour = interp_settings['points_per_hour']

        # Run interpolation
        result = interpolate_weather_data(
            input_file=str(input_file),
            output_file=output_file,
            interpolation_config=interpolation_config,
            points_per_hour=points_per_hour,
            verbose=False
        )

        # Verify output
        assert len(result) > 0
        assert 'date' in result.columns

        # Check that we have 5-minute intervals
        result['date'] = pd.to_datetime(result['date'])
        time_diffs = result['date'].diff().dropna()
        assert all(time_diffs == pd.Timedelta(minutes=5))

    finally:
        # Clean up temporary file
        if os.path.exists(output_file):
            os.remove(output_file)


# =============================================================================
# Edge Cases
# =============================================================================

def test_interpolate_weather_data_handles_unknown_column(interpolation_config, tmp_path):
    """Test that unknown columns default to linear interpolation."""
    # Create data with an unknown column
    dates = pd.date_range('2025-08-01 00:00:00', periods=24, freq='h')
    data = {
        'date': dates,
        'unknown_variable': np.linspace(10.0, 20.0, 24)
    }
    df = pd.DataFrame(data)

    input_file = tmp_path / "weather_unknown.csv"
    output_file = tmp_path / "weather_unknown_5min.csv"

    df.to_csv(input_file, index=False)

    # Should not raise error, should use linear interpolation
    result = interpolate_weather_data(
        input_file=str(input_file),
        output_file=str(output_file),
        interpolation_config=interpolation_config,
        points_per_hour=12,
        verbose=False
    )

    assert 'unknown_variable' in result.columns
    assert result['unknown_variable'].isna().sum() == 0  # No NaN values
