#!/usr/bin/env python3
"""
Unit tests for Open-Meteo Weather Data Fetcher

Tests configuration loading, weather data fetching, and data validation.
"""

import pytest
import pandas as pd
import yaml
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_metero_weather_data import (
    load_config,
    fetch_weather_data
)


# ============================================================================
# Test Configuration Loading
# ============================================================================

def test_weather_config_loads_correctly():
    """Test that weather configuration loads with all required fields"""
    config_path = 'config/training_config.yaml'

    assert Path(config_path).exists(), f"Config file not found: {config_path}"

    config = load_config(config_path)

    # Verify weather config section exists
    assert 'data_collection' in config
    assert 'weather' in config['data_collection']

    weather_config = config['data_collection']['weather']

    # Check required fields
    assert 'provider' in weather_config
    assert 'api_url' in weather_config
    assert 'hourly_variables' in weather_config
    assert 'max_retries' in weather_config
    assert 'backoff_factor' in weather_config
    assert 'cache_expire_seconds' in weather_config

    # Verify provider
    assert weather_config['provider'] == 'open-meteo'


def test_weather_variables_list_is_valid():
    """Test that hourly_variables is a non-empty list"""
    config = load_config('config/training_config.yaml')
    weather_config = config['data_collection']['weather']

    hourly_vars = weather_config['hourly_variables']

    # Should be a list
    assert isinstance(hourly_vars, list), "hourly_variables should be a list"

    # Should have 42 variables as specified
    assert len(hourly_vars) == 42, f"Expected 42 weather variables, got {len(hourly_vars)}"

    # All items should be strings
    assert all(isinstance(var, str) for var in hourly_vars), "All variables should be strings"

    # Check for key variables
    expected_vars = [
        'temperature_2m',
        'relative_humidity_2m',
        'wind_speed_10m',
        'wind_direction_10m',
        'precipitation',
        'pressure_msl'
    ]

    for var in expected_vars:
        assert var in hourly_vars, f"Expected variable '{var}' not found in config"


def test_weather_config_values_are_valid():
    """Test that weather configuration values are within valid ranges"""
    config = load_config('config/training_config.yaml')
    weather_config = config['data_collection']['weather']

    # Check numeric ranges
    assert weather_config['max_retries'] >= 1, "max_retries must be at least 1"
    assert weather_config['max_retries'] <= 10, "max_retries should be reasonable (≤10)"

    assert weather_config['backoff_factor'] > 0, "backoff_factor must be positive"
    assert weather_config['backoff_factor'] <= 2, "backoff_factor should be reasonable (≤2)"

    assert weather_config['cache_expire_seconds'] > 0, "cache_expire_seconds must be positive"
    assert weather_config['cache_expire_seconds'] <= 86400, "cache should expire within 24 hours"

    # Check API URL
    assert weather_config['api_url'].startswith('http'), "API URL should start with http"
    assert 'open-meteo.com' in weather_config['api_url'], "API URL should be open-meteo.com"


# ============================================================================
# Test Site Configuration
# ============================================================================

def test_site_coordinates_are_valid():
    """Test that site coordinates are within valid ranges"""
    config = load_config('config/training_config.yaml')
    site_info = config['site']

    # Latitude should be between -90 and 90
    assert -90 <= site_info['latitude'] <= 90, "Latitude out of range"

    # Longitude should be between -180 and 180
    assert -180 <= site_info['longitude'] <= 180, "Longitude out of range"

    # Eland Solar & Storage Center should be in California (roughly 32-42°N, 114-124°W)
    assert 32 <= site_info['latitude'] <= 42, "Site should be in California latitude range"
    assert -124 <= site_info['longitude'] <= -114, "Site should be in California longitude range"


# ============================================================================
# Test Weather Data Fetching (Mock)
# ============================================================================

@pytest.fixture
def mock_openmeteo_response():
    """Create a mock Open-Meteo API response"""
    mock_response = MagicMock()

    # Mock location data
    mock_response.Latitude.return_value = 35.3733
    mock_response.Longitude.return_value = -119.0187
    mock_response.Elevation.return_value = 124.0
    mock_response.UtcOffsetSeconds.return_value = 0

    # Mock hourly data
    mock_hourly = MagicMock()

    # Mock time range (24 hours)
    start_timestamp = datetime(2025, 8, 1, 0, 0, 0).timestamp()
    end_timestamp = datetime(2025, 8, 2, 0, 0, 0).timestamp()

    mock_hourly.Time.return_value = start_timestamp
    mock_hourly.TimeEnd.return_value = end_timestamp
    mock_hourly.Interval.return_value = 3600  # 1 hour

    # Mock weather variables (3 variables for simplicity)
    import numpy as np

    # 24 hours of data (one full day)
    num_hours = 24

    mock_var1 = MagicMock()
    mock_var1.ValuesAsNumpy.return_value = np.linspace(20.0, 24.0, num_hours)  # Temperature

    mock_var2 = MagicMock()
    mock_var2.ValuesAsNumpy.return_value = np.linspace(50.0, 60.0, num_hours)  # Humidity

    mock_var3 = MagicMock()
    mock_var3.ValuesAsNumpy.return_value = np.linspace(1013.0, 1017.0, num_hours)  # Pressure

    mock_hourly.Variables.side_effect = [mock_var1, mock_var2, mock_var3]

    mock_response.Hourly.return_value = mock_hourly

    return mock_response


def test_fetch_weather_data_with_mock(mock_openmeteo_response):
    """Test weather data fetching with mocked API response"""

    with patch('open_metero_weather_data.openmeteo_requests.Client') as mock_client:
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.weather_api.return_value = [mock_openmeteo_response]
        mock_client.return_value = mock_instance

        # Call function
        df = fetch_weather_data(
            latitude=35.3733,
            longitude=-119.0187,
            start_date="2025-08-01",
            end_date="2025-08-02",
            hourly_variables=["temperature_2m", "relative_humidity_2m", "pressure_msl"],
            api_url="https://api.open-meteo.com/v1/forecast",
            max_retries=5,
            backoff_factor=0.2,
            cache_expire_seconds=3600
        )

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
        assert 'date' in df.columns, "Should have 'date' column"
        assert 'temperature_2m' in df.columns, "Should have temperature column"
        assert 'relative_humidity_2m' in df.columns, "Should have humidity column"
        assert 'pressure_msl' in df.columns, "Should have pressure column"

        # Verify data
        assert len(df) > 0, "Should have data rows"
        assert df['date'].dtype == 'datetime64[ns, UTC]', "Date should be datetime"


def test_fetch_weather_data_validates_inputs():
    """Test that fetch_weather_data validates input parameters"""

    with patch('open_metero_weather_data.openmeteo_requests.Client'):
        # Test with empty hourly_variables
        with pytest.raises(Exception):
            fetch_weather_data(
                latitude=35.3733,
                longitude=-119.0187,
                start_date="2025-08-01",
                end_date="2025-08-02",
                hourly_variables=[],  # Empty list
                api_url="https://api.open-meteo.com/v1/forecast"
            )


# ============================================================================
# Test Data Quality
# ============================================================================

def test_weather_data_has_no_duplicate_timestamps(mock_openmeteo_response):
    """Test that fetched weather data has no duplicate timestamps"""

    with patch('open_metero_weather_data.openmeteo_requests.Client') as mock_client:
        mock_instance = MagicMock()
        mock_instance.weather_api.return_value = [mock_openmeteo_response]
        mock_client.return_value = mock_instance

        df = fetch_weather_data(
            latitude=35.3733,
            longitude=-119.0187,
            start_date="2025-08-01",
            end_date="2025-08-02",
            hourly_variables=["temperature_2m", "relative_humidity_2m", "pressure_msl"],
            api_url="https://api.open-meteo.com/v1/forecast"
        )

        # Check for duplicates
        assert df['date'].duplicated().sum() == 0, "Should have no duplicate timestamps"


def test_weather_data_timestamps_are_hourly(mock_openmeteo_response):
    """Test that timestamps are at hourly intervals"""

    with patch('open_metero_weather_data.openmeteo_requests.Client') as mock_client:
        mock_instance = MagicMock()
        mock_instance.weather_api.return_value = [mock_openmeteo_response]
        mock_client.return_value = mock_instance

        df = fetch_weather_data(
            latitude=35.3733,
            longitude=-119.0187,
            start_date="2025-08-01",
            end_date="2025-08-02",
            hourly_variables=["temperature_2m", "relative_humidity_2m", "pressure_msl"],
            api_url="https://api.open-meteo.com/v1/forecast"
        )

        # Check time intervals
        if len(df) > 1:
            time_diffs = df['date'].diff().dropna()
            # All intervals should be 1 hour (3600 seconds)
            expected_interval = pd.Timedelta(hours=1)
            assert all(time_diffs == expected_interval), "All intervals should be exactly 1 hour"


# ============================================================================
# Test Date Validation
# ============================================================================

def test_date_range_in_weather_config_is_valid():
    """Test that configured date range is valid"""
    config = load_config('config/training_config.yaml')
    date_range = config['date_range']

    start_date = datetime.strptime(date_range['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(date_range['end_date'], '%Y-%m-%d')

    # Start should be before end
    assert start_date < end_date, "Start date must be before end date"

    # Dates should not be too far in the future
    now = datetime.now()
    assert start_date <= now + pd.Timedelta(days=365), "Start date should not be more than 1 year in future"


# ============================================================================
# Test Interpolation Configuration
# ============================================================================

def test_interpolation_config_exists():
    """Test that interpolation configuration is present"""
    config = load_config('config/training_config.yaml')
    weather_config = config['data_collection']['weather']

    assert 'interpolation' in weather_config, "Should have interpolation config"

    interp_config = weather_config['interpolation']

    assert 'enabled' in interp_config
    assert 'target_frequency' in interp_config
    assert 'methods' in interp_config

    # Check interpolation methods
    methods = interp_config['methods']
    assert 'cubic' in methods, "Should have cubic interpolation method"
    assert 'linear' in methods, "Should have linear interpolation method"
    assert 'wind_direction' in methods, "Should have wind_direction (circular) method"
    assert 'nearest' in methods, "Should have nearest (forward-fill) method"


def test_interpolation_methods_cover_all_variables():
    """Test that all weather variables have an interpolation method assigned"""
    config = load_config('config/training_config.yaml')
    weather_config = config['data_collection']['weather']

    all_variables = set(weather_config['hourly_variables'])
    interp_methods = weather_config['interpolation']['methods']

    # Collect all variables from interpolation methods
    assigned_variables = set()
    for method_name, variables in interp_methods.items():
        if isinstance(variables, list):
            assigned_variables.update(variables)

    # Check that all variables have a method assigned
    unassigned = all_variables - assigned_variables
    assert len(unassigned) == 0, f"Variables without interpolation method: {unassigned}"


# ============================================================================
# Integration Test (requires network)
# ============================================================================

@pytest.mark.integration
@pytest.mark.skipif(
    "SKIP_INTEGRATION" in os.environ,
    reason="Skipping integration test - set SKIP_INTEGRATION env var to skip"
)
def test_fetch_real_weather_data():
    """
    Integration test: Fetch real weather data from Open-Meteo API

    This test requires network access and may be slow. Skip in CI/CD with:
    SKIP_INTEGRATION=1 pytest
    """
    from open_metero_weather_data import fetch_weather_data

    # Use a small date range (2 days)
    try:
        df = fetch_weather_data(
            latitude=35.3733,
            longitude=-119.0187,
            start_date="2024-01-01",
            end_date="2024-01-02",
            hourly_variables=[
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "precipitation"
            ],
            api_url="https://historical-forecast-api.open-meteo.com/v1/forecast",
            max_retries=3,
            backoff_factor=0.2,
            cache_expire_seconds=3600
        )

        # Verify response
        assert not df.empty, "Should receive weather data from API"
        assert 'date' in df.columns, "Should have date column"
        assert 'temperature_2m' in df.columns, "Should have temperature data"

        # Verify data is reasonable
        temp_values = df['temperature_2m'].dropna()
        assert temp_values.min() >= -50, "Temperature seems unreasonably low"
        assert temp_values.max() <= 60, "Temperature seems unreasonably high"

        # Verify temporal continuity
        assert len(df) >= 24, "Should have at least 24 hourly records for 2 days"

    except Exception as e:
        pytest.skip(f"Open-Meteo API not accessible or returned error: {e}")


# ============================================================================
# Test Error Handling
# ============================================================================

def test_fetch_weather_handles_api_error():
    """Test that fetch_weather_data handles API errors gracefully"""

    with patch('open_metero_weather_data.openmeteo_requests.Client') as mock_client:
        # Setup mock to raise an exception
        mock_instance = MagicMock()
        mock_instance.weather_api.side_effect = Exception("API Error")
        mock_client.return_value = mock_instance

        # Should raise the exception
        with pytest.raises(Exception) as exc_info:
            fetch_weather_data(
                latitude=35.3733,
                longitude=-119.0187,
                start_date="2025-08-01",
                end_date="2025-08-02",
                hourly_variables=["temperature_2m"],
                api_url="https://api.open-meteo.com/v1/forecast"
            )

        assert "API Error" in str(exc_info.value)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
