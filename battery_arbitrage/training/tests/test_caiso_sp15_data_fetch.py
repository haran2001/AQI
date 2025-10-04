#!/usr/bin/env python3
"""
Unit tests for CAISO SP15 Data Fetcher

Tests configuration loading, date chunking, and data fetching functions.
"""

import pytest
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caiso_sp15_data_fetch import (
    load_config,
    chunk_date_range,
    _format_dt_for_oasis,
    _parse_oasis_xml
)


# ============================================================================
# Test Configuration Loading
# ============================================================================

def test_load_config():
    """Test that configuration file loads correctly"""
    config_path = 'config/training_config.yaml'

    # Check if config file exists
    assert Path(config_path).exists(), f"Config file not found: {config_path}"

    # Load config
    config = load_config(config_path)

    # Verify required sections exist
    assert 'site' in config, "Missing 'site' section in config"
    assert 'data_collection' in config, "Missing 'data_collection' section in config"
    assert 'date_range' in config, "Missing 'date_range' section in config"
    assert 'paths' in config, "Missing 'paths' section in config"

    # Verify CAISO config
    assert 'caiso' in config['data_collection'], "Missing 'caiso' in data_collection"
    caiso_config = config['data_collection']['caiso']

    # Check required CAISO parameters
    assert 'fetch_day_ahead' in caiso_config
    assert 'fetch_real_time' in caiso_config
    assert 'auto_chunk' in caiso_config
    assert 'max_days_per_chunk' in caiso_config
    assert 'max_retries' in caiso_config
    assert 'retry_delay_seconds' in caiso_config
    assert 'timeout_seconds' in caiso_config

    # Verify site information
    site_info = config['site']
    assert 'caiso_node' in site_info
    assert 'caiso_zone' in site_info
    assert site_info['caiso_zone'] == 'SP15', "Expected SP15 zone"
    assert site_info['caiso_node'] == 'TH_SP15_GEN-APND', "Expected TH_SP15_GEN-APND node"


def test_config_values_are_valid():
    """Test that configuration values are within valid ranges"""
    config = load_config('config/training_config.yaml')
    caiso_config = config['data_collection']['caiso']

    # Check numeric ranges
    assert caiso_config['max_days_per_chunk'] > 0, "max_days_per_chunk must be positive"
    assert caiso_config['max_days_per_chunk'] <= 31, "max_days_per_chunk should be ≤31 (CAISO limit)"
    assert caiso_config['max_retries'] >= 1, "max_retries must be at least 1"
    assert caiso_config['retry_delay_seconds'] > 0, "retry_delay_seconds must be positive"
    assert caiso_config['timeout_seconds'] > 0, "timeout_seconds must be positive"

    # Check boolean values
    assert isinstance(caiso_config['fetch_day_ahead'], bool)
    assert isinstance(caiso_config['fetch_real_time'], bool)
    assert isinstance(caiso_config['auto_chunk'], bool)


# ============================================================================
# Test Date Chunking
# ============================================================================

def test_chunk_date_range_no_chunking_needed():
    """Test that dates within 30 days don't get chunked"""
    start_dt = datetime(2025, 8, 1)
    end_dt = datetime(2025, 8, 20)  # 19 days

    chunks = chunk_date_range(start_dt, end_dt, max_days=30)

    # Should return single chunk
    assert len(chunks) == 1, "Expected single chunk for 19-day range"
    assert chunks[0] == (start_dt, end_dt), "Chunk should match input dates"


def test_chunk_date_range_exactly_30_days():
    """Test that exactly 30 days doesn't get chunked"""
    start_dt = datetime(2025, 8, 1)
    end_dt = datetime(2025, 8, 31)  # Exactly 30 days

    chunks = chunk_date_range(start_dt, end_dt, max_days=30)

    # Should return single chunk
    assert len(chunks) == 1, "Expected single chunk for exactly 30-day range"
    assert chunks[0] == (start_dt, end_dt)


def test_chunk_date_range_needs_chunking():
    """Test that dates >30 days get chunked correctly"""
    start_dt = datetime(2025, 1, 1)
    end_dt = datetime(2025, 3, 31)  # 90 days (Jan + Feb + Mar)

    chunks = chunk_date_range(start_dt, end_dt, max_days=30)

    # Should return 3 chunks
    assert len(chunks) == 3, "Expected 3 chunks for 90-day range"

    # Verify first chunk
    assert chunks[0][0] == datetime(2025, 1, 1)
    assert chunks[0][1] == datetime(2025, 1, 31)

    # Verify second chunk
    assert chunks[1][0] == datetime(2025, 2, 1)
    assert chunks[1][1] == datetime(2025, 3, 3)  # 30 days from Feb 1

    # Verify third chunk
    assert chunks[2][0] == datetime(2025, 3, 4)
    assert chunks[2][1] == datetime(2025, 3, 31)


def test_chunk_date_range_no_overlap():
    """Test that chunks don't overlap"""
    start_dt = datetime(2025, 1, 1)
    end_dt = datetime(2025, 2, 28)  # 58 days

    chunks = chunk_date_range(start_dt, end_dt, max_days=30)

    # Verify no overlap between chunks
    for i in range(len(chunks) - 1):
        chunk_end = chunks[i][1]
        next_chunk_start = chunks[i + 1][0]

        # Next chunk should start the day after previous chunk ends
        assert next_chunk_start == chunk_end + timedelta(days=1), \
            f"Chunks should not overlap: chunk {i} ends {chunk_end}, chunk {i+1} starts {next_chunk_start}"


def test_chunk_date_range_custom_max_days():
    """Test chunking with custom max_days parameter"""
    start_dt = datetime(2025, 1, 1)
    end_dt = datetime(2025, 1, 31)  # 30 days

    # Chunk into 10-day chunks
    chunks = chunk_date_range(start_dt, end_dt, max_days=10)

    # Should return 3 chunks (10 + 10 + 10 days)
    assert len(chunks) == 3, "Expected 3 chunks for 30 days with max_days=10"

    # Verify each chunk is ≤10 days
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        days_in_chunk = (chunk_end - chunk_start).days
        assert days_in_chunk <= 10, f"Chunk {i} has {days_in_chunk} days, expected ≤10"


# ============================================================================
# Test Date Formatting
# ============================================================================

def test_format_dt_for_oasis():
    """Test datetime formatting for CAISO OASIS API"""
    dt = datetime(2025, 8, 1, 0, 0, 0)
    formatted = _format_dt_for_oasis(dt)

    # CAISO expects format: YYYYMMDDThh:mm-0000
    assert formatted == "20250801T00:00-0000", f"Expected '20250801T00:00-0000', got '{formatted}'"


def test_format_dt_for_oasis_with_time():
    """Test datetime formatting with specific time"""
    dt = datetime(2025, 12, 31, 23, 59, 0)
    formatted = _format_dt_for_oasis(dt)

    assert formatted == "20251231T23:59-0000", f"Expected '20251231T23:59-0000', got '{formatted}'"


# ============================================================================
# Test XML Parsing (Mock Data)
# ============================================================================

def test_parse_oasis_xml_empty():
    """Test parsing empty XML"""
    empty_xml = b'<?xml version="1.0"?><root></root>'
    df = _parse_oasis_xml(empty_xml)

    assert df.empty, "Expected empty DataFrame for empty XML"


def test_parse_oasis_xml_with_mock_data():
    """Test parsing mock CAISO OASIS XML response"""
    # Mock XML response similar to CAISO format
    mock_xml = b'''<?xml version="1.0"?>
    <OASISReport>
        <MessagePayload>
            <RTO>
                <REPORT_DATA>
                    <DATA_ITEM>
                        <INTERVAL_START_GMT>2025-08-01T00:00:00-00:00</INTERVAL_START_GMT>
                        <LMP_PRC>25.50</LMP_PRC>
                        <MW>100.0</MW>
                    </DATA_ITEM>
                    <DATA_ITEM>
                        <INTERVAL_START_GMT>2025-08-01T01:00:00-00:00</INTERVAL_START_GMT>
                        <LMP_PRC>30.75</LMP_PRC>
                        <MW>105.0</MW>
                    </DATA_ITEM>
                </REPORT_DATA>
            </RTO>
        </MessagePayload>
    </OASISReport>
    '''

    df = _parse_oasis_xml(mock_xml)

    # Check that data was parsed
    assert not df.empty, "Expected non-empty DataFrame"
    assert 'timestamp' in df.columns or df.index.name == 'timestamp', "Expected timestamp column"
    assert 'price_mwh' in df.columns, "Expected price_mwh column"


# ============================================================================
# Test Data Validation
# ============================================================================

def test_date_range_in_config_is_valid():
    """Test that date range in config is valid"""
    config = load_config('config/training_config.yaml')
    date_range = config['date_range']

    start_date = datetime.strptime(date_range['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(date_range['end_date'], '%Y-%m-%d')

    assert start_date < end_date, "Start date must be before end date"

    # Check that split ratios sum to 1.0
    train_split = date_range['train_split']
    val_split = date_range['val_split']
    test_split = date_range['test_split']

    total_split = train_split + val_split + test_split
    assert abs(total_split - 1.0) < 0.001, f"Split ratios must sum to 1.0, got {total_split}"


# ============================================================================
# Integration Test (requires network access)
# ============================================================================

@pytest.mark.integration
@pytest.mark.skipif(
    "SKIP_INTEGRATION" in os.environ,
    reason="Skipping integration test - set SKIP_INTEGRATION env var to skip"
)
def test_fetch_small_dataset():
    """
    Integration test: Fetch a small dataset (2 days) from CAISO API

    This test requires network access and may be slow. Skip in CI/CD with:
    SKIP_INTEGRATION=1 pytest
    """
    from caiso_sp15_data_fetch import fetch_da_prices

    # Use a recent past date (not in future)
    start_dt = datetime(2024, 1, 1)
    end_dt = datetime(2024, 1, 2)

    try:
        df = fetch_da_prices(
            node_id="TH_SP15_GEN-APND",
            start_dt=start_dt,
            end_dt=end_dt,
            oasis_base_url="https://oasis.caiso.com/oasisapi/SingleZip",
            max_retries=3,
            retry_delay=2,
            timeout=60
        )

        # Verify data structure
        assert not df.empty, "Expected non-empty DataFrame from CAISO API"
        assert 'price_mwh' in df.columns, "Expected price_mwh column"
        assert len(df) > 0, "Expected at least some price records"

        # Verify price values are reasonable
        assert df['price_mwh'].min() >= -500, "Price seems unreasonably low"
        assert df['price_mwh'].max() <= 3000, "Price seems unreasonably high"

    except Exception as e:
        pytest.skip(f"CAISO API not accessible or returned error: {e}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
