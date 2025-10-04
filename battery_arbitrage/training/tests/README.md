# Battery Arbitrage Training Pipeline Tests

## Overview

This directory contains unit and integration tests for the battery arbitrage model training pipeline.

## Test Structure

```
tests/
├── __init__.py
├── README.md
└── test_caiso_sp15_data_fetch.py    # Tests for CAISO data fetcher
```

## Running Tests

### Run All Unit Tests (Fast)

```bash
pytest tests/ -k "not integration"
```

### Run All Tests (Including Integration Tests)

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_caiso_sp15_data_fetch.py -v
```

### Run Specific Test Function

```bash
pytest tests/test_caiso_sp15_data_fetch.py::test_load_config -v
```

### Run with Coverage Report

```bash
pip install pytest-cov
pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### Unit Tests (Fast, No Network)

- **Configuration Tests**: Validate YAML config loading and structure
- **Date Chunking Tests**: Test date range splitting logic for >30 day periods
- **Formatting Tests**: Test datetime formatting for CAISO API
- **Parsing Tests**: Test XML parsing with mock data
- **Validation Tests**: Test data validation logic

### Integration Tests (Slow, Network Required)

- **API Tests**: Test actual CAISO API calls (marked with `@pytest.mark.integration`)

**Skip integration tests:**
```bash
pytest tests/ -k "not integration"
# or set environment variable
SKIP_INTEGRATION=1 pytest tests/
```

## Test Coverage

### `test_caiso_sp15_data_fetch.py`

✅ **test_load_config** - Verifies YAML config loads correctly
✅ **test_config_values_are_valid** - Validates config parameter ranges
✅ **test_chunk_date_range_no_chunking_needed** - Tests <30 day ranges
✅ **test_chunk_date_range_exactly_30_days** - Tests exactly 30 days
✅ **test_chunk_date_range_needs_chunking** - Tests 90-day chunking
✅ **test_chunk_date_range_no_overlap** - Verifies chunks don't overlap
✅ **test_chunk_date_range_custom_max_days** - Tests custom chunk sizes
✅ **test_format_dt_for_oasis** - Tests datetime formatting
✅ **test_format_dt_for_oasis_with_time** - Tests with specific times
✅ **test_parse_oasis_xml_empty** - Tests empty XML handling
✅ **test_parse_oasis_xml_with_mock_data** - Tests XML parsing
✅ **test_date_range_in_config_is_valid** - Validates config dates
🔶 **test_fetch_small_dataset** - Integration test (requires network)

**Current Coverage: 12/13 tests passing (92%)**

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Example Test

```python
def test_my_function():
    """Test description"""
    # Arrange
    input_data = "test"

    # Act
    result = my_function(input_data)

    # Assert
    assert result == expected_value, "Error message"
```

### Marking Tests

```python
@pytest.mark.integration
def test_api_call():
    """This test requires network access"""
    pass

@pytest.mark.slow
def test_long_running():
    """This test takes >10 seconds"""
    pass
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ -k "not integration" --cov
```

## Dependencies

Required packages:
- `pytest>=7.0`
- `pyyaml` (for config loading)
- `pandas` (for data handling)

Optional packages:
- `pytest-cov` (for coverage reports)
- `pytest-xdist` (for parallel test execution)

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running pytest from the project root:

```bash
cd /Users/hari/Desktop/aqi/battery_arbitrage/training
pytest tests/
```

### Integration Tests Timeout

If CAISO API tests timeout, increase the timeout in the config:

```yaml
data_collection:
  caiso:
    timeout_seconds: 300
```

### Config File Not Found

Tests expect config at `config/training_config.yaml`. Verify the file exists:

```bash
ls -la config/training_config.yaml
```

## Future Tests

Planned test additions:

- [ ] `test_open_metero_weather_data.py` - Weather data fetcher tests
- [ ] `test_weather_data_interpolator.py` - Interpolation logic tests
- [ ] `test_create_merged_dataset.py` - Dataset merging tests
- [ ] `test_xgboost_price_forecaster.py` - Model training tests
- [ ] `test_rolling_intrinsic_battery_arbitrage.py` - Trading strategy tests
- [ ] `test_integration_pipeline.py` - End-to-end pipeline tests

## Contact

For questions or issues with tests, please open a GitHub issue.
