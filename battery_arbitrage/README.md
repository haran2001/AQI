# Battery Arbitrage Analysis for Eland Solar & Storage Center

This repository contains scripts for analyzing battery arbitrage opportunities at the Eland Solar & Storage Center using CAISO price data and weather conditions in Kern County, California.

## Project Overview

**Location**: Eland Solar & Storage Center, Phase 2
**Owner**: Avantus
**Capacity**: 200 MW
**Location**: Kern County, CA (35.3733°N, -119.0187°W)
**CAISO Zone**: SP15 (Southern California)
**Trading Hub**: TH_SP15_GEN-APND

## Repository Structure

```
battery_arbitrage/
├── data/                          # Generated data files
├── caiso.py                       # Original CAISO analysis (reference)
├── caiso_sp15_data_fetch.py       # CAISO price data fetcher
├── open_metero_weather_data.py    # Weather data fetcher
├── weather_data_interpolator.py   # Weather interpolation tool
├── grid_status_caiso_data_fetch.py# Alternative CAISO data source
└── data_analysis.ipynb           # Jupyter notebook for analysis
```

## Data Pipeline & Usage Sequence

### 1. Fetch CAISO Price Data

**Script**: `caiso_sp15_data_fetch.py`

Fetches Day-Ahead hourly and Real-Time 5-minute LMP prices from CAISO OASIS API for the SP15 zone.

```bash
python caiso_sp15_data_fetch.py
```

**Outputs**:
- `data/eland_sp15_da_prices_2025-08-01_2025-08-31.csv` - Day-ahead hourly prices
- `data/eland_sp15_rt_prices_2025-08-01_2025-08-31.csv` - Real-time 5-minute prices
- `data/eland_sp15_combined_prices_2025-08-01_2025-08-31.csv` - Combined hourly data with spreads

**Key Features**:
- Uses CAISO OASIS API directly
- Handles rate limiting with exponential backoff
- Respects 31-day API limit
- Provides price statistics and spread analysis

### 2. Fetch Weather Data

**Script**: `open_metero_weather_data.py`

Fetches hourly weather data from Open-Meteo API for the exact Eland Solar location.

```bash
python open_metero_weather_data.py
```

**Outputs**:
- `data/2025-08-01_2025-08-30_open_metero_weather_data.csv` - Hourly weather data

**Weather Variables** (42 total):
- Temperature (multiple altitudes)
- Humidity and atmospheric conditions
- Wind speed/direction (multiple altitudes)
- Precipitation and weather codes
- Soil temperature and moisture
- Atmospheric pressure
- Cloud cover and visibility

### 3. Interpolate Weather Data (Optional)

**Script**: `weather_data_interpolator.py`

Interpolates hourly weather data to match the 5-minute frequency of real-time price data.

```bash
# 5-minute intervals (default - 12 points per hour)
python weather_data_interpolator.py --points-per-hour 12

# Other intervals
python weather_data_interpolator.py --points-per-hour 4   # 15-minute
python weather_data_interpolator.py --points-per-hour 6   # 10-minute
```

**Outputs**:
- `data/2025-08-01_2025-08-30_open_metero_weather_data_5min.csv` - 5-minute interpolated weather data

**Interpolation Methods**:
- **Cubic**: Temperature data (smooth curves)
- **Linear**: Humidity, pressure, cloud cover
- **Circular**: Wind direction (handles 360° boundary correctly)
- **Forward-fill**: Precipitation, weather codes (discrete events)

### 4. Analysis and Modeling

**Script**: `data_analysis.ipynb`

Jupyter notebook for correlation analysis and battery arbitrage modeling.

## Data Specifications

### CAISO Price Data
- **Frequency**: Day-Ahead (hourly), Real-Time (5-minute)
- **Unit**: $/MWh (Locational Marginal Price)
- **Market**: DAM (Day-Ahead Market), RTM (Real-Time Market)
- **Node**: TH_SP15_GEN-APND (SP15 Trading Hub)

### Weather Data
- **Frequency**: Hourly (can be interpolated to 5-minute)
- **Coordinates**: 35.3733°N, -119.0187°W (Eland Solar location)
- **Variables**: 42 weather parameters
- **Source**: Open-Meteo Historical Forecast API

### Data Volume Examples
| Dataset | Frequency | Records | Size |
|---------|-----------|---------|------|
| DA Prices | Hourly | 720 | ~24KB |
| RT Prices | 5-minute | 8,640 | ~291KB |
| Weather (Hourly) | Hourly | 720 | ~XXkB |
| Weather (5-min) | 5-minute | 8,629 | ~XXkB |

## Key Analysis Features

### Price Analysis
- Day-Ahead vs Real-Time price spreads
- Price volatility and arbitrage opportunities
- Negative pricing periods (oversupply indicators)
- Peak/off-peak patterns

### Weather Correlations
- Temperature impact on electricity demand
- Wind conditions affecting renewable generation
- Cloud cover correlation with solar output
- Atmospheric pressure and humidity effects

### Battery Arbitrage Metrics
- Optimal charging/discharging schedules
- Revenue potential from price spreads
- Weather-informed predictive modeling
- Risk assessment for different strategies

## Dependencies

```bash
pip install pandas numpy scipy requests openmeteo-requests
pip install requests-cache retry-requests
```

## Configuration

### Date Ranges
- **Current default**: August 1-31, 2025 (modify dates in each script)
- **CAISO limit**: Maximum 31 days per API request
- **Historical data**: Available for past dates only

### Location Settings
All scripts are configured for Eland Solar & Storage Center:
- **Coordinates**: 35.3733°N, -119.0187°W
- **CAISO Zone**: SP15
- **Node**: TH_SP15_GEN-APND

## Alternative Data Sources

### Grid Status API (Alternative)
**Script**: `grid_status_caiso_data_fetch.py`

Alternative CAISO data source using GridStatus.io API (requires API key).

```bash
# Set environment variable
export GRID_STATUS_API_KEY="your_api_key"
python grid_status_caiso_data_fetch.py
```

## Usage Notes

1. **Run scripts in sequence**: Price data → Weather data → Interpolation → Analysis
2. **Date consistency**: Ensure all scripts use the same date range
3. **API limits**: CAISO OASIS has a 31-day limit per request
4. **Rate limiting**: Scripts include retry logic for API rate limits
5. **Data validation**: Check output files for completeness before analysis

## Output File Naming Convention

```
{start_date}_{end_date}_{source}_{location}_{frequency}_{data_type}.csv

Examples:
- 2025-08-01_2025-08-31_caiso_sp15_combined_prices.csv
- 2025-08-01_2025-08-30_open_metero_weather_data_5min.csv
- eland_sp15_da_prices_2025-08-01_2025-08-31.csv
```

## Troubleshooting

### Common Issues

1. **"No price column found"**: Usually indicates future dates or API errors
2. **Rate limiting**: Scripts include automatic retry with exponential backoff
3. **Missing interpolated values**: Fixed in latest version of weather_data_interpolator.py
4. **Date format errors**: Ensure consistent YYYY-MM-DD format

### API Errors

- **CAISO Error 1004**: Date range exceeds 31 days
- **Open-Meteo timeout**: Check internet connection, retry automatically included

## Contributing

When adding new scripts or modifying existing ones:
1. Maintain consistent date range parameters
2. Include proper error handling and rate limiting
3. Follow the naming convention for output files
4. Update this README with new features

## License

This project is for research and analysis purposes related to battery storage arbitrage in California energy markets.