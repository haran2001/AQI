#!/usr/bin/env python3
"""
Combined Power Grid and Weather Data Fetcher
Fetches CAISO market data (for CA locations) and weather/AQI data from Open-Meteo API
Configurable via YAML file
"""

import os
import io
import sys
import yaml
import zipfile
import requests
import numpy as np
import pandas as pd
import argparse
import logging
import xml.etree.ElementTree as ET
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import requests_cache
from retry_requests import retry
import openmeteo_requests

# ===========================
# Logging Setup
# ===========================
def setup_logging(config: dict):
    """Setup logging based on configuration"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))

    handlers = []
    if log_config.get('console', True):
        handlers.append(logging.StreamHandler())

    if log_config.get('file'):
        handlers.append(logging.FileHandler(log_config['file']))

    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if log_config.get('timestamps', True) else '%(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers
    )

    return logging.getLogger(__name__)

# ===========================
# CAISO Data Fetching
# ===========================
class CAISODataFetcher:
    """Fetches CAISO market data"""

    OASIS_BASE = "https://oasis.caiso.com/oasisapi/SingleZip"

    # Zone to node mapping
    ZONE_TO_NODE = {
        'NP15': 'TH_NP15_GEN-APND',
        'SP15': 'TH_SP15_GEN-APND',
        'ZP26': 'TH_ZP26_GEN-APND'
    }

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.api_config = config.get('api_config', {})

    def _resolve_node_id(self, zone_or_node: str) -> str:
        """Resolve zone to node identifier"""
        if '_' in zone_or_node:
            return zone_or_node
        return self.ZONE_TO_NODE.get(zone_or_node.upper(), zone_or_node)

    def _format_dt_for_oasis(self, dt: datetime) -> str:
        """Format datetime for OASIS API"""
        return dt.strftime("%Y%m%dT%H:%M-0000")

    def _make_oasis_request(self, params: dict) -> requests.Response:
        """Make request to OASIS API with retry logic"""
        max_retries = self.api_config.get('max_retries', 3)
        timeout = self.api_config.get('timeout', 60)
        retry_delay = self.api_config.get('retry_delay', 2)

        for attempt in range(max_retries):
            try:
                response = requests.get(self.OASIS_BASE, params=params, timeout=timeout)

                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        self.logger.warning(f"Rate limited, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise requests.RequestException(f"Rate limited after {max_retries} attempts")

                if response.status_code != 200:
                    raise requests.RequestException(f"API returned status {response.status_code}")

                return response

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.warning(f"Request failed: {e}, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e

        raise requests.RequestException(f"Failed after {max_retries} attempts")

    def _parse_oasis_response(self, content: bytes) -> pd.DataFrame:
        """Parse OASIS API response (XML or zipped XML)"""
        # Check if response is zipped
        if content[:2] == b'PK':
            z = zipfile.ZipFile(io.BytesIO(content))
            files = z.namelist()
            if not files:
                raise ValueError("Empty zip file returned from CAISO API")

            # Check for error files
            if any('INVALID' in f.upper() or 'ERROR' in f.upper() for f in files):
                error_file = next(f for f in files if 'INVALID' in f.upper() or 'ERROR' in f.upper())
                error_data = z.read(error_file)
                error_text = error_data.decode('utf-8')
                raise ValueError(f"CAISO API error: {error_text[:500]}")

            data = z.read(files[0])
        else:
            data = content

        # Parse XML
        try:
            txt = data.decode('utf-8')
        except:
            txt = data.decode('latin1')

        if not txt:
            return pd.DataFrame()

        # Parse XML structure
        try:
            root = ET.fromstring(txt)
        except:
            idx = txt.find('<')
            if idx >= 0:
                root = ET.fromstring(txt[idx:])
            else:
                return pd.DataFrame()

        rows = []
        for elem in root.iter():
            tag = elem.tag.lower()
            if any(keyword in tag for keyword in ['item', 'row', 'result', 'data', 'lmp', 'price']):
                row = {}
                for child in list(elem):
                    key = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    val = child.text.strip() if child.text else None
                    row[key] = val
                if row:
                    rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Find and parse timestamp columns
        ts_cols = [c for c in df.columns if any(word in c.lower() for word in ['date', 'time', 'interval', 'start'])]
        if ts_cols:
            for c in ts_cols:
                try:
                    df['timestamp'] = pd.to_datetime(df[c])
                    break
                except:
                    continue

        # Find and parse price columns
        price_cols = [c for c in df.columns if any(word in c.lower() for word in ['value', 'price', 'lmp'])]
        if price_cols:
            col = price_cols[0]
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.rename(columns={col: 'price_mwh'})

        return df

    def fetch_da_prices(self, zone: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch Day-Ahead Market prices"""
        node_id = self._resolve_node_id(zone)

        params = {
            "queryname": "PRC_LMP",
            "market_run_id": "DAM",
            "node": node_id,
            "startdatetime": self._format_dt_for_oasis(start_dt),
            "enddatetime": self._format_dt_for_oasis(end_dt),
            "version": "1"
        }

        self.logger.info(f"Fetching DA prices for {zone} ({node_id})")

        try:
            response = self._make_oasis_request(params)
            df = self._parse_oasis_response(response.content)

            if df.empty:
                self.logger.warning(f"No DA data returned for {zone}")
                return pd.DataFrame()

            if 'timestamp' in df.columns:
                df = df.set_index('timestamp').sort_index()
                df = df[~df.index.duplicated(keep='first')]

            if 'price_mwh' in df.columns:
                return df[['price_mwh']].rename(columns={'price_mwh': 'da_price_mwh'})

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error fetching DA prices for {zone}: {e}")
            return pd.DataFrame()

    def fetch_rt_prices(self, zone: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch Real-Time Market prices"""
        node_id = self._resolve_node_id(zone)

        params = {
            "queryname": "PRC_INTVL_LMP",
            "market_run_id": "RTM",
            "node": node_id,
            "startdatetime": self._format_dt_for_oasis(start_dt),
            "enddatetime": self._format_dt_for_oasis(end_dt),
            "version": "1"
        }

        self.logger.info(f"Fetching RT prices for {zone} ({node_id})")

        try:
            response = self._make_oasis_request(params)
            df = self._parse_oasis_response(response.content)

            if df.empty:
                self.logger.warning(f"No RT data returned for {zone}")
                return pd.DataFrame()

            if 'timestamp' in df.columns:
                df = df.set_index('timestamp').sort_index()
                df = df[~df.index.duplicated(keep='first')]

            if 'price_mwh' in df.columns:
                return df[['price_mwh']].rename(columns={'price_mwh': 'rt_price_mwh'})

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error fetching RT prices for {zone}: {e}")
            return pd.DataFrame()

# ===========================
# Weather and AQI Data Fetching
# ===========================
class WeatherDataFetcher:
    """Fetches weather and air quality data from Open-Meteo API"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Setup Open-Meteo client
        cache_config = config.get('api_config', {}).get('cache', {})
        if cache_config.get('enabled', True):
            cache_session = requests_cache.CachedSession(
                cache_config.get('directory', '.cache'),
                expire_after=cache_config.get('expire_after', 3600)
            )
        else:
            cache_session = requests.Session()

        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

    def fetch_weather_data(self, location: dict, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch weather data for a location"""
        weather_features = self.config.get('weather_features', [])

        if not weather_features:
            return pd.DataFrame()

        self.logger.info(f"Fetching weather data for {location['name']}")

        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": location['latitude'],
            "longitude": location['longitude'],
            "hourly": weather_features,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "timezone": self.config.get('time_parameters', {}).get('timezone', 'auto')
        }

        try:
            responses = self.openmeteo.weather_api(weather_url, params=params)
            response = responses[0]

            hourly = response.Hourly()
            hourly_data = {
                "datetime": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }

            for i, var in enumerate(weather_features):
                try:
                    hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
                except:
                    self.logger.warning(f"Could not fetch {var}")
                    hourly_data[var] = None

            df = pd.DataFrame(data=hourly_data)
            self.logger.info(f"Fetched {len(df)} weather records")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching weather data: {e}")
            return pd.DataFrame()

    def fetch_air_quality_data(self, location: dict, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch air quality data for a location"""
        aq_features = self.config.get('air_quality_features', [])

        if not aq_features:
            return pd.DataFrame()

        self.logger.info(f"Fetching air quality data for {location['name']}")

        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": location['latitude'],
            "longitude": location['longitude'],
            "hourly": aq_features,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "timezone": self.config.get('time_parameters', {}).get('timezone', 'auto')
        }

        try:
            responses = self.openmeteo.weather_api(aq_url, params=params)
            response = responses[0]

            hourly = response.Hourly()
            hourly_data = {
                "datetime": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }

            for i, var in enumerate(aq_features):
                try:
                    hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
                except:
                    self.logger.warning(f"Could not fetch {var}")
                    hourly_data[var] = None

            df = pd.DataFrame(data=hourly_data)
            self.logger.info(f"Fetched {len(df)} air quality records")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching air quality data: {e}")
            return pd.DataFrame()

    def calculate_aqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate AQI from PM2.5 and PM10"""
        if not self.config.get('processing', {}).get('calculate_aqi', True):
            return df

        if 'pm2_5' not in df.columns and 'pm10' not in df.columns:
            return df

        def calculate_simple_aqi(row):
            if pd.isna(row.get('pm2_5')) and pd.isna(row.get('pm10')):
                return None

            aqi_values = []

            # PM2.5 AQI calculation
            if not pd.isna(row.get('pm2_5')):
                pm25 = row['pm2_5']
                if pm25 <= 12:
                    aqi_pm25 = (50/12) * pm25
                elif pm25 <= 35.4:
                    aqi_pm25 = 50 + ((100-50)/(35.4-12)) * (pm25-12)
                elif pm25 <= 55.4:
                    aqi_pm25 = 100 + ((150-100)/(55.4-35.4)) * (pm25-35.4)
                elif pm25 <= 150.4:
                    aqi_pm25 = 150 + ((200-150)/(150.4-55.4)) * (pm25-55.4)
                elif pm25 <= 250.4:
                    aqi_pm25 = 200 + ((300-200)/(250.4-150.4)) * (pm25-150.4)
                else:
                    aqi_pm25 = 300 + ((500-300)/(500.4-250.4)) * (pm25-250.4)
                aqi_values.append(aqi_pm25)

            # PM10 AQI calculation
            if not pd.isna(row.get('pm10')):
                pm10 = row['pm10']
                if pm10 <= 54:
                    aqi_pm10 = (50/54) * pm10
                elif pm10 <= 154:
                    aqi_pm10 = 50 + ((100-50)/(154-54)) * (pm10-54)
                elif pm10 <= 254:
                    aqi_pm10 = 100 + ((150-100)/(254-154)) * (pm10-154)
                elif pm10 <= 354:
                    aqi_pm10 = 150 + ((200-150)/(354-254)) * (pm10-254)
                elif pm10 <= 424:
                    aqi_pm10 = 200 + ((300-200)/(424-354)) * (pm10-354)
                else:
                    aqi_pm10 = 300 + ((500-300)/(604-424)) * (pm10-424)
                aqi_values.append(aqi_pm10)

            return max(aqi_values) if aqi_values else None

        df['calculated_aqi'] = df.apply(calculate_simple_aqi, axis=1)
        return df

# ===========================
# Main Data Fetcher
# ===========================
class CombinedDataFetcher:
    """Main class to orchestrate data fetching"""

    def __init__(self, config_path: str):
        """Initialize with configuration file"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.logger = setup_logging(self.config)
        self.caiso_fetcher = CAISODataFetcher(self.config, self.logger)
        self.weather_fetcher = WeatherDataFetcher(self.config, self.logger)

        # Create output directory if needed
        output_dir = self.config.get('output', {}).get('directory', './data')
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def get_time_range(self) -> tuple:
        """Get start and end datetime from configuration"""
        time_params = self.config.get('time_parameters', {})

        if time_params.get('end_date'):
            end_dt = pd.to_datetime(time_params['end_date'])
        else:
            # Use yesterday as end date to avoid future date issues with weather API
            end_dt = datetime.now() - timedelta(days=1)

        if time_params.get('start_date'):
            start_dt = pd.to_datetime(time_params['start_date'])
        else:
            duration_days = time_params.get('duration_days', 100)
            print('Duration days: ', duration_days)
            start_dt = end_dt - timedelta(days=duration_days)

        # Ensure times are at hour boundaries
        start_dt = start_dt.replace(minute=0, second=0, microsecond=0)
        end_dt = end_dt.replace(minute=0, second=0, microsecond=0)

        return start_dt, end_dt

    def process_location(self, location: dict) -> Optional[pd.DataFrame]:
        """Process a single location"""
        self.logger.info(f"\nProcessing: {location['name']}")
        self.logger.info(f"Location: {location['county']}, {location['state']}")
        self.logger.info(f"Coordinates: ({location['latitude']:.4f}, {location['longitude']:.4f})")

        start_dt, end_dt = self.get_time_range()
        self.logger.info(f"Date range: {start_dt} to {end_dt}")

        # Fetch weather data
        weather_df = self.weather_fetcher.fetch_weather_data(location, start_dt, end_dt)

        # Fetch air quality data
        aq_df = self.weather_fetcher.fetch_air_quality_data(location, start_dt, end_dt)

        # Merge weather and AQ data
        if not weather_df.empty and not aq_df.empty:
            # Ensure both dataframes have the same timezone handling
            if 'datetime' in weather_df.columns:
                weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.tz_localize(None)
            if 'datetime' in aq_df.columns:
                aq_df['datetime'] = pd.to_datetime(aq_df['datetime']).dt.tz_localize(None)
            combined_df = pd.merge(weather_df, aq_df, on='datetime', how='outer')
        elif not weather_df.empty:
            if 'datetime' in weather_df.columns:
                weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.tz_localize(None)
            combined_df = weather_df
        elif not aq_df.empty:
            if 'datetime' in aq_df.columns:
                aq_df['datetime'] = pd.to_datetime(aq_df['datetime']).dt.tz_localize(None)
            combined_df = aq_df
        else:
            self.logger.warning(f"No weather or AQ data for {location['name']}")
            combined_df = pd.DataFrame()

        # Fetch CAISO data if applicable
        caiso_params = self.config.get('caiso_parameters', {})
        if (caiso_params.get('enabled', True) and
            location.get('state') == 'CA' and
            location.get('caiso_zone')):

            zone = location['caiso_zone']

            # Fetch DA prices
            if 'DAM' in caiso_params.get('markets', []):
                da_df = self.caiso_fetcher.fetch_da_prices(zone, start_dt, end_dt)
                if not da_df.empty:
                    # Resample to hourly if needed
                    da_df = da_df.resample('1h').mean()
                    da_df.index = da_df.index.tz_localize(None)
                    combined_df = pd.merge(
                        combined_df,
                        da_df,
                        left_on='datetime',
                        right_index=True,
                        how='left'
                    )

            # Fetch RT prices
            if 'RTM' in caiso_params.get('markets', []):
                rt_df = self.caiso_fetcher.fetch_rt_prices(zone, start_dt, end_dt)
                if not rt_df.empty:
                    # Resample based on time interval
                    interval = self.config.get('time_parameters', {}).get('time_interval', 'hourly')
                    if interval == 'hourly':
                        rt_df = rt_df.resample('1h').mean()
                    elif interval == '5min':
                        rt_df = rt_df.resample('5min').mean()
                    elif interval == '15min':
                        rt_df = rt_df.resample('15min').mean()
                    elif interval == '30min':
                        rt_df = rt_df.resample('30min').mean()

                    rt_df.index = rt_df.index.tz_localize(None)
                    combined_df = pd.merge(
                        combined_df,
                        rt_df,
                        left_on='datetime',
                        right_index=True,
                        how='left'
                    )

        if combined_df.empty:
            return None

        # Add location metadata
        combined_df['location_name'] = location['name']
        combined_df['operator'] = location.get('operator', '')
        combined_df['capacity_mw'] = location.get('capacity_mw', 0.0)
        combined_df['county'] = location['county']
        combined_df['state'] = location['state']
        combined_df['grid'] = location.get('grid', '')
        combined_df['latitude'] = location['latitude']
        combined_df['longitude'] = location['longitude']

        # Calculate AQI if requested
        combined_df = self.weather_fetcher.calculate_aqi(combined_df)

        # Sort by datetime
        combined_df = combined_df.sort_values('datetime')

        # Reorder columns for readability
        priority_cols = [
            'datetime', 'location_name', 'operator', 'capacity_mw',
            'county', 'state', 'grid', 'latitude', 'longitude',
            'calculated_aqi', 'pm10', 'pm2_5',
            'da_price_mwh', 'rt_price_mwh',
            'temperature_2m', 'relative_humidity_2m',
            'wind_speed_10m', 'precipitation', 'pressure_msl'
        ]

        existing_priority = [col for col in priority_cols if col in combined_df.columns]
        remaining_cols = [col for col in combined_df.columns if col not in existing_priority]
        combined_df = combined_df[existing_priority + remaining_cols]

        return combined_df

    def save_data(self, df: pd.DataFrame, location: dict, interval: str = 'hourly'):
        """Save data to CSV file"""
        output_config = self.config.get('output', {})
        output_dir = Path(output_config.get('directory', './data'))

        # Create filename
        start_dt, end_dt = self.get_time_range()
        safe_name = location['name'].replace(' ', '_').replace(',', '').replace('&', 'and')
        safe_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe_name)

        pattern = output_config.get('filename_pattern', '{location}_{interval}_{start_date}_to_{end_date}.csv')
        filename = pattern.format(
            location=safe_name,
            interval=interval,
            start_date=start_dt.strftime('%Y%m%d'),
            end_date=end_dt.strftime('%Y%m%d')
        )

        filepath = output_dir / filename

        # Apply column filters
        include_cols = output_config.get('include_columns', [])
        exclude_cols = output_config.get('exclude_columns', [])

        if include_cols:
            df = df[[col for col in include_cols if col in df.columns]]
        if exclude_cols:
            df = df[[col for col in df.columns if col not in exclude_cols]]

        # Save to CSV
        csv_options = output_config.get('csv_options', {})
        df.to_csv(
            filepath,
            index=csv_options.get('index', False),
            compression=csv_options.get('compression', None)
        )

        self.logger.info(f"Saved {len(df)} records to {filepath}")

        # Create daily aggregation if requested
        if output_config.get('create_daily_aggregation', True) and interval == 'hourly':
            daily_df = self.create_daily_aggregation(df)
            if not daily_df.empty:
                daily_filename = filename.replace('_hourly_', '_daily_')
                daily_filepath = output_dir / daily_filename
                daily_df.to_csv(
                    daily_filepath,
                    index=csv_options.get('index', False),
                    compression=csv_options.get('compression', None)
                )
                self.logger.info(f"Saved {len(daily_df)} daily records to {daily_filepath}")

        # Print summary statistics
        self.print_summary(df, location['name'])

    def create_daily_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create daily aggregated data from hourly"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['datetime']).dt.date

        # Columns to keep as-is
        keep_cols = ['location_name', 'operator', 'capacity_mw', 'county', 'state', 'grid', 'latitude', 'longitude']

        # Aggregation rules
        agg_dict = {}
        for col in df.columns:
            if col not in ['datetime', 'date'] + keep_cols:
                if col in ['precipitation', 'rain']:
                    agg_dict[col] = 'sum'  # Sum for precipitation
                else:
                    agg_dict[col] = 'mean'  # Mean for everything else

        daily_df = df.groupby('date').agg(agg_dict).reset_index()

        # Add back non-aggregated columns
        for col in keep_cols:
            if col in df.columns:
                daily_df[col] = df[col].iloc[0]

        return daily_df

    def print_summary(self, df: pd.DataFrame, location_name: str):
        """Print summary statistics"""
        self.logger.info(f"\nSummary for {location_name}:")
        self.logger.info("-" * 40)

        if 'calculated_aqi' in df.columns and not df['calculated_aqi'].isna().all():
            self.logger.info(f"AQI - Avg: {df['calculated_aqi'].mean():.1f}, "
                           f"Min: {df['calculated_aqi'].min():.1f}, "
                           f"Max: {df['calculated_aqi'].max():.1f}")

        if 'da_price_mwh' in df.columns and not df['da_price_mwh'].isna().all():
            self.logger.info(f"DA Price - Avg: ${df['da_price_mwh'].mean():.2f}/MWh, "
                           f"Min: ${df['da_price_mwh'].min():.2f}/MWh, "
                           f"Max: ${df['da_price_mwh'].max():.2f}/MWh")

        if 'rt_price_mwh' in df.columns and not df['rt_price_mwh'].isna().all():
            self.logger.info(f"RT Price - Avg: ${df['rt_price_mwh'].mean():.2f}/MWh, "
                           f"Min: ${df['rt_price_mwh'].min():.2f}/MWh, "
                           f"Max: ${df['rt_price_mwh'].max():.2f}/MWh")

        if 'temperature_2m' in df.columns and not df['temperature_2m'].isna().all():
            self.logger.info(f"Temperature - Avg: {df['temperature_2m'].mean():.1f}°C, "
                           f"Min: {df['temperature_2m'].min():.1f}°C, "
                           f"Max: {df['temperature_2m'].max():.1f}°C")

    def run(self, location_filter: Optional[str] = None):
        """Run the data fetcher for all configured locations"""
        locations = self.config.get('locations', [])

        if location_filter:
            locations = [loc for loc in locations if location_filter.lower() in loc['name'].lower()]

        if not locations:
            self.logger.error("No locations to process")
            return

        self.logger.info(f"Processing {len(locations)} locations...")

        run_config = self.config.get('run', {})
        continue_on_error = run_config.get('continue_on_error', True)

        successful = []
        failed = []

        for location in locations:
            try:
                df = self.process_location(location)
                if df is not None and not df.empty:
                    interval = self.config.get('time_parameters', {}).get('time_interval', 'hourly')
                    self.save_data(df, location, interval)
                    successful.append(location['name'])
                else:
                    self.logger.warning(f"No data retrieved for {location['name']}")
                    failed.append(location['name'])
            except Exception as e:
                self.logger.error(f"Error processing {location['name']}: {e}")
                failed.append(location['name'])
                if not continue_on_error:
                    break

        # Final report
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Successfully processed: {len(successful)} locations")
        if successful:
            for loc in successful:
                self.logger.info(f"  ✓ {loc}")

        if failed:
            self.logger.info(f"\nFailed to process: {len(failed)} locations")
            for loc in failed:
                self.logger.info(f"  ✗ {loc}")

# ===========================
# CLI Entry Point
# ===========================
def main():
    parser = argparse.ArgumentParser(description='Combined Power Grid and Weather Data Fetcher')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--location',
        type=str,
        help='Filter to process only locations containing this string'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without fetching data'
    )

    args = parser.parse_args()

    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)

    try:
        # Create fetcher and run
        fetcher = CombinedDataFetcher(args.config)

        if args.dry_run:
            fetcher.logger.info("Dry run mode - configuration validated successfully")
        else:
            fetcher.run(location_filter=args.location)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()