#!/usr/bin/env python3
"""
CAISO SP15 Price Data Fetcher for Eland Solar & Storage Center

Fetches Day-Ahead hourly and Real-Time 5-minute LMP prices from CAISO OASIS API
for the SP15 zone where Eland Solar & Storage Center is located.

This version is config-driven - all parameters loaded from training_config.yaml

Usage:
    python caiso_sp15_data_fetch.py --config config/training_config.yaml
    python caiso_sp15_data_fetch.py --config config/training_config.yaml --start-date 2025-08-01 --end-date 2025-08-31

Output:
    - CSV files with Day-Ahead and Real-Time prices
    - Combined hourly price data
"""

import os
import io
import zipfile
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import time
import yaml
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ----------------------------
# Configuration Loading
# ----------------------------

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def chunk_date_range(start_dt: datetime, end_dt: datetime, max_days: int = 30) -> List[Tuple[datetime, datetime]]:
    """
    Split a date range into chunks of max_days or less.

    Args:
        start_dt: Start date
        end_dt: End date
        max_days: Maximum days per chunk (default: 30 for CAISO limit of 31)

    Returns:
        List of (start, end) datetime tuples
    """
    chunks = []
    current_start = start_dt

    while current_start < end_dt:
        # Calculate chunk end (max_days from current_start or end_dt, whichever is earlier)
        current_end = min(current_start + timedelta(days=max_days), end_dt)
        chunks.append((current_start, current_end))

        # Move to next chunk (add 1 day to avoid overlap)
        current_start = current_end + timedelta(days=1)

    return chunks


# ----------------------------
# Helper Functions
# ----------------------------

def _format_dt_for_oasis(dt: datetime) -> str:
    """Format datetime for CAISO OASIS API (UTC timezone)"""
    return dt.strftime("%Y%m%dT%H:%M-0000")


def _make_oasis_request(url: str, params: dict, max_retries: int = 3,
                        retry_delay: int = 2, timeout: int = 60) -> requests.Response:
    """
    Make a request to CAISO OASIS API with retry logic for rate limiting

    Args:
        url: API endpoint URL
        params: Query parameters
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay in seconds between retries
        timeout: Request timeout in seconds
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)

            # Handle rate limiting
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * retry_delay  # Exponential backoff
                    print(f"Rate limited, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise requests.RequestException(f"Rate limited after {max_retries} attempts")

            # Handle other HTTP errors
            if response.status_code != 200:
                raise requests.RequestException(f"API returned status {response.status_code}: {response.text}")

            return response

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1
                print(f"Request failed: {e}, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                raise e

    raise requests.RequestException(f"Failed after {max_retries} attempts")


def _parse_oasis_xml(payload_bytes: bytes) -> pd.DataFrame:
    """Parse CAISO OASIS XML response to extract price data"""
    try:
        txt = payload_bytes.decode('utf-8')
    except:
        try:
            txt = payload_bytes.decode('latin1')
        except:
            txt = None

    if not txt:
        return pd.DataFrame()

    # Parse XML
    try:
        root = ET.fromstring(txt)
    except Exception:
        # Sometimes OASIS wraps XML inside other text
        idx = txt.find('<')
        if idx >= 0:
            try:
                root = ET.fromstring(txt[idx:])
            except Exception:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    rows = []

    # Look for data items in the XML structure
    for elem in root.iter():
        tag = elem.tag.lower()
        # Check for various CAISO data row patterns
        if (tag.endswith('item') or tag.endswith('row') or tag.endswith('result') or
            tag.endswith('rowdata') or tag.endswith('data') or
            'lmp' in tag or 'price' in tag):
            # Collect children into dict
            row = {}
            for child in list(elem):
                # Strip XML namespace from tag names
                key = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                val = child.text.strip() if child.text else None
                row[key] = val
            if row:
                rows.append(row)

    # If no rows found, try to find any elements with timestamp and price data
    if not rows:
        for elem in root.iter():
            if list(elem):  # Has children
                row = {}
                has_timestamp = False
                has_price = False
                for child in list(elem):
                    key = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    val = child.text.strip() if child.text else None
                    row[key] = val
                    # Check if this looks like timestamp or price data
                    if any(ts_word in key.lower() for ts_word in ['time', 'date', 'interval', 'start']):
                        has_timestamp = True
                    if any(price_word in key.lower() for price_word in ['price', 'lmp', 'value', 'cost']):
                        has_price = True

                # Only include rows that have both timestamp and price data
                if has_timestamp and has_price and row:
                    rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Find timestamp columns
    preferred_ts_cols = [c for c in df.columns if c.upper() in ['INTERVAL_START_GMT', 'INTERVAL_START', 'INTERVALSTART']]
    ts_cols = preferred_ts_cols or [c for c in df.columns if any(ts_word in c.lower() for ts_word in ['date', 'time', 'interval', 'start'])]
    if ts_cols:
        for c in ts_cols:
            try:
                df['timestamp'] = pd.to_datetime(df[c])
                break
            except:
                continue

    # Find price columns
    val_cols = [c for c in df.columns if any(price_word in c.lower() for price_word in ['value', 'price', 'lmp', 'cost'])]
    if val_cols:
        col = val_cols[0]
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'price_mwh' not in df.columns:
            df = df.rename(columns={col: 'price_mwh'})

    return df


# ----------------------------
# Data Fetching Functions
# ----------------------------

def fetch_da_prices(node_id: str, start_dt: datetime, end_dt: datetime,
                    oasis_base_url: str, max_retries: int = 3,
                    retry_delay: int = 2, timeout: int = 60) -> pd.DataFrame:
    """
    Fetch Day-Ahead hourly LMP prices from CAISO OASIS API

    Args:
        node_id: CAISO node identifier
        start_dt: Start datetime
        end_dt: End datetime
        oasis_base_url: CAISO OASIS API base URL
        max_retries: Maximum retry attempts
        retry_delay: Retry delay in seconds
        timeout: Request timeout in seconds
    """
    params = {
        "queryname": "PRC_LMP",
        "market_run_id": "DAM",
        "node": node_id,
        "startdatetime": _format_dt_for_oasis(start_dt),
        "enddatetime": _format_dt_for_oasis(end_dt),
        "version": "1"
    }

    print(f"Fetching Day-Ahead prices for {node_id} ({start_dt.date()} to {end_dt.date()})...")
    response = _make_oasis_request(oasis_base_url, params, max_retries, retry_delay, timeout)

    # Process response (usually a ZIP file)
    content = response.content
    if content[:2] == b'PK':  # ZIP file
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

    df = _parse_oasis_xml(data)

    if df.empty:
        raise ValueError("No data returned from CAISO API")

    # Set timestamp as index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

        # Ensure price_mwh is numeric
        if 'price_mwh' in df.columns:
            df['price_mwh'] = pd.to_numeric(df['price_mwh'], errors='coerce')

            # Drop rows with NaN prices
            df = df.dropna(subset=['price_mwh'])

            # Sort and remove duplicates
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]

            # Resample to hourly (only numeric columns)
            df_numeric = df[['price_mwh']]
            df_hourly = df_numeric.resample('h').mean()

            return df_hourly
        else:
            raise ValueError(f"No price column found. Available columns: {list(df.columns)}")
    else:
        raise ValueError(f"No timestamp column found. Available columns: {list(df.columns)}")


def fetch_rt_prices(node_id: str, start_dt: datetime, end_dt: datetime,
                    oasis_base_url: str, max_retries: int = 3,
                    retry_delay: int = 2, timeout: int = 60) -> pd.DataFrame:
    """
    Fetch Real-Time 5-minute LMP prices from CAISO OASIS API

    Args:
        node_id: CAISO node identifier
        start_dt: Start datetime
        end_dt: End datetime
        oasis_base_url: CAISO OASIS API base URL
        max_retries: Maximum retry attempts
        retry_delay: Retry delay in seconds
        timeout: Request timeout in seconds
    """
    params = {
        "queryname": "PRC_INTVL_LMP",
        "market_run_id": "RTM",
        "node": node_id,
        "startdatetime": _format_dt_for_oasis(start_dt),
        "enddatetime": _format_dt_for_oasis(end_dt),
        "version": "1"
    }

    print(f"Fetching Real-Time prices for {node_id} ({start_dt.date()} to {end_dt.date()})...")
    response = _make_oasis_request(oasis_base_url, params, max_retries, retry_delay, timeout)

    # Process response
    content = response.content
    if content[:2] == b'PK':  # ZIP file
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

    df = _parse_oasis_xml(data)

    if df.empty:
        raise ValueError("No data returned from CAISO API")

    # Set timestamp as index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

        # Ensure price_mwh is numeric
        if 'price_mwh' in df.columns:
            df['price_mwh'] = pd.to_numeric(df['price_mwh'], errors='coerce')

            # Drop rows with NaN prices
            df = df.dropna(subset=['price_mwh'])

            # Sort and remove duplicates
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]

            # Keep only price column
            return df[['price_mwh']]
        else:
            raise ValueError(f"No price column found. Available columns: {list(df.columns)}")
    else:
        raise ValueError(f"No timestamp column found. Available columns: {list(df.columns)}")


def fetch_prices_with_chunking(node_id: str, start_dt: datetime, end_dt: datetime,
                               fetch_func, oasis_base_url: str, max_days: int = 30,
                               max_retries: int = 3, retry_delay: int = 2,
                               timeout: int = 60, rate_limit_delay: int = 1) -> pd.DataFrame:
    """
    Fetch prices with automatic chunking for date ranges > max_days

    Args:
        node_id: CAISO node identifier
        start_dt: Start datetime
        end_dt: End datetime
        fetch_func: Function to fetch prices (fetch_da_prices or fetch_rt_prices)
        oasis_base_url: CAISO OASIS API base URL
        max_days: Maximum days per chunk
        max_retries: Maximum retry attempts
        retry_delay: Retry delay in seconds
        timeout: Request timeout in seconds
        rate_limit_delay: Delay between chunks in seconds
    """
    # Calculate number of days
    total_days = (end_dt - start_dt).days

    if total_days <= max_days:
        # Single request
        return fetch_func(node_id, start_dt, end_dt, oasis_base_url, max_retries, retry_delay, timeout)

    # Multiple chunks required
    chunks = chunk_date_range(start_dt, end_dt, max_days)
    print(f"Date range spans {total_days} days - splitting into {len(chunks)} chunks")

    all_data = []

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print(f"\nFetching chunk {i}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}")

        try:
            chunk_data = fetch_func(node_id, chunk_start, chunk_end, oasis_base_url,
                                   max_retries, retry_delay, timeout)
            all_data.append(chunk_data)

            # Rate limiting between chunks
            if i < len(chunks):
                print(f"Waiting {rate_limit_delay} seconds before next chunk...")
                time.sleep(rate_limit_delay)

        except Exception as e:
            print(f"Error fetching chunk {i}: {e}")
            raise

    # Concatenate all chunks
    print(f"\nConcatenating {len(all_data)} chunks...")
    combined_df = pd.concat(all_data, axis=0)

    # Remove any duplicate timestamps at chunk boundaries
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()

    print(f"Combined dataset: {len(combined_df)} total records")

    return combined_df


# ----------------------------
# Main Execution
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Fetch CAISO price data from OASIS API')
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
    caiso_config = config['data_collection']['caiso']
    date_range = config['date_range']
    paths = config['paths']

    # Use command-line args if provided, otherwise use config
    start_date = args.start_date if args.start_date else date_range['start_date']
    end_date = args.end_date if args.end_date else date_range['end_date']
    output_dir = args.output_dir if args.output_dir else paths['raw_dir']

    # Parse dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # OASIS API URL
    oasis_base = "https://oasis.caiso.com/oasisapi/SingleZip"

    # Display configuration
    print("=" * 70)
    print("CAISO Price Data Fetcher (Config-Driven)")
    print("=" * 70)
    print(f"Site: {site_info['name']}")
    print(f"Owner: {site_info['owner']}")
    print(f"Capacity: {site_info['capacity_mw']} MW")
    print(f"Location: {site_info['county']}, {site_info['state']}")
    print(f"CAISO Zone: {site_info['caiso_zone']}")
    print(f"Node: {site_info['caiso_node']}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Auto-chunking: {'Enabled' if caiso_config['auto_chunk'] else 'Disabled'}")
    print(f"Max retries: {caiso_config['max_retries']}")
    print("-" * 70)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Fetch Day-Ahead prices (with chunking if needed)
        if caiso_config['fetch_day_ahead']:
            print("\n" + "=" * 70)
            print("Fetching Day-Ahead Prices")
            print("=" * 70)

            if caiso_config['auto_chunk']:
                da_prices = fetch_prices_with_chunking(
                    node_id=site_info['caiso_node'],
                    start_dt=start_dt,
                    end_dt=end_dt,
                    fetch_func=fetch_da_prices,
                    oasis_base_url=oasis_base,
                    max_days=caiso_config['max_days_per_chunk'],
                    max_retries=caiso_config['max_retries'],
                    retry_delay=caiso_config['retry_delay_seconds'],
                    timeout=caiso_config['timeout_seconds'],
                    rate_limit_delay=caiso_config['rate_limit_delay_seconds']
                )
            else:
                da_prices = fetch_da_prices(
                    node_id=site_info['caiso_node'],
                    start_dt=start_dt,
                    end_dt=end_dt,
                    oasis_base_url=oasis_base,
                    max_retries=caiso_config['max_retries'],
                    retry_delay=caiso_config['retry_delay_seconds'],
                    timeout=caiso_config['timeout_seconds']
                )

            da_filename = f"{output_dir}/caiso_da_prices_{start_date}_{end_date}.csv"
            da_prices.to_csv(da_filename)
            print(f"\n✓ Day-Ahead prices saved to: {da_filename}")
            print(f"  Records: {len(da_prices)}")
            print(f"  Avg price: ${da_prices['price_mwh'].mean():.2f}/MWh")
            print(f"  Min price: ${da_prices['price_mwh'].min():.2f}/MWh")
            print(f"  Max price: ${da_prices['price_mwh'].max():.2f}/MWh")

        # Fetch Real-Time prices (with chunking if needed)
        if caiso_config['fetch_real_time']:
            print("\n" + "=" * 70)
            print("Fetching Real-Time Prices")
            print("=" * 70)

            if caiso_config['auto_chunk']:
                rt_prices = fetch_prices_with_chunking(
                    node_id=site_info['caiso_node'],
                    start_dt=start_dt,
                    end_dt=end_dt,
                    fetch_func=fetch_rt_prices,
                    oasis_base_url=oasis_base,
                    max_days=caiso_config['max_days_per_chunk'],
                    max_retries=caiso_config['max_retries'],
                    retry_delay=caiso_config['retry_delay_seconds'],
                    timeout=caiso_config['timeout_seconds'],
                    rate_limit_delay=caiso_config['rate_limit_delay_seconds']
                )
            else:
                rt_prices = fetch_rt_prices(
                    node_id=site_info['caiso_node'],
                    start_dt=start_dt,
                    end_dt=end_dt,
                    oasis_base_url=oasis_base,
                    max_retries=caiso_config['max_retries'],
                    retry_delay=caiso_config['retry_delay_seconds'],
                    timeout=caiso_config['timeout_seconds']
                )

            rt_filename = f"{output_dir}/caiso_rt_prices_{start_date}_{end_date}.csv"
            rt_prices.to_csv(rt_filename)
            print(f"\n✓ Real-Time prices saved to: {rt_filename}")
            print(f"  Records: {len(rt_prices)}")
            print(f"  Avg price: ${rt_prices['price_mwh'].mean():.2f}/MWh")
            print(f"  Min price: ${rt_prices['price_mwh'].min():.2f}/MWh")
            print(f"  Max price: ${rt_prices['price_mwh'].max():.2f}/MWh")

        # Create combined hourly dataset
        if caiso_config['fetch_day_ahead'] and caiso_config['fetch_real_time']:
            print("\n" + "=" * 70)
            print("Creating Combined Dataset")
            print("=" * 70)

            combined_df = pd.DataFrame({
                'da_price_mwh': da_prices['price_mwh'],
                'rt_price_mwh': rt_prices['price_mwh'].resample('h').mean()
            })

            # Add price spread and metadata
            combined_df['price_spread_mwh'] = combined_df['rt_price_mwh'] - combined_df['da_price_mwh']
            combined_df['site'] = site_info['name']
            combined_df['zone'] = site_info['caiso_zone']
            combined_df['node'] = site_info['caiso_node']

            combined_filename = f"{output_dir}/caiso_combined_prices_{start_date}_{end_date}.csv"
            combined_df.to_csv(combined_filename)
            print(f"\n✓ Combined hourly prices saved to: {combined_filename}")
            print(f"  Avg spread (RT-DA): ${combined_df['price_spread_mwh'].mean():.2f}/MWh")
            print(f"  Spread volatility: ${combined_df['price_spread_mwh'].std():.2f}/MWh")

        print("\n" + "=" * 70)
        print("Data fetch completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError fetching data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
