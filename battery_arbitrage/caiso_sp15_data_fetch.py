#!/usr/bin/env python3
"""
CAISO SP15 Price Data Fetcher for Eland Solar & Storage Center

Fetches Day-Ahead hourly and Real-Time 5-minute LMP prices from CAISO OASIS API
for the SP15 zone where Eland Solar & Storage Center is located.

Location: Eland Solar & Storage Center, Phase 2
Owner: Avantus
Capacity: 200.0 MW
COD: 01/07/2025
County: Kern, CA
Zone: SP15 (LDWP area)

Usage:
    python caiso_sp15_data_fetch.py

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
from datetime import datetime, timedelta

# ----------------------------
# Configuration
# ----------------------------

# Eland Solar & Storage Center Configuration
SITE_INFO = {
    'name': 'Eland Solar & Storage Center, Phase 2',
    'owner': 'Avantus',
    'capacity_mw': 200.0,
    'cod': '01/07/2025',
    'county': 'Kern',
    'state': 'CA',
    'utility': 'LDWP',
    'zone': 'SP15',
    'node': 'TH_SP15_GEN-APND'  # SP15 Trading Hub node for CAISO OASIS
}

# CAISO OASIS API Configuration
OASIS_BASE = "https://oasis.caiso.com/oasisapi/SingleZip"

# Date range for data fetch
# Note: CAISO OASIS limits requests to 31 days maximum
START_DATE = '2025-08-01'
END_DATE = '2025-08-31'  # Changed to 31 days (Aug has 31 days)

# Convert to datetime objects
start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')

# ----------------------------
# Helper Functions
# ----------------------------

def _format_dt_for_oasis(dt: datetime):
    """Format datetime for CAISO OASIS API (UTC timezone)"""
    return dt.strftime("%Y%m%dT%H:%M-0000")

def _make_oasis_request(url: str, params: dict, max_retries: int = 3) -> requests.Response:
    """Make a request to CAISO OASIS API with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=60)

            # Handle rate limiting
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff
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

def fetch_da_prices(node_id: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch Day-Ahead hourly LMP prices from CAISO OASIS API"""
    params = {
        "queryname": "PRC_LMP",
        "market_run_id": "DAM",
        "node": node_id,
        "startdatetime": _format_dt_for_oasis(start_dt),
        "enddatetime": _format_dt_for_oasis(end_dt),
        "version": "1"
    }

    print(f"Fetching Day-Ahead prices for {node_id}...")
    response = _make_oasis_request(OASIS_BASE, params)

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

def fetch_rt_prices(node_id: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch Real-Time 5-minute LMP prices from CAISO OASIS API"""
    params = {
        "queryname": "PRC_INTVL_LMP",
        "market_run_id": "RTM",
        "node": node_id,
        "startdatetime": _format_dt_for_oasis(start_dt),
        "enddatetime": _format_dt_for_oasis(end_dt),
        "version": "1"
    }

    print(f"Fetching Real-Time prices for {node_id}...")
    response = _make_oasis_request(OASIS_BASE, params)

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

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CAISO Price Data Fetcher for Eland Solar & Storage Center")
    print("=" * 60)
    print(f"Site: {SITE_INFO['name']}")
    print(f"Owner: {SITE_INFO['owner']}")
    print(f"Capacity: {SITE_INFO['capacity_mw']} MW")
    print(f"Location: {SITE_INFO['county']}, {SITE_INFO['state']}")
    print(f"CAISO Zone: {SITE_INFO['zone']}")
    print(f"Node: {SITE_INFO['node']}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print("-" * 60)

    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Fetch Day-Ahead prices
        da_prices = fetch_da_prices(SITE_INFO['node'], start_dt, end_dt)
        da_filename = f"data/eland_sp15_da_prices_{START_DATE}_{END_DATE}.csv"
        da_prices.to_csv(da_filename)
        print(f"✓ Day-Ahead prices saved to: {da_filename}")
        print(f"  Records: {len(da_prices)}")
        print(f"  Avg price: ${da_prices['price_mwh'].mean():.2f}/MWh")
        print(f"  Min price: ${da_prices['price_mwh'].min():.2f}/MWh")
        print(f"  Max price: ${da_prices['price_mwh'].max():.2f}/MWh")

        # Fetch Real-Time prices
        rt_prices = fetch_rt_prices(SITE_INFO['node'], start_dt, end_dt)
        rt_filename = f"data/eland_sp15_rt_prices_{START_DATE}_{END_DATE}.csv"
        rt_prices.to_csv(rt_filename)
        print(f"\n✓ Real-Time prices saved to: {rt_filename}")
        print(f"  Records: {len(rt_prices)}")
        print(f"  Avg price: ${rt_prices['price_mwh'].mean():.2f}/MWh")
        print(f"  Min price: ${rt_prices['price_mwh'].min():.2f}/MWh")
        print(f"  Max price: ${rt_prices['price_mwh'].max():.2f}/MWh")

        # Create combined hourly dataset
        combined_df = pd.DataFrame({
            'da_price_mwh': da_prices['price_mwh'],
            'rt_price_mwh': rt_prices['price_mwh'].resample('h').mean()
        })

        # Add price spread and metadata
        combined_df['price_spread_mwh'] = combined_df['rt_price_mwh'] - combined_df['da_price_mwh']
        combined_df['site'] = SITE_INFO['name']
        combined_df['zone'] = SITE_INFO['zone']
        combined_df['node'] = SITE_INFO['node']

        combined_filename = f"data/eland_sp15_combined_prices_{START_DATE}_{END_DATE}.csv"
        combined_df.to_csv(combined_filename)
        print(f"\n✓ Combined hourly prices saved to: {combined_filename}")
        print(f"  Avg spread (RT-DA): ${combined_df['price_spread_mwh'].mean():.2f}/MWh")
        print(f"  Spread volatility: ${combined_df['price_spread_mwh'].std():.2f}/MWh")

        print("\n" + "=" * 60)
        print("Data fetch completed successfully!")

    except Exception as e:
        print(f"\nError fetching data: {e}")
        import traceback
        traceback.print_exc()