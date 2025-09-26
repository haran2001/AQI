#!/usr/bin/env python3
"""
caiso_bess_backtester.py

Fetch CAISO OASIS Day-Ahead hourly LMPs and Real-Time 5-min LMPs (OASIS),
map DA -> RT (1 hour -> 12 x 5-min), run DA optimizer and RT controller (plug-ins),
and compute two-settlement accounting.

Defaults:
 - Zones: NP15, SP15, ZP26 (edit ZONES list to add more)
 - Lookback: last 2 days (configurable)
 - RT interval: 5 minutes

Usage:
    python caiso_bess_backtester.py

Dependencies:
    pip install requests pandas numpy matplotlib

Plugin hooks:
 - implement my_da_optimizer(prices_da: pd.Series, timestamps_da: pd.DatetimeIndex, bp: dict) -> np.ndarray (kW per DA hour)
 - implement my_rt_controller(t_rt, soc_kwh, price_history_rt, price_forecast_rt, bp, rt_params, scheduled_power_kw) -> desired power kW

Author: adapted for you
"""
import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
from datetime import datetime, timedelta

# ----------------------------
# Config / defaults
# ----------------------------
OASIS_BASE = "https://oasis.caiso.com/oasisapi/SingleZip"  # CAISO OASIS API base
ZONES = ["NP15", "SP15", "ZP26"]   # default set; add others if you want
LOOKBACK_DAYS = 2                  # last 2 days of data (configurable)
RT_INTERVAL_MIN = 5                # 5-minute RT
# Battery defaults (example)
BATTERY_PARAMS_DEFAULT = {
    'capacity_kwh': 1000.0,
    'p_charge_kw': 250.0,
    'p_discharge_kw': 250.0,
    'eff_roundtrip': 0.9,
    'initial_soc_kwh': 500.0,
    'soc_min_kwh': 0.0,
    'soc_max_kwh': 1000.0,
    'degradation_cost_per_mwh': 0.0
}

# ----------------------------
# Helpers: OASIS query builders
# ----------------------------
# Map common CAISO zone names to representative node identifiers acceptable to OASIS
# Users may also pass explicit node ids directly (e.g., 'NP15_7_N001') in ZONES
ZONE_TO_NODE = {
    # Map zones to Trading Hub node identifiers (commonly used in OASIS PRC_* reports)
    'NP15': 'TH_NP15_GEN-APND',
    'SP15': 'TH_SP15_GEN-APND',
    'ZP26': 'TH_ZP26_GEN-APND'
}

def _resolve_node_id(zone_or_node: str) -> str:
    """
    Return a node identifier for a given input. If input already looks like a
    node id (contains an underscore), return as-is. Otherwise, map common zone
    to a default node id using ZONE_TO_NODE.
    """
    if '_' in zone_or_node:
        return zone_or_node
    return ZONE_TO_NODE.get(zone_or_node.upper(), zone_or_node)
def _format_dt_for_oasis(dt: datetime):
    # OASIS expects YYYYMMDDThh:00-0000 style (UTC timezone)
    # Convert to UTC and format properly
    return dt.strftime("%Y%m%dT%H:%M-0000")

def _make_oasis_request(url: str, params: dict, max_retries: int = 3) -> requests.Response:
    """
    Make a request to CAISO OASIS API with retry logic for rate limiting
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            
            # Handle rate limiting
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
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
                wait_time = (2 ** attempt) * 1  # Shorter backoff for other errors
                print(f"Request failed: {e}, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                raise e
    
    raise requests.RequestException(f"Failed after {max_retries} attempts")

def _prepare_rt_series(df_rt: pd.DataFrame, expected_minutes: int = RT_INTERVAL_MIN) -> pd.DataFrame:
    """
    Clean and regularize RT 5-min series without altering fetch logic:
      - sort index, drop exact duplicate timestamps
      - if spacing is irregular or zero, aggregate by timestamp first
      - if final spacing != expected, resample to expected interval using mean
    Returns DataFrame with a clean DateTimeIndex at expected cadence.
    """
    if df_rt.empty:
        return df_rt
    df = df_rt.copy().sort_index()
    # drop duplicate timestamps keeping first
    df = df[~df.index.duplicated(keep='first')]
    # ensure numeric
    if 'price_mwh' in df.columns:
        df['price_mwh'] = pd.to_numeric(df['price_mwh'], errors='coerce')
    # if still duplicates exist (different tz normalization), group by index
    df = df.groupby(level=0).mean(numeric_only=True)
    # compute min positive delta
    if len(df.index) >= 2:
        diffs = pd.Series(df.index[1:]) - pd.Series(df.index[:-1])
        pos_diffs = [d for d in diffs if pd.notnull(d) and d.total_seconds() > 0]
        min_delta_min = int(min(pos_diffs).total_seconds() / 60) if pos_diffs else 0
    else:
        min_delta_min = 0
    if min_delta_min != expected_minutes and expected_minutes > 0:
        # resample to expected cadence using mean to preserve price characteristics
        rule = f"{expected_minutes}min"
        df = df.resample(rule).mean()
    return df

def _prepare_da_series(df_da: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DA hourly series:
      - sort index, drop duplicate timestamps
      - coerce numeric price
      - resample to hourly mean to ensure strict 1-hour cadence
    """
    if df_da.empty:
        return df_da
    df = df_da.copy().sort_index()
    df = df[~df.index.duplicated(keep='first')]
    if 'price_mwh' in df.columns:
        df['price_mwh'] = pd.to_numeric(df['price_mwh'], errors='coerce')
    df = df.groupby(level=0).mean(numeric_only=True)
    # resample to hourly to normalize
    df = df.resample('1h').mean()
    return df

# ----------------------------
# Public helpers: fetch current DA/RT price series (no backtest)
# ----------------------------
def get_caiso_prices(
    zones: list,
    da_start: datetime,
    da_end: datetime,
    rt_start: datetime,
    rt_end: datetime,
) -> dict:
    """
    Fetch CAISO Day-Ahead hourly and Real-Time interval prices for the given zones (or node ids) over
    provided windows. Returns a dict per zone with raw (lightly cleaned) DataFrames:
      {
        zone: {
          'da_df': DataFrame[price_mwh] indexed by timestamp,
          'rt_df': DataFrame[price_mwh] indexed by timestamp,
          'node_id': resolved_node_id
        }
      }
    Cleaning applied: sort_index, drop duplicate timestamps, type coercion. No imputation.
    """
    out: dict = {}
    for z in zones:
        node_id = _resolve_node_id(z)
        try:
            da_df = fetch_oasis_da_zone_lmp(node_id, da_start, da_end)
            rt_df = fetch_oasis_rt_5min_zone_lmp(node_id, rt_start, rt_end)
        except Exception as e:
            out[z] = {'error': str(e), 'node_id': node_id}
            continue
        # light cleaning only
        if not da_df.empty:
            da_df = da_df.sort_index()
            da_df = da_df[~da_df.index.duplicated(keep='first')]
            if 'price_mwh' in da_df.columns:
                da_df['price_mwh'] = pd.to_numeric(da_df['price_mwh'], errors='coerce')
        if not rt_df.empty:
            rt_df = rt_df.sort_index()
            rt_df = rt_df[~rt_df.index.duplicated(keep='first')]
            if 'price_mwh' in rt_df.columns:
                rt_df['price_mwh'] = pd.to_numeric(rt_df['price_mwh'], errors='coerce')
        out[z] = {'da_df': da_df, 'rt_df': rt_df, 'node_id': node_id}
    return out

def get_recent_caiso_prices(
    zones: list = None,
    da_lookback_hours: int = 24,
    rt_lookback_hours: int = 6
) -> dict:
    """
    Convenience wrapper to fetch recent DA and RT prices ending at current hour (UTC aligned):
      - DA window: now-DA_lookback_hours -> now
      - RT window: now-RT_lookback_hours -> now
    """
    zones = zones or ZONES
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    da_start = now - timedelta(hours=da_lookback_hours)
    da_end = now
    rt_start = now - timedelta(hours=rt_lookback_hours)
    rt_end = now
    return get_caiso_prices(zones, da_start, da_end, rt_start, rt_end)

def fetch_oasis_da_zone_lmp(zone: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch Day-Ahead hourly LMPs for a CAISO zone via OASIS API (queryname=PRC_LMP, market_run_id=DAM)
    Returns DataFrame indexed by timestamp (naive datetimes in US/Pacific local time assumed by OASIS).
    """
    node_id = _resolve_node_id(zone)
    params = {
        "queryname": "PRC_LMP",
        "market_run_id": "DAM",
        "node": node_id,  # node identifier (zone mapped if needed)
        "startdatetime": _format_dt_for_oasis(start_dt),
        "enddatetime": _format_dt_for_oasis(end_dt),
        "version": "1"  # Added version parameter
    }
    url = OASIS_BASE
    r = _make_oasis_request(url, params)
    
    # CAISO returns zipped XML for some query types; try to detect
    content = r.content
    # If response is a zip file
    if content[:2] == b'PK':
        z = zipfile.ZipFile(io.BytesIO(content))
        # find first file with .xml or .txt
        files = z.namelist()
        if not files:
            raise ValueError("Empty zip file returned from CAISO API")
        
        # Check for error files
        if any('INVALID' in f.upper() or 'ERROR' in f.upper() for f in files):
            # Read the error file to get details
            error_file = next(f for f in files if 'INVALID' in f.upper() or 'ERROR' in f.upper())
            error_data = z.read(error_file)
            error_text = error_data.decode('utf-8')
            raise ValueError(f"CAISO API returned error: {error_text[:500]}")
        
        name = files[0]
        data = z.read(name)
    else:
        data = content
    
    # parse XML or CSV-ish structure robustly
    df = _parse_oasis_generic(data)
    
    if df.empty:
        raise ValueError("No data returned from CAISO API")
    
    # filter for zone location and DA if necessary; keep columns timestamp and price_mwh
    # CAISO PRC_LMP returns prices in $/MWh often under an element like 'value' or 'price'
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = _prepare_da_series(df)
        # regularize RT series
        # Clean without imputing: preserve only exact reported intervals
        df = _prepare_rt_series(df, expected_minutes=RT_INTERVAL_MIN)
    # try to find price column and normalize name
    price_col = next((c for c in df.columns if c.lower() in ['price','value','lmp','price_mw','price_mwh','lmp_price']), None)
    if price_col:
        df = df.rename(columns={price_col: 'price_mwh'})
    else:
        raise ValueError(f"No price column found in CAISO data. Available columns: {list(df.columns)}")
    
    return df[['price_mwh']].copy()

def fetch_oasis_rt_5min_zone_lmp(zone: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch RT interval LMPs (5-min) for a zone via OASIS API (queryname=PRC_INTVL_LMP, market_run_id=RTM).
    Note: OASIS can return 5-min intervals; we request RTM and interval type.
    """
    node_id = _resolve_node_id(zone)
    params = {
        "queryname": "PRC_INTVL_LMP",
        "market_run_id": "RTM",
        "node": node_id,  # node identifier (zone mapped if needed)
        "startdatetime": _format_dt_for_oasis(start_dt),
        "enddatetime": _format_dt_for_oasis(end_dt),
        "version": "1"  # Added version parameter
    }
    url = OASIS_BASE
    r = _make_oasis_request(url, params)
    
    content = r.content
    if content[:2] == b'PK':
        z = zipfile.ZipFile(io.BytesIO(content))
        files = z.namelist()
        if not files:
            raise ValueError("Empty zip file returned from CAISO API")
        
        # Check for error files
        if any('INVALID' in f.upper() or 'ERROR' in f.upper() for f in files):
            # Read the error file to get details
            error_file = next(f for f in files if 'INVALID' in f.upper() or 'ERROR' in f.upper())
            error_data = z.read(error_file)
            error_text = error_data.decode('utf-8')
            raise ValueError(f"CAISO API returned error: {error_text[:500]}")
        
        name = files[0]
        data = z.read(name)
    else:
        data = content
    
    df = _parse_oasis_generic(data)
    
    if df.empty:
        raise ValueError("No data returned from CAISO API")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
    price_col = next((c for c in df.columns if c.lower() in ['price','value','lmp','price_mw','price_mwh','lmp_price']), None)
    if price_col:
        df = df.rename(columns={price_col: 'price_mwh'})
    else:
        raise ValueError(f"No price column found in CAISO data. Available columns: {list(df.columns)}")
    
    return df[['price_mwh']].copy()

def _parse_oasis_generic(payload_bytes: bytes) -> pd.DataFrame:
    """
    Generic OASIS parser: attempts to parse zipped XML or raw XML as returned by OASIS.
    CAISO OASIS responses vary; we attempt a flexible parse:
      - If XML: find items and extract child tags as columns
      - If CSV-like: try to decode as text and read with pandas.read_csv
    Returns a DataFrame.
    """
    try:
        txt = payload_bytes.decode('utf-8')
    except:
        try:
            txt = payload_bytes.decode('latin1')
        except:
            txt = None
    if not txt:
        return pd.DataFrame()
    
    # quick heuristic: is this XML?
    if txt.strip().startswith('<?xml') or '<Results' in txt or '<DATA' in txt or '<OASIS' in txt.upper():
        # parse XML
        try:
            root = ET.fromstring(txt)
        except Exception:
            # sometimes OASIS wraps XML inside other text; try to find first '<' position
            idx = txt.find('<')
            if idx >= 0:
                try:
                    root = ET.fromstring(txt[idx:])
                except Exception:
                    # fallback
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
        
        # CAISO-specific parsing: look for data items in the XML structure
        rows = []
        
        # Try different XML structures that CAISO might use
        # Look for elements that contain data rows
        for elem in root.iter():
            tag = elem.tag.lower()
            # Check for various CAISO data row patterns
            if (tag.endswith('item') or tag.endswith('row') or tag.endswith('result') or 
                tag.endswith('rowdata') or tag.endswith('data') or 
                'lmp' in tag or 'price' in tag):
                # collect children into dict
                row = {}
                for child in list(elem):
                    # strip XML namespace from tag names for easier matching
                    key = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    val = child.text
                    # sanitize
                    if val is not None:
                        val = val.strip()
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
                        if any(ts_word in key.lower() for ts_word in ['time', 'date', 'interval', 'start', 'end']):
                            has_timestamp = True
                        if any(price_word in key.lower() for price_word in ['price', 'lmp', 'value', 'cost']):
                            has_price = True
                    
                    # Only include rows that have both timestamp and price data
                    if has_timestamp and has_price and row:
                        rows.append(row)
        
        if not rows:
            # try CSV fallback
            return _parse_oasis_csv_fallback(txt)
        
        df = pd.DataFrame(rows)
        
        # some fields like 'IntervalStart' or 'DateTime' or 'TIME' indicate timestamps - try to normalize
        # Prefer explicit interval start fields if present
        preferred_ts_cols = [c for c in df.columns if c.upper() in ['INTERVAL_START_GMT', 'INTERVAL_START', 'INTERVALSTART']]
        ts_cols = preferred_ts_cols or [c for c in df.columns if any(ts_word in c.lower() for ts_word in ['date', 'time', 'interval', 'start', 'end'])]
        if ts_cols:
            # pick the first candidate and try parse
            for c in ts_cols:
                try:
                    df['timestamp'] = pd.to_datetime(df[c])
                    break
                except Exception:
                    continue
        
        # find numeric 'value' column and convert
        # Treat VALUE as the numeric price field for PRC_* reports
        val_cols = [c for c in df.columns if any(price_word in c.lower() for price_word in ['value', 'price', 'lmp', 'intervalprice', 'price_mw', 'price_mwh', 'cost'])]
        if val_cols:
            col = val_cols[0]
            # coerce numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # normalize column name to price_mwh
            if 'price_mwh' not in df.columns:
                df = df.rename(columns={col: 'price_mwh'})
        
        return df
    else:
        # assume CSV-like
        return _parse_oasis_csv_fallback(txt)

def _parse_oasis_csv_fallback(txt: str) -> pd.DataFrame:
    """Try reading as CSV/TSV via pandas"""
    try:
        df = pd.read_csv(io.StringIO(txt))
        return df
    except Exception:
        # try stripping leading lines until header found
        lines = txt.splitlines()
        for i in range(min(10, len(lines))):
            try:
                test = '\n'.join(lines[i:])
                df = pd.read_csv(io.StringIO(test))
                return df
            except Exception:
                continue
    # give up
    return pd.DataFrame()

# ----------------------------
# Battery model & sim (same as earlier)
# ----------------------------
def battery_step(soc_kwh, power_kw, dt_h, params):
    cap = params['capacity_kwh']
    p_charge_max = params['p_charge_kw']
    p_discharge_max = params['p_discharge_kw']
    eff_c = params.get('eff_charge', np.sqrt(params.get('eff_roundtrip', 0.9)))
    eff_d = params.get('eff_discharge', np.sqrt(params.get('eff_roundtrip', 0.9)))
    soc_min = params.get('soc_min_kwh', 0.0)
    soc_max = params.get('soc_max_kwh', cap)
    if power_kw >= 0:
        p_exec = min(power_kw, p_discharge_max)
    else:
        p_exec = max(power_kw, -p_charge_max)
    if p_exec >= 0:
        energy_delivered = p_exec * dt_h
        energy_removed_from_soc = energy_delivered / eff_d
        if soc_kwh - energy_removed_from_soc < soc_min - 1e-9:
            energy_removed_from_soc = soc_kwh - soc_min
            energy_delivered = energy_removed_from_soc * eff_d
            p_exec = energy_delivered / dt_h if dt_h>0 else 0.0
        energy_charged = 0.0
        energy_discharged = energy_delivered
        new_soc = soc_kwh - energy_removed_from_soc
    else:
        energy_into_batt = (-p_exec) * dt_h * eff_c
        if soc_kwh + energy_into_batt > soc_max + 1e-9:
            energy_into_batt = soc_max - soc_kwh
            p_exec = - (energy_into_batt / eff_c) / dt_h if dt_h>0 else 0.0
        energy_charged = energy_into_batt
        energy_discharged = 0.0
        new_soc = soc_kwh + energy_into_batt
    new_soc = max(min(new_soc, soc_max), soc_min)
    return new_soc, energy_charged, energy_discharged, p_exec

# ----------------------------
# DA optimizer & RT controller hooks (replace with your algorithms)
# ----------------------------
def my_da_optimizer(prices_da: pd.Series, timestamps_da: pd.DatetimeIndex, bp: dict) -> np.ndarray:
    """
    Example placeholder DA optimizer.
    Replace this with your DA optimization algo (use prices_da values in $/MWh).
    Must return an array of length == len(prices_da) of scheduled kW values (positive=discharge).
    """
    # simple threshold example (charge on low, discharge on high)
    p_kW = np.zeros(len(prices_da))
    prices_kwh = prices_da.values / 1000.0
    low = np.percentile(prices_kwh, 25)
    high = np.percentile(prices_kwh, 75)
    for i, p in enumerate(prices_kwh):
        if p < low:
            p_kW[i] = -bp['p_charge_kw']
        elif p > high:
            p_kW[i] = bp['p_discharge_kw']
        else:
            p_kW[i] = 0.0
    return p_kW

def my_rt_controller(t_rt: int, soc_kwh: float, price_history_rt: np.ndarray, price_forecast_rt: np.ndarray, bp: dict, rt_params: dict, scheduled_power_kw: float) -> float:
    """
    Example RT controller placeholder. Replace with your RT control algorithm.
    Should return desired power (kW) for RT step t_rt.
    """
    # default behaviour: follow DA schedule exactly
    return float(scheduled_power_kw)

# ----------------------------
# Orchestration: map DA->RT and run backtest for zones
# ----------------------------
def run_caiso_backtest(zones=None, lookback_days=2, bp=None):
    zones = zones or ZONES
    bp = bp or BATTERY_PARAMS_DEFAULT
    now = datetime.utcnow()
    # CAISO OASIS expects UTC times in the API requests
    end_dt = now.replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=lookback_days)
    results = {}
    for zone in zones:
        node_id = _resolve_node_id(zone)
        if node_id == zone and '_' not in node_id:
            print(f"\n--- Zone {zone} --- (no explicit node mapping found; attempting as provided) fetching data {start_dt} -> {end_dt}")
        else:
            print(f"\n--- Zone {zone} (node {node_id}) --- fetching data {start_dt} -> {end_dt}")
        try:
            da_df = fetch_oasis_da_zone_lmp(node_id, start_dt, end_dt)
            rt_df = fetch_oasis_rt_5min_zone_lmp(node_id, start_dt, end_dt)
        except Exception as e:
            print("  fetch error:", e)
            continue
        if da_df.empty or rt_df.empty:
            print("  no data for zone in this window; skipping")
            continue
        # resample/validate RT index: ensure 5-min spacing
        rt_df = rt_df.sort_index()
        # compute RT minutes spacing (guard small/empty cases)
        if len(rt_df.index) >= 2:
            rt_delta = int(max(1, round((rt_df.index[1] - rt_df.index[0]).total_seconds() / 60)))
            if rt_delta != RT_INTERVAL_MIN:
                print(f"  warning: RT spacing {rt_delta} min != expected {RT_INTERVAL_MIN} min")
        else:
            rt_delta = RT_INTERVAL_MIN
        # map DA hours to RT steps: compute steps per DA hour from prepared RT cadence
        steps_per_hour = int(60 // RT_INTERVAL_MIN) if RT_INTERVAL_MIN > 0 else 12
        # build scheduled DA (call user's DA optimizer)
        prices_da_series = da_df['price_mwh']
        da_index = prices_da_series.index
        scheduled_da_kw = my_da_optimizer(prices_da_series, da_index, bp)
        if len(scheduled_da_kw) != len(da_index):
            raise ValueError("DA optimizer must return same length as DA hours")
        # repeat each DA hour into RT steps
        scheduled_rt_kw = np.repeat(scheduled_da_kw, steps_per_hour)
        # align scheduled_rt_kw length to rt_df
        if len(scheduled_rt_kw) < len(rt_df):
            # pad with last value
            scheduled_rt_kw = np.pad(scheduled_rt_kw, (0, len(rt_df)-len(scheduled_rt_kw)), 'edge')
        elif len(scheduled_rt_kw) > len(rt_df):
            scheduled_rt_kw = scheduled_rt_kw[:len(rt_df)]
        # Ensure indices are unique after cleaning
        rt_df = rt_df[~rt_df.index.duplicated(keep='first')]
        da_df = da_df[~da_df.index.duplicated(keep='first')]
        # Now run RT loop
        soc = bp['initial_soc_kwh']
        dt_h = RT_INTERVAL_MIN / 60.0
        recs = []
        da_rev_total = 0.0
        # DA settlement: energy = scheduled_kw * 1 hour for each DA interval
        for i, ts in enumerate(da_index):
            p_sched = scheduled_da_kw[i]
            da_val = da_df.loc[ts,'price_mwh']
            # Skip intervals without a reported DA price (no imputation)
            if pd.isna(da_val):
                continue
            da_price = da_val / 1000.0
            da_rev_total += p_sched * 1.0 * da_price  # kW * 1h * $/kWh
        total_rt_dev_rev = 0.0
        total_deg_cost = 0.0
        cum_throughput_kwh = 0.0
        price_history = []
        rt_index = rt_df.index
        for t_idx, ts in enumerate(rt_index):
            rt_val = rt_df.loc[ts,'price_mwh']
            rt_price = (rt_val / 1000.0) if pd.notna(rt_val) else np.nan
            price_history.append(rt_price if pd.notna(rt_price) else 0.0)
            scheduled_kw = float(scheduled_rt_kw[t_idx])
            desired_kw = my_rt_controller(t_idx, soc, np.array(price_history), np.array([]), bp, {}, scheduled_kw)
            new_soc, e_ch, e_dis, p_exec = battery_step(soc, desired_kw, dt_h, bp)
            deviation_kw = p_exec - scheduled_kw
            # If RT price is missing, exclude this interval from revenue (no imputation)
            rt_dev_rev = 0.0 if pd.isna(rt_price) else deviation_kw * dt_h * rt_price
            total_rt_dev_rev += rt_dev_rev
            throughput = e_ch + e_dis
            cum_throughput_kwh += throughput
            deg_cost = bp.get('degradation_cost_per_mwh', 0.0) * (throughput/1000.0)
            total_deg_cost += deg_cost
            recs.append({
                'timestamp': ts,
                'rt_price_mwh': rt_df.loc[ts,'price_mwh'],
                'rt_price_kwh': rt_price,
                'scheduled_kw': scheduled_kw,
                'desired_kw': desired_kw,
                'executed_kw': p_exec,
                'deviation_kw': deviation_kw,
                'energy_charged_kwh': e_ch,
                'energy_discharged_kwh': e_dis,
                'soc_before_kwh': soc,
                'soc_after_kwh': new_soc,
                'rt_dev_rev_$': rt_dev_rev,
                'deg_cost_$': deg_cost
            })
            soc = new_soc
        df_rt = pd.DataFrame(recs).set_index('timestamp')
        total_profit = da_rev_total + total_rt_dev_rev - total_deg_cost
        summary = {
            'zone': zone,
            'node_id': node_id,
            'da_rev_total_$': da_rev_total,
            'rt_dev_rev_total_$': total_rt_dev_rev,
            'deg_cost_total_$': total_deg_cost,
            'total_profit_$': total_profit,
            'total_throughput_kwh': cum_throughput_kwh,
            'final_soc_kwh': soc,
            'n_da_steps': len(da_index),
            'n_rt_steps': len(rt_index)
        }
        # save artifacts
        fname = f"caiso_backtest_{zone}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
        df_rt.to_csv(fname)
        print(f"Saved {fname}  | profit ${total_profit:.2f}")
        # quick plots
        try:
            plt.figure(figsize=(10,3))
            da_df['price_mwh'].plot(title=f"{zone} DA Price ($/MWh)")
            plt.tight_layout(); plt.savefig(f"{zone}_da_price.png"); plt.close()
            plt.figure(figsize=(10,3))
            rt_df['price_mwh'].plot(title=f"{zone} RT Price ($/MWh)")
            plt.tight_layout(); plt.savefig(f"{zone}_rt_price.png"); plt.close()
            plt.figure(figsize=(10,3))
            df_rt['soc_after_kwh'].plot(title=f"{zone} SOC (kWh)"); plt.tight_layout(); plt.savefig(f"{zone}_soc.png"); plt.close()
        except Exception as e:
            print("  plotting error:", e)
        results[zone] = {'df_rt': df_rt, 'summary': summary, 'da_df': da_df, 'rt_df': rt_df}
    return results

# ----------------------------
# Run if main
# ----------------------------
if __name__ == "__main__":
    print("Running CAISO BESS backtester (5-min RT, 2-day window) for zones:", ZONES)
    res = run_caiso_backtest(zones=ZONES, lookback_days=LOOKBACK_DAYS, bp=BATTERY_PARAMS_DEFAULT)
    for z, r in res.items():
        print(z, r['summary'])
    print("Done. CSVs and PNGs saved in current directory.")