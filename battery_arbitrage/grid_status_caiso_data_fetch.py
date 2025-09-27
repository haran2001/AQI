from gridstatusio import GridStatusClient
from load_dotenv import load_dotenv
import os

load_dotenv()

# CAISO Zones and Node Mappings
CAISO_ZONES = {
    'NP15': 'TH_NP15_GEN-APND',  # Northern California Trading Hub
    'SP15': 'TH_SP15_GEN-APND',  # Southern California Trading Hub
    'ZP26': 'TH_ZP26_GEN-APND'   # Zone P26 Trading Hub
}

# CAISO Market Types
MARKETS = {
    'DAM': 'Day-Ahead Market',
    'RTM': 'Real-Time Market (5-min intervals)'
}

GRID_STATUS_API_KEY = os.getenv('GRID_STATUS_API_KEY')
client = GridStatusClient(api_key=GRID_STATUS_API_KEY)

QUERY_LIMIT = 10_000
start_date = '2025-08-25'
end_date = '2025-09-25'

# Note: CAISO uses zones rather than specific lat/lon
# Eland Solar location (Kern County) is in SP15 zone
selected_zone = 'SP15'

# Fetch CAISO data (not ERCOT)
# GridStatus.io datasets for CAISO include:
# - caiso_lmp_day_ahead_hourly
# - caiso_lmp_real_time_5min
data_utc = client.get_dataset(
    dataset="caiso_lmp_day_ahead_hourly",  # Changed from ercot to caiso
    start=start_date,
    end=end_date,
    limit=QUERY_LIMIT,
    # Filter for SP15 zone if possible
    # node=CAISO_ZONES[selected_zone]  # Uncomment if GridStatus.io supports node filtering
)

df_name = f"{start_date}_{end_date}_gridstatus_caiso_{selected_zone}_data.csv"
data_utc.to_csv('data/' + df_name, index=False)

print(f"Fetched CAISO data for zone: {selected_zone}")
print(f"Zone node identifier: {CAISO_ZONES[selected_zone]}")
print(f"Data saved to: data/{df_name}")