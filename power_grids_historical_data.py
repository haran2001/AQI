import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import sys

# Power grid locations with coordinates
POWER_GRID_LOCATIONS = [
    {
        "name": "Eland Solar & Storage Center, Phase 2 Hy",
        "operator": "Avantus",
        "capacity_mw": 200.0,
        "county": "Kern",
        "state": "CA",
        "grid": "LDWP",
        "latitude": 35.3733,  # Kern County, CA
        "longitude": -119.0187
    },
    {
        "name": "Bakeoven Solar",
        "operator": "Avangrid Renewables",
        "capacity_mw": 60.0,
        "county": "Wasco",
        "state": "OR",
        "grid": "AVRN",
        "latitude": 45.2588,  # Wasco County, OR
        "longitude": -121.0949
    },
    {
        "name": "Corpus Refinery",
        "operator": "Flint Hills Resources",
        "capacity_mw": 27.5,
        "county": "Nueces",
        "state": "TX",
        "grid": "ERCO",
        "latitude": 27.8006,  # Corpus Christi, Nueces County, TX
        "longitude": -97.3964
    },
    {
        "name": "CTGB Maloney and Webster",
        "operator": "TotalEnergies",
        "capacity_mw": 2.0,
        "county": "Tolland",
        "state": "CT",
        "grid": "ISNE",
        "latitude": 41.8722,  # Tolland County, CT
        "longitude": -72.3695
    },
    {
        "name": "CSU Northridge Plant",
        "operator": "Unable to find",
        "capacity_mw": 0.8,
        "county": "Los Angeles",
        "state": "CA",
        "grid": "LDWP",
        "latitude": 34.2381,  # CSU Northridge, Los Angeles, CA
        "longitude": -118.5276
    }
]

def fetch_location_historical_data(location, days_back=365):
    """
    Fetch historical AQI and weather data for a specific location from Open-Meteo API
    """

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Location coordinates
    lat = location["latitude"]
    lon = location["longitude"]
    location_name = location["name"]

    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    print(f"\nFetching data for {location_name}")
    print(f"Location: {location['county']}, {location['state']}")
    print(f"Coordinates: ({lat:.4f}, {lon:.4f})")
    print(f"Date range: {start_date} to {end_date}")
    print("-" * 60)

    # Fetch Air Quality Data
    print("Fetching air quality data...")
    air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    air_quality_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                   "sulphur_dioxide", "ozone", "dust", "uv_index", "uv_index_clear_sky",
                   "ammonia", "alder_pollen", "birch_pollen", "grass_pollen",
                   "mugwort_pollen", "olive_pollen", "ragweed_pollen"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "auto"
    }

    try:
        air_responses = openmeteo.weather_api(air_quality_url, params=air_quality_params)
        air_response = air_responses[0]

        # Process air quality hourly data
        air_hourly = air_response.Hourly()
        air_hourly_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(air_hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(air_hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=air_hourly.Interval()),
                inclusive="left"
            )
        }

        # Add all air quality variables
        air_variables = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                        "sulphur_dioxide", "ozone", "dust", "uv_index", "uv_index_clear_sky",
                        "ammonia", "alder_pollen", "birch_pollen", "grass_pollen",
                        "mugwort_pollen", "olive_pollen", "ragweed_pollen"]

        for i, var in enumerate(air_variables):
            try:
                air_hourly_data[var] = air_hourly.Variables(i).ValuesAsNumpy()
            except:
                print(f"Warning: Could not fetch {var}")
                air_hourly_data[var] = None

        air_quality_df = pd.DataFrame(data=air_hourly_data)
        print(f"✓ Air quality data fetched: {len(air_quality_df)} hourly records")

    except Exception as e:
        print(f"Error fetching air quality data: {e}")
        air_quality_df = pd.DataFrame()

    # Fetch Weather Data
    print("Fetching weather data...")
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                   "apparent_temperature", "precipitation", "rain", "snowfall",
                   "weather_code", "pressure_msl", "surface_pressure",
                   "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                   "evapotranspiration", "et0_fao_evapotranspiration",
                   "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
                   "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m",
                   "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "auto"
    }

    try:
        weather_responses = openmeteo.weather_api(weather_url, params=weather_params)
        weather_response = weather_responses[0]

        # Process weather hourly data
        weather_hourly = weather_response.Hourly()
        weather_hourly_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(weather_hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(weather_hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=weather_hourly.Interval()),
                inclusive="left"
            )
        }

        # Add all weather variables
        weather_variables = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                           "apparent_temperature", "precipitation", "rain", "snowfall",
                           "weather_code", "pressure_msl", "surface_pressure",
                           "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                           "evapotranspiration", "et0_fao_evapotranspiration",
                           "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
                           "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m",
                           "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm"]

        for i, var in enumerate(weather_variables):
            try:
                weather_hourly_data[var] = weather_hourly.Variables(i).ValuesAsNumpy()
            except:
                print(f"Warning: Could not fetch {var}")
                weather_hourly_data[var] = None

        weather_df = pd.DataFrame(data=weather_hourly_data)
        print(f"✓ Weather data fetched: {len(weather_df)} hourly records")

    except Exception as e:
        print(f"Error fetching weather data: {e}")
        weather_df = pd.DataFrame()

    # Merge air quality and weather data
    print("Merging datasets...")
    if not air_quality_df.empty and not weather_df.empty:
        # Merge on datetime
        combined_df = pd.merge(air_quality_df, weather_df, on='datetime', how='outer')
    elif not air_quality_df.empty:
        combined_df = air_quality_df
    elif not weather_df.empty:
        combined_df = weather_df
    else:
        print("No data to save")
        return None

    # Add location and power grid information
    combined_df['location_name'] = location_name
    combined_df['operator'] = location["operator"]
    combined_df['capacity_mw'] = location["capacity_mw"]
    combined_df['county'] = location["county"]
    combined_df['state'] = location["state"]
    combined_df['grid'] = location["grid"]
    combined_df['latitude'] = lat
    combined_df['longitude'] = lon

    # Sort by datetime
    combined_df = combined_df.sort_values('datetime')

    # Calculate AQI based on PM2.5 and PM10 (simplified version)
    def calculate_simple_aqi(row):
        if pd.isna(row['pm2_5']) and pd.isna(row['pm10']):
            return None

        aqi_values = []

        # PM2.5 AQI calculation (simplified)
        if not pd.isna(row['pm2_5']):
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

        # PM10 AQI calculation (simplified)
        if not pd.isna(row['pm10']):
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

        # Return the maximum AQI value
        return max(aqi_values) if aqi_values else None

    combined_df['calculated_aqi'] = combined_df.apply(calculate_simple_aqi, axis=1)

    # Reorder columns for better readability
    column_order = ['datetime', 'location_name', 'operator', 'capacity_mw',
                   'county', 'state', 'grid', 'latitude', 'longitude', 'calculated_aqi',
                   'pm10', 'pm2_5', 'temperature_2m', 'relative_humidity_2m',
                   'wind_speed_10m', 'precipitation', 'pressure_msl']

    # Add remaining columns
    remaining_cols = [col for col in combined_df.columns if col not in column_order]
    column_order.extend(remaining_cols)

    # Reorder columns (only include columns that exist)
    existing_columns = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[existing_columns]

    return combined_df

def create_safe_filename(location_name):
    """Create a safe filename from location name"""
    # Replace special characters with underscores
    safe_name = location_name.replace(' ', '_').replace(',', '').replace('&', 'and')
    safe_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe_name)
    # Remove multiple consecutive underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    return safe_name.strip('_')

def save_location_data(df, location):
    """Save dataframe to CSV file for a specific location"""
    safe_name = create_safe_filename(location["name"])

    # Save hourly data
    hourly_filename = f"{safe_name}_hourly_aqi_weather.csv"
    df.to_csv(hourly_filename, index=False)
    print(f"\n✓ Hourly data saved to {hourly_filename}")
    print(f"  Total records: {len(df)}")

    # Create and save daily aggregated data
    daily_df = df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['datetime']).dt.date

    # Columns to keep as-is (not aggregate)
    keep_columns = ['location_name', 'operator', 'capacity_mw', 'county', 'state',
                   'grid', 'latitude', 'longitude']

    # Aggregate by day (mean for most metrics)
    agg_dict = {}
    for col in daily_df.columns:
        if col not in ['datetime', 'date'] + keep_columns:
            if col in ['precipitation', 'rain']:
                agg_dict[col] = 'sum'  # Sum precipitation for daily total
            else:
                agg_dict[col] = 'mean'  # Mean for everything else

    daily_aggregated = daily_df.groupby('date').agg(agg_dict).reset_index()

    # Add back the non-aggregated columns
    for col in keep_columns:
        if col in df.columns:
            daily_aggregated[col] = df[col].iloc[0]

    daily_filename = f"{safe_name}_daily_aqi_weather.csv"
    daily_aggregated.to_csv(daily_filename, index=False)
    print(f"✓ Daily aggregated data saved to {daily_filename}")
    print(f"  Total days: {len(daily_aggregated)}")

    # Display summary statistics
    print("\nData Summary:")
    print("-" * 40)

    if 'calculated_aqi' in df.columns and not df['calculated_aqi'].isna().all():
        print(f"AQI Statistics:")
        print(f"  Average: {df['calculated_aqi'].mean():.1f}")
        print(f"  Min: {df['calculated_aqi'].min():.1f}")
        print(f"  Max: {df['calculated_aqi'].max():.1f}")

    if 'pm2_5' in df.columns and not df['pm2_5'].isna().all():
        print(f"PM2.5 Statistics:")
        print(f"  Average: {df['pm2_5'].mean():.1f} μg/m³")
        print(f"  Min: {df['pm2_5'].min():.1f} μg/m³")
        print(f"  Max: {df['pm2_5'].max():.1f} μg/m³")

    if 'temperature_2m' in df.columns and not df['temperature_2m'].isna().all():
        print(f"Temperature Statistics:")
        print(f"  Average: {df['temperature_2m'].mean():.1f}°C")
        print(f"  Min: {df['temperature_2m'].min():.1f}°C")
        print(f"  Max: {df['temperature_2m'].max():.1f}°C")

def main():
    print("=" * 80)
    print("POWER GRID LOCATIONS - HISTORICAL AQI & WEATHER DATA COLLECTOR")
    print("=" * 80)

    # Check command line arguments
    days_back = 365  # Default to 1 year

    if len(sys.argv) > 1:
        try:
            days_back = int(sys.argv[1])
            print(f"\nFetching data for the last {days_back} days")
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}. Using default 365 days.")
    else:
        print("\nFetching data for the last 365 days (1 year)")

    print(f"\nProcessing {len(POWER_GRID_LOCATIONS)} power grid locations...")
    print("=" * 80)

    # Process each location
    successful_locations = []
    failed_locations = []

    for idx, location in enumerate(POWER_GRID_LOCATIONS, 1):
        print(f"\n[{idx}/{len(POWER_GRID_LOCATIONS)}] Processing: {location['name']}")
        print("=" * 80)

        try:
            # Fetch data for this location
            df = fetch_location_historical_data(location, days_back)

            if df is not None and not df.empty:
                # Save to CSV files
                save_location_data(df, location)
                successful_locations.append(location['name'])
            else:
                print(f"Failed to fetch data for {location['name']}")
                failed_locations.append(location['name'])

        except Exception as e:
            print(f"Error processing {location['name']}: {e}")
            failed_locations.append(location['name'])

    # Final summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully processed: {len(successful_locations)} locations")
    if successful_locations:
        for loc in successful_locations:
            print(f"  ✓ {loc}")

    if failed_locations:
        print(f"\nFailed to process: {len(failed_locations)} locations")
        for loc in failed_locations:
            print(f"  ✗ {loc}")

    print("\nFiles created:")
    print("  - [location_name]_hourly_aqi_weather.csv - Hourly data")
    print("  - [location_name]_daily_aqi_weather.csv - Daily aggregated data")

if __name__ == "__main__":
    main()