import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import sys

def fetch_bangalore_historical_data(days_back=365):
    """
    Fetch historical AQI and weather data for Bangalore from Open-Meteo API
    """

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Bangalore coordinates
    bangalore_lat = 12.9716
    bangalore_lon = 77.5946

    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    print(f"Fetching data for Bangalore from {start_date} to {end_date}")
    print("=" * 60)

    # Fetch Air Quality Data
    print("Fetching air quality data...")
    air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    air_quality_params = {
        "latitude": bangalore_lat,
        "longitude": bangalore_lon,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                   "sulphur_dioxide", "ozone", "dust", "uv_index", "uv_index_clear_sky",
                   "ammonia", "alder_pollen", "birch_pollen", "grass_pollen",
                   "mugwort_pollen", "olive_pollen", "ragweed_pollen"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "Asia/Kolkata"
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
    print("\nFetching weather data...")
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": bangalore_lat,
        "longitude": bangalore_lon,
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
        "timezone": "Asia/Kolkata"
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
    print("\nMerging datasets...")
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

    # Add location information
    combined_df['city'] = 'Bangalore'
    combined_df['latitude'] = bangalore_lat
    combined_df['longitude'] = bangalore_lon

    # Sort by datetime
    combined_df = combined_df.sort_values('datetime')

    # Calculate AQI based on PM2.5 and PM10 (simplified version)
    # This is a basic AQI calculation - real AQI is more complex
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
    column_order = ['datetime', 'city', 'latitude', 'longitude', 'calculated_aqi',
                   'pm10', 'pm2_5', 'temperature_2m', 'relative_humidity_2m',
                   'wind_speed_10m', 'precipitation', 'pressure_msl']

    # Add remaining columns
    remaining_cols = [col for col in combined_df.columns if col not in column_order]
    column_order.extend(remaining_cols)

    # Reorder columns (only include columns that exist)
    existing_columns = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[existing_columns]

    return combined_df

def save_to_csv(df, filename="bangalore_historical_aqi_weather.csv"):
    """Save dataframe to CSV file"""
    df.to_csv(filename, index=False)
    print(f"\n✓ Data saved to {filename}")
    print(f"Total records: {len(df)}")

    # Display summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    if 'calculated_aqi' in df.columns and not df['calculated_aqi'].isna().all():
        print(f"\nAQI Statistics:")
        print(f"  Average AQI: {df['calculated_aqi'].mean():.1f}")
        print(f"  Min AQI: {df['calculated_aqi'].min():.1f}")
        print(f"  Max AQI: {df['calculated_aqi'].max():.1f}")

    if 'pm2_5' in df.columns and not df['pm2_5'].isna().all():
        print(f"\nPM2.5 Statistics:")
        print(f"  Average: {df['pm2_5'].mean():.1f} μg/m³")
        print(f"  Min: {df['pm2_5'].min():.1f} μg/m³")
        print(f"  Max: {df['pm2_5'].max():.1f} μg/m³")

    if 'temperature_2m' in df.columns and not df['temperature_2m'].isna().all():
        print(f"\nTemperature Statistics:")
        print(f"  Average: {df['temperature_2m'].mean():.1f}°C")
        print(f"  Min: {df['temperature_2m'].min():.1f}°C")
        print(f"  Max: {df['temperature_2m'].max():.1f}°C")

    if 'relative_humidity_2m' in df.columns and not df['relative_humidity_2m'].isna().all():
        print(f"\nHumidity Statistics:")
        print(f"  Average: {df['relative_humidity_2m'].mean():.1f}%")
        print(f"  Min: {df['relative_humidity_2m'].min():.1f}%")
        print(f"  Max: {df['relative_humidity_2m'].max():.1f}%")

    print(f"\nDate Range: {df['datetime'].min()} to {df['datetime'].max()}")

def main():
    print("=" * 60)
    print("BANGALORE HISTORICAL AQI & WEATHER DATA COLLECTOR")
    print("=" * 60)

    # Check command line arguments
    days_back = 365  # Default to 1 year

    if len(sys.argv) > 1:
        try:
            days_back = int(sys.argv[1])
            print(f"Fetching data for the last {days_back} days")
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}. Using default 365 days.")
    else:
        print("Fetching data for the last 365 days (1 year)")

    # Fetch data
    df = fetch_bangalore_historical_data(days_back)

    if df is not None and not df.empty:
        # Save to CSV
        save_to_csv(df)

        # Also save a daily aggregated version
        print("\nCreating daily aggregated data...")
        daily_df = df.copy()
        daily_df['date'] = pd.to_datetime(daily_df['datetime']).dt.date

        # Aggregate by day (mean for most metrics)
        agg_dict = {col: 'mean' for col in daily_df.columns if col not in ['datetime', 'date', 'city', 'latitude', 'longitude']}
        agg_dict['precipitation'] = 'sum'  # Sum precipitation for daily total
        agg_dict['rain'] = 'sum' if 'rain' in daily_df.columns else 'mean'

        daily_aggregated = daily_df.groupby('date').agg(agg_dict).reset_index()
        daily_aggregated['city'] = 'Bangalore'
        daily_aggregated['latitude'] = 12.9716
        daily_aggregated['longitude'] = 77.5946

        daily_aggregated.to_csv("bangalore_daily_aqi_weather.csv", index=False)
        print(f"✓ Daily aggregated data saved to bangalore_daily_aqi_weather.csv")
        print(f"  Total days: {len(daily_aggregated)}")
    else:
        print("Failed to fetch data. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()