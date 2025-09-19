#!/usr/bin/env python3
"""
Air Quality Data Collection Script for Bangalore
Collects both independent variables (weather) and dependent variable (AQI) for ML model training
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple
import os
from dataclasses import dataclass
import traceback

from dotenv import load_dotenv
load_dotenv()

open_weather_api_key = os.getenv('OPEN_WEATHER_API_KEY')
waqi_token = os.getenv('WAQI_API_TOKEN')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('air_quality_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BangaloreConfig:
    """Configuration for Bangalore location and API keys"""
    # Bangalore BTM coordinates (you can adjust these for specific location)
    latitude: float = 12.9116  # BTM Layout coordinates
    longitude: float = 77.6104
    
    # API Keys - Replace with your actual keys
    openweather_api_key: str = open_weather_api_key
    waqi_token: str = "YOUR_WAQI_TOKEN"  # Get from https://aqicn.org/data-platform/token/
    
    # Data collection parameters
    days_back: int = 365  # One year
    delay_between_requests: float = 1.0  # Respect API rate limits

class AirQualityDataCollector:
    """Main class for collecting air quality and weather data"""
    
    def __init__(self, config: BangaloreConfig):
        self.config = config
        self.db_name = 'bangalore_air_quality_data.db'
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for storing collected data"""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS air_quality_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME UNIQUE,
                    
                    -- Independent Variables (Weather)
                    temperature REAL,
                    feels_like REAL,
                    humidity INTEGER,
                    pressure REAL,
                    sea_level_pressure REAL,
                    ground_level_pressure REAL,
                    visibility INTEGER,
                    wind_speed REAL,
                    wind_direction INTEGER,
                    wind_gust REAL,
                    cloudiness INTEGER,
                    uv_index REAL,
                    dew_point REAL,
                    precipitation_1h REAL,
                    precipitation_3h REAL,
                    
                    -- Temporal Features
                    hour INTEGER,
                    day_of_week INTEGER,
                    day_of_month INTEGER,
                    month INTEGER,
                    year INTEGER,
                    is_weekend INTEGER,
                    
                    -- Dependent Variable (Air Quality)
                    aqi INTEGER,
                    pm2_5 REAL,
                    pm10 REAL,
                    co REAL,
                    no REAL,
                    no2 REAL,
                    o3 REAL,
                    so2 REAL,
                    nh3 REAL,
                    
                    -- Data source tracking
                    weather_source TEXT,
                    aqi_source TEXT,
                    data_quality_score REAL
                )
            ''')
            conn.commit()
            logger.info("Database initialized successfully")

    def get_openweather_historical_weather(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Collect historical weather data from OpenWeatherMap
        Note: One Call API 3.0 has historical data, but free tier is limited
        """
        weather_data = []
        base_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
        
        current_date = start_date
        while current_date <= end_date:
            try:
                # Convert to Unix timestamp
                timestamp = int(current_date.timestamp())
                
                params = {
                    'lat': self.config.latitude,
                    'lon': self.config.longitude,
                    'dt': timestamp,
                    'appid': self.config.openweather_api_key,
                    'units': 'metric'
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        hourly_data = data['data'][0]  # Get the data for that day
                        
                        weather_record = {
                            'timestamp': current_date,
                            'temperature': hourly_data.get('temp'),
                            'feels_like': hourly_data.get('feels_like'),
                            'humidity': hourly_data.get('humidity'),
                            'pressure': hourly_data.get('pressure'),
                            'dew_point': hourly_data.get('dew_point'),
                            'uv_index': hourly_data.get('uvi'),
                            'cloudiness': hourly_data.get('clouds'),
                            'visibility': hourly_data.get('visibility'),
                            'wind_speed': hourly_data.get('wind_speed'),
                            'wind_direction': hourly_data.get('wind_deg'),
                            'wind_gust': hourly_data.get('wind_gust'),
                            'weather_source': 'openweather_historical'
                        }
                        
                        # Handle precipitation data
                        if 'rain' in hourly_data:
                            weather_record['precipitation_1h'] = hourly_data['rain'].get('1h', 0)
                        
                        weather_data.append(weather_record)
                        logger.info(f"Collected weather data for {current_date.date()}")
                
                elif response.status_code == 401:
                    logger.error("OpenWeatherMap API key is invalid")
                    break
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded, waiting...")
                    time.sleep(60)
                    continue
                else:
                    logger.warning(f"Failed to get weather data for {current_date.date()}: {response.status_code}")
                
                current_date += timedelta(days=1)
                time.sleep(self.config.delay_between_requests)
                
            except Exception as e:
                logger.error(f"Error collecting weather data for {current_date.date()}: {str(e)}")
                current_date += timedelta(days=1)
                continue
                
        return weather_data

    def get_openweather_current_weather(self) -> Dict:
        """Get current weather data from OpenWeatherMap (more reliable for recent data)"""
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        
        params = {
            'lat': self.config.latitude,
            'lon': self.config.longitude,
            'appid': self.config.openweather_api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                weather_record = {
                    'timestamp': datetime.now(),
                    'temperature': data['main'].get('temp'),
                    'feels_like': data['main'].get('feels_like'),
                    'humidity': data['main'].get('humidity'),
                    'pressure': data['main'].get('pressure'),
                    'sea_level_pressure': data['main'].get('sea_level'),
                    'ground_level_pressure': data['main'].get('grnd_level'),
                    'visibility': data.get('visibility'),
                    'wind_speed': data.get('wind', {}).get('speed'),
                    'wind_direction': data.get('wind', {}).get('deg'),
                    'wind_gust': data.get('wind', {}).get('gust'),
                    'cloudiness': data.get('clouds', {}).get('all'),
                    'weather_source': 'openweather_current'
                }
                
                # Handle precipitation
                if 'rain' in data:
                    weather_record['precipitation_1h'] = data['rain'].get('1h', 0)
                    weather_record['precipitation_3h'] = data['rain'].get('3h', 0)
                
                return weather_record
                
        except Exception as e:
            logger.error(f"Error getting current weather: {str(e)}")
            return {}

    def get_openweather_air_pollution_historical(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Collect historical air pollution data from OpenWeatherMap
        Available from November 27, 2020
        """
        air_quality_data = []
        base_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
        
        # Process in chunks of 30 days to avoid large responses
        chunk_size = timedelta(days=30)
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)
            
            try:
                params = {
                    'lat': self.config.latitude,
                    'lon': self.config.longitude,
                    'start': int(current_start.timestamp()),
                    'end': int(current_end.timestamp()),
                    'appid': self.config.openweather_api_key
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for record in data.get('list', []):
                        air_quality_record = {
                            'timestamp': datetime.fromtimestamp(record['dt']),
                            'aqi': record['main'].get('aqi'),
                            'co': record['components'].get('co'),
                            'no': record['components'].get('no'),
                            'no2': record['components'].get('no2'),
                            'o3': record['components'].get('o3'),
                            'so2': record['components'].get('so2'),
                            'pm2_5': record['components'].get('pm2_5'),
                            'pm10': record['components'].get('pm10'),
                            'nh3': record['components'].get('nh3'),
                            'aqi_source': 'openweather_historical'
                        }
                        air_quality_data.append(air_quality_record)
                    
                    logger.info(f"Collected air quality data from {current_start.date()} to {current_end.date()}")
                
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded, waiting...")
                    time.sleep(60)
                    continue
                else:
                    logger.warning(f"Failed to get air quality data: {response.status_code}")
                
                current_start = current_end
                time.sleep(self.config.delay_between_requests)
                
            except Exception as e:
                logger.error(f"Error collecting air quality data: {str(e)}")
                current_start = current_end
                continue
                
        return air_quality_data

    def get_waqi_current_data(self) -> Dict:
        """Get current air quality data from WAQI"""
        try:
            # Try specific Bangalore BTM station first
            urls_to_try = [
                f"http://api.waqi.info/feed/bangalore/btm/?token={self.config.waqi_token}",
                f"http://api.waqi.info/feed/bangalore/?token={self.config.waqi_token}",
                f"http://api.waqi.info/feed/geo:{self.config.latitude};{self.config.longitude}/?token={self.config.waqi_token}"
            ]
            
            for url in urls_to_try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('status') == 'ok':
                        aqi_data = data.get('data', {})
                        
                        record = {
                            'timestamp': datetime.now(),
                            'aqi': aqi_data.get('aqi'),
                            'aqi_source': 'waqi_current'
                        }
                        
                        # Extract individual pollutants
                        iaqi = aqi_data.get('iaqi', {})
                        if 'pm25' in iaqi:
                            record['pm2_5'] = iaqi['pm25'].get('v')
                        if 'pm10' in iaqi:
                            record['pm10'] = iaqi['pm10'].get('v')
                        if 'co' in iaqi:
                            record['co'] = iaqi['co'].get('v')
                        if 'no2' in iaqi:
                            record['no2'] = iaqi['no2'].get('v')
                        if 'o3' in iaqi:
                            record['o3'] = iaqi['o3'].get('v')
                        if 'so2' in iaqi:
                            record['so2'] = iaqi['so2'].get('v')
                        
                        # Extract weather data if available
                        iaqi = aqi_data.get('iaqi', {})
                        if 't' in iaqi:  # temperature
                            record['temperature'] = iaqi['t'].get('v')
                        if 'h' in iaqi:  # humidity
                            record['humidity'] = iaqi['h'].get('v')
                        if 'p' in iaqi:  # pressure
                            record['pressure'] = iaqi['p'].get('v')
                        if 'w' in iaqi:  # wind
                            record['wind_speed'] = iaqi['w'].get('v')
                        
                        return record
                        
        except Exception as e:
            logger.error(f"Error getting WAQI data: {str(e)}")
            
        return {}

    def add_temporal_features(self, timestamp: datetime) -> Dict:
        """Add temporal features for ML model"""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),  # 0=Monday, 6=Sunday
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'year': timestamp.year,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0
        }

    def calculate_data_quality_score(self, record: Dict) -> float:
        """Calculate a quality score based on completeness of data"""
        total_fields = 15  # Key fields we care about
        filled_fields = 0
        
        key_fields = [
            'temperature', 'humidity', 'pressure', 'wind_speed',
            'aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2',
            'visibility', 'cloudiness', 'wind_direction', 'dew_point'
        ]
        
        for field in key_fields:
            if record.get(field) is not None:
                filled_fields += 1
                
        return filled_fields / total_fields

    def merge_and_store_data(self, weather_data: List[Dict], air_quality_data: List[Dict]):
        """Merge weather and air quality data, then store in database"""
        
        # Create dataframes for easier merging
        weather_df = pd.DataFrame(weather_data) if weather_data else pd.DataFrame()
        air_quality_df = pd.DataFrame(air_quality_data) if air_quality_data else pd.DataFrame()
        
        merged_records = []
        
        if not weather_df.empty and not air_quality_df.empty:
            # Round timestamps to nearest hour for better matching
            weather_df['timestamp_hour'] = pd.to_datetime(weather_df['timestamp']).dt.round('H')
            air_quality_df['timestamp_hour'] = pd.to_datetime(air_quality_df['timestamp']).dt.round('H')
            
            # Merge on rounded timestamps
            merged_df = pd.merge(weather_df, air_quality_df, on='timestamp_hour', how='outer', suffixes=('_weather', '_aqi'))
            
            for _, row in merged_df.iterrows():
                # Use the original timestamp, preferring weather data timestamp
                timestamp = row.get('timestamp_weather') or row.get('timestamp_aqi')
                if pd.isna(timestamp):
                    continue
                    
                timestamp = pd.to_datetime(timestamp)
                
                record = {
                    'timestamp': timestamp,
                    **{k: v for k, v in row.items() if not k.startswith('timestamp') and pd.notna(v)}
                }
                
                # Add temporal features
                record.update(self.add_temporal_features(timestamp))
                
                # Calculate data quality score
                record['data_quality_score'] = self.calculate_data_quality_score(record)
                
                merged_records.append(record)
        
        # Store in database
        self.store_records(merged_records)
        
    def store_records(self, records: List[Dict]):
        """Store records in SQLite database"""
        if not records:
            logger.warning("No records to store")
            return
            
        with sqlite3.connect(self.db_name) as conn:
            for record in records:
                try:
                    # Prepare the record for insertion
                    columns = []
                    values = []
                    placeholders = []
                    
                    for key, value in record.items():
                        if value is not None and not pd.isna(value):
                            columns.append(key)
                            values.append(value)
                            placeholders.append('?')
                    
                    if columns:
                        sql = f"""
                        INSERT OR REPLACE INTO air_quality_data 
                        ({', '.join(columns)}) 
                        VALUES ({', '.join(placeholders)})
                        """
                        
                        conn.execute(sql, values)
                        
                except Exception as e:
                    logger.error(f"Error storing record: {str(e)}")
                    continue
            
            conn.commit()
            logger.info(f"Stored {len(records)} records in database")

    def collect_historical_data(self):
        """Main method to collect one year of historical data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.days_back)
        
        # Ensure we don't go before OpenWeatherMap's historical data availability
        earliest_date = datetime(2020, 11, 27)
        if start_date < earliest_date:
            start_date = earliest_date
            logger.info(f"Adjusted start date to {start_date.date()} (earliest available)")
        
        logger.info(f"Collecting data from {start_date.date()} to {end_date.date()}")
        
        # Collect weather data
        logger.info("Collecting historical weather data...")
        weather_data = self.get_openweather_historical_weather(start_date, end_date)
        
        # Collect air quality data
        logger.info("Collecting historical air quality data...")
        air_quality_data = self.get_openweather_air_pollution_historical(start_date, end_date)
        
        # Merge and store data
        logger.info("Merging and storing data...")
        self.merge_and_store_data(weather_data, air_quality_data)
        
        # Get current data as well
        logger.info("Collecting current data...")
        current_weather = self.get_openweather_current_weather()
        current_aqi = self.get_waqi_current_data()
        
        if current_weather or current_aqi:
            current_record = {**current_weather, **current_aqi}
            current_record.update(self.add_temporal_features(datetime.now()))
            current_record['data_quality_score'] = self.calculate_data_quality_score(current_record)
            self.store_records([current_record])

    def export_to_csv(self, filename: str = "bangalore_air_quality_dataset.csv"):
        """Export collected data to CSV for ML model training"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                df = pd.read_sql_query("SELECT * FROM air_quality_data ORDER BY timestamp", conn)
                
                # Basic data cleaning
                df = df.drop_duplicates(subset=['timestamp'])
                df = df.sort_values('timestamp')
                
                # Save to CSV
                df.to_csv(filename, index=False)
                logger.info(f"Exported {len(df)} records to {filename}")
                
                # Print basic statistics
                print("\n" + "="*50)
                print("DATASET SUMMARY")
                print("="*50)
                print(f"Total records: {len(df)}")
                print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"Average data quality score: {df['data_quality_score'].mean():.2f}")
                
                # Show completion rates for key variables
                key_vars = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'pm2_5', 'pm10']
                print("\nData Completeness:")
                for var in key_vars:
                    if var in df.columns:
                        completion = (df[var].notna().sum() / len(df)) * 100
                        print(f"  {var}: {completion:.1f}%")
                
                return df
                
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return None

def main():
    """Main execution function"""
    
    # Configuration
    config = BangaloreConfig()
    
    # Validate API keys
    if config.openweather_api_key == "YOUR_OPENWEATHER_API_KEY":
        print("⚠️  Please set your OpenWeatherMap API key in the BangaloreConfig class")
        print("   Get it from: https://openweathermap.org/api")
        return
    
    if config.waqi_token == "YOUR_WAQI_TOKEN":
        print("⚠️  Please set your WAQI token in the BangaloreConfig class")
        print("   Get it from: https://aqicn.org/data-platform/token/")
        print("   (Note: WAQI token is optional, OpenWeatherMap data will still be collected)")
    
    # Initialize collector
    collector = AirQualityDataCollector(config)
    
    # Collect data
    try:
        print("🚀 Starting data collection for Bangalore air quality monitoring...")
        print(f"📍 Location: {config.latitude}, {config.longitude} (BTM Layout)")
        print(f"📅 Collecting {config.days_back} days of historical data")
        
        collector.collect_historical_data()
        
        # Export results
        df = collector.export_to_csv()
        
        if df is not None:
            print("\n✅ Data collection completed successfully!")
            print(f"💾 Data saved to: bangalore_air_quality_dataset.csv")
            print(f"🗄️  Database: {collector.db_name}")
            
        else:
            print("❌ Failed to export data")
            
    except KeyboardInterrupt:
        print("\n🛑 Data collection interrupted by user")
    except Exception as e:
        print(f"❌ Error during data collection: {str(e)}")
        logger.error(f"Fatal error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()