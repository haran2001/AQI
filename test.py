import requests
import json
import csv
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time
import sys

load_dotenv()

token = os.getenv('WAQI_API_TOKEN')

class AQIHistoricalCollector:
    def __init__(self, token):
        self.token = token
        self.base_url = "https://api.waqi.info"
        self.data_collected = []
        
    def get_current_air_quality(self, city):
        """Get current air quality data for a city"""
        try:
            url = f"{self.base_url}/feed/{city}/?token={self.token}"
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"HTTP Error: {response.status_code}")
                return None
                
            data = response.json()
            
            if data.get('status') != 'ok':
                print(f"API Error: {data.get('data', 'Unknown error')}")
                return None
                
            return data['data']
            
        except Exception as e:
            print(f"Error getting current data: {e}")
            return None
    
    def get_station_id(self, city):
        """Get the station ID for a city"""
        current_data = self.get_current_air_quality(city)
        if current_data and 'idx' in current_data:
            return current_data['idx']
        return None
    
    def try_historical_api(self, station_id, start_date, end_date):
        """Try to get historical data via API (may not be available in free tier)"""
        try:
            # This is a hypothetical endpoint - may not work with free API
            url = f"{self.base_url}/feed/@{station_id}/history"
            params = {
                'token': self.token,
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok':
                    return data.get('data', [])
            
            return None
            
        except Exception as e:
            print(f"Historical API not available or error: {e}")
            return None
    
    def collect_data_over_time(self, city, days_back=365, interval_days=7):
        """
        Collect data by making multiple API calls over time
        Note: This collects current data points, not true historical data
        """
        print(f"Collecting AQI data for {city} over {days_back} days...")
        print("Note: This collects current snapshots, not historical data")
        print("For true historical data, you may need a paid API plan")
        print("-" * 60)
        
        station_id = self.get_station_id(city)
        if not station_id:
            print(f"Could not get station ID for {city}")
            return None
            
        print(f"Station ID: {station_id}")
        
        # Try historical API first
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        historical_data = self.try_historical_api(station_id, start_date, end_date)
        
        if historical_data:
            print("Successfully retrieved historical data via API!")
            return historical_data
        
        # Fallback: Collect current data and simulate historical collection
        print("Historical API not available. Collecting current data snapshot...")
        
        current_data = self.get_current_air_quality(city)
        if current_data:
            # Create a single data point with current timestamp
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'city': current_data.get('city', {}).get('name', city),
                'aqi': current_data.get('aqi', None),
                'station_id': station_id
            }
            
            # Add pollutant data
            if 'iaqi' in current_data:
                for pollutant, info in current_data['iaqi'].items():
                    if isinstance(info, dict) and 'v' in info:
                        data_point[f'{pollutant}_value'] = info['v']
            
            # Add weather data if available
            if 'weather' in current_data:
                weather = current_data['weather']
                data_point['temperature'] = weather.get('tp')
                data_point['humidity'] = weather.get('hu')
                data_point['pressure'] = weather.get('pr')
                data_point['wind_speed'] = weather.get('ws')
            
            self.data_collected.append(data_point)
            return [data_point]
        
        return None
    
    def save_to_csv(self, filename="aqi_data.csv"):
        """Save collected data to CSV file"""
        if not self.data_collected:
            print("No data to save")
            return
            
        df = pd.DataFrame(self.data_collected)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        print(f"Total records: {len(self.data_collected)}")
        
        # Display summary
        if 'aqi' in df.columns:
            print(f"\nAQI Summary:")
            print(f"Average AQI: {df['aqi'].mean():.1f}")
            print(f"Min AQI: {df['aqi'].min()}")
            print(f"Max AQI: {df['aqi'].max()}")
    
    def save_to_json(self, filename="aqi_data.json"):
        """Save collected data to JSON file"""
        if not self.data_collected:
            print("No data to save")
            return
            
        with open(filename, 'w') as f:
            json.dump(self.data_collected, f, indent=2)
        print(f"Data saved to {filename}")

def main():
    if not token:
        print("Error: WAQI_API_TOKEN not found in environment variables")
        print("Please add your API token to a .env file:")
        print("WAQI_API_TOKEN=your_token_here")
        print("\nGet a free token from: https://aqicn.org/data-platform/token/")
        return
    
    # Configuration
    city = "bangalore"  # Change this to your desired city
    days_back = 365     # Number of days to go back
    
    print("AQICN Historical Air Quality Data Collector")
    print("=" * 50)
    print(f"Target city: {city}")
    print(f"Days back: {days_back}")
    print(f"API Token: {'*' * (len(token) - 4) + token[-4:]}")
    print()
    
    # Initialize collector
    collector = AQIHistoricalCollector(token)
    
    # Collect data
    data = collector.collect_data_over_time(city, days_back)
    
    if data:
        print("\n" + "=" * 50)
        print("Data Collection Summary:")
        
        # Save to files
        collector.save_to_csv(f"{city}_aqi_data.csv")
        collector.save_to_json(f"{city}_aqi_data.json")
        
        print("\nData collection completed!")
        print("\nNOTE: For comprehensive historical data, consider:")
        print("1. AQICN Pro/Paid API plans")
        print("2. Alternative APIs like OpenWeatherMap Air Pollution API")
        print("3. Government air quality databases")
        
    else:
        print("Failed to collect data. Please check:")
        print("1. Your API token is valid")
        print("2. The city name is correct")
        print("3. Internet connectivity")
        print("4. API rate limits")

def test_api_connection():
    """Test API connection with current data"""
    print("Testing API connection...")
    
    if not token:
        print("No API token found!")
        return False
    
    collector = AQIHistoricalCollector(token)
    
    # Test cities
    test_cities = ["bangalore", "bengaluru", "delhi", "mumbai", "shanghai"]
    
    for city in test_cities:
        print(f"\nTesting {city}...")
        current_data = collector.get_current_air_quality(city)
        if current_data:
            print(f"✓ {city}: AQI {current_data.get('aqi', 'N/A')}")
            return True
        else:
            print(f"✗ {city}: Failed")
    
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_api_connection()
    else:
        main()