import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv('WAQI_API_TOKEN')


def get_air_quality(city, token):
    """
    Get air quality data for a city
    """
    base_url = "https://api.waqi.info"
    
    try:
        # Make the API request
        url = f"{base_url}/feed/{city}/?token={token}"
        print(f"Making request to: {url}")
        
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Raw Response: {response.text}")
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"HTTP Error: {response.status_code}")
            return None
            
        # Try to parse JSON
        try:
            data = response.json()
            print(f"Parsed JSON: {json.dumps(data, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Response text: {response.text}")
            return None
            
        # Check API response status
        if data.get('status') != 'ok':
            print(f"API Error: {data.get('data', 'Unknown error')}")
            return None
            
        # Extract and display data
        city_data = data['data']
        city_name = city_data.get('city', {}).get('name', 'Unknown')
        aqi = city_data.get('aqi', 'N/A')
        
        print(f"\n=== AIR QUALITY DATA ===")
        print(f"City: {city_name}")
        print(f"AQI: {aqi}")
        
        # Additional data if available
        if 'iaqi' in city_data:
            pollutants = city_data['iaqi']
            print(f"\nPollutant Details:")
            for pollutant, info in pollutants.items():
                if isinstance(info, dict) and 'v' in info:
                    print(f"  {pollutant.upper()}: {info['v']}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

def main():
    # Configuration
    city = "bangalore"  # You can change this to any city
    # token = "demo"  # Replace with your actual token from https://aqicn.org/data-platform/token/
    
    print("AQICN Air Quality Data Fetcher")
    print("=" * 40)
    
    # Get air quality data
    result = get_air_quality(city, token)
    
    if result is None:
        print("\nTroubleshooting steps:")
        print("1. Check if your API token is valid")
        print("2. Verify the city name is correct")
        print("3. Ensure you have internet connectivity")
        print("4. Get a free token from: https://aqicn.org/data-platform/token/")

if __name__ == "__main__":
    main()