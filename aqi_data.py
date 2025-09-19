import requests
from dotenv import load_dotenv,  load_dotenv
import os

load_dotenv()

base_url = "https://api.waqi.info"
token = os.getenv('WAQI_API_TOKEN')  # Get from https://aqicn.org/data-platform/token/

# City-based query
city = 'Bangalore'
r = requests.get(f"{base_url}/feed/{city}/?token={token}")
data = r.json()
print(f"City: {data['data']['city']['name']}, AQI: {data['data']['aqi']}")

# Coordinate-based query  
lat, lng = 12.9116, 77.6104
r = requests.get(f"{base_url}/feed/geo:{lat};{lng}/?token={token}")
data = r.json()

# Map bounds query (multiple stations)
latlngbox = "48.639956,1.761273,49.159944,2.947797"
r = requests.get(f"{base_url}/map/bounds/?latlng={latlngbox}&token={token}")