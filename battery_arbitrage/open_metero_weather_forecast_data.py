#forecast data

from tracemalloc import start
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
start_date = '2025-08-01'
end_date = '2025-08-30'
latitude=35.3733
longitude=-119.0187

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below


# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_snow_depth = hourly.Variables(2).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
hourly_showers = hourly.Variables(4).ValuesAsNumpy()
hourly_rain = hourly.Variables(5).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(6).ValuesAsNumpy()
hourly_precipitation_probability = hourly.Variables(7).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(8).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(9).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(10).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(11).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(12).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(13).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(14).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(15).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(16).ValuesAsNumpy()
hourly_visibility = hourly.Variables(17).ValuesAsNumpy()
hourly_evapotranspiration = hourly.Variables(18).ValuesAsNumpy()
hourly_et0_fao_evapotranspiration = hourly.Variables(19).ValuesAsNumpy()
hourly_vapour_pressure_deficit = hourly.Variables(20).ValuesAsNumpy()
hourly_temperature_180m = hourly.Variables(21).ValuesAsNumpy()
hourly_temperature_80m = hourly.Variables(22).ValuesAsNumpy()
hourly_temperature_120m = hourly.Variables(23).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(24).ValuesAsNumpy()
hourly_wind_direction_180m = hourly.Variables(25).ValuesAsNumpy()
hourly_wind_direction_120m = hourly.Variables(26).ValuesAsNumpy()
hourly_wind_direction_80m = hourly.Variables(27).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(28).ValuesAsNumpy()
hourly_wind_speed_180m = hourly.Variables(29).ValuesAsNumpy()
hourly_wind_speed_120m = hourly.Variables(30).ValuesAsNumpy()
hourly_wind_speed_80m = hourly.Variables(31).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(32).ValuesAsNumpy()
hourly_soil_temperature_0cm = hourly.Variables(33).ValuesAsNumpy()
hourly_soil_temperature_6cm = hourly.Variables(34).ValuesAsNumpy()
hourly_soil_temperature_18cm = hourly.Variables(35).ValuesAsNumpy()
hourly_soil_temperature_54cm = hourly.Variables(36).ValuesAsNumpy()
hourly_soil_moisture_0_to_1cm = hourly.Variables(37).ValuesAsNumpy()
hourly_soil_moisture_1_to_3cm = hourly.Variables(38).ValuesAsNumpy()
hourly_soil_moisture_3_to_9cm = hourly.Variables(39).ValuesAsNumpy()
hourly_soil_moisture_9_to_27cm = hourly.Variables(40).ValuesAsNumpy()
hourly_soil_moisture_27_to_81cm = hourly.Variables(41).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["snow_depth"] = hourly_snow_depth
hourly_data["snowfall"] = hourly_snowfall
hourly_data["showers"] = hourly_showers
hourly_data["rain"] = hourly_rain
hourly_data["precipitation"] = hourly_precipitation
hourly_data["precipitation_probability"] = hourly_precipitation_probability
hourly_data["apparent_temperature"] = hourly_apparent_temperature
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["weather_code"] = hourly_weather_code
hourly_data["pressure_msl"] = hourly_pressure_msl
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["visibility"] = hourly_visibility
hourly_data["evapotranspiration"] = hourly_evapotranspiration
hourly_data["et0_fao_evapotranspiration"] = hourly_et0_fao_evapotranspiration
hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit
hourly_data["temperature_180m"] = hourly_temperature_180m
hourly_data["temperature_80m"] = hourly_temperature_80m
hourly_data["temperature_120m"] = hourly_temperature_120m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
hourly_data["wind_direction_180m"] = hourly_wind_direction_180m
hourly_data["wind_direction_120m"] = hourly_wind_direction_120m
hourly_data["wind_direction_80m"] = hourly_wind_direction_80m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_speed_180m"] = hourly_wind_speed_180m
hourly_data["wind_speed_120m"] = hourly_wind_speed_120m
hourly_data["wind_speed_80m"] = hourly_wind_speed_80m
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["soil_temperature_0cm"] = hourly_soil_temperature_0cm
hourly_data["soil_temperature_6cm"] = hourly_soil_temperature_6cm
hourly_data["soil_temperature_18cm"] = hourly_soil_temperature_18cm
hourly_data["soil_temperature_54cm"] = hourly_soil_temperature_54cm
hourly_data["soil_moisture_0_to_1cm"] = hourly_soil_moisture_0_to_1cm
hourly_data["soil_moisture_1_to_3cm"] = hourly_soil_moisture_1_to_3cm
hourly_data["soil_moisture_3_to_9cm"] = hourly_soil_moisture_3_to_9cm
hourly_data["soil_moisture_9_to_27cm"] = hourly_soil_moisture_9_to_27cm
hourly_data["soil_moisture_27_to_81cm"] = hourly_soil_moisture_27_to_81cm

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)

df_name = start_date + '_' + end_date + '_open_metero_weather_data.csv'
hourly_dataframe.to_csv('data/' + df_name, index=False)


import redis
import time
import requests

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)
stream_name = 'api_data_stream'
api_url = 'https://historical-forecast-api.open-meteo.com/v1/forecast'

def fetch_and_stream_data():
	try:
		url = ""
		params = {
			"latitude": latitude,
			"longitude": longitude,
			"start_date": start_date,
			"end_date": end_date,
			"hourly": ["temperature_2m", "relative_humidity_2m", "snow_depth", "snowfall", "showers", "rain", "precipitation", "precipitation_probability", "apparent_temperature", "dew_point_2m", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "temperature_180m", "temperature_80m", "temperature_120m", "wind_gusts_10m", "wind_direction_180m", "wind_direction_120m", "wind_direction_80m", "wind_direction_10m", "wind_speed_180m", "wind_speed_120m", "wind_speed_80m", "wind_speed_10m", "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"],
		}
		response = requests.get(api_url)
		response.raise_for_status() # Raise an exception for HTTP errors
		response = openmeteo.weather_api(url, params=params)
		data = response.json()

		# Add data to Redis Stream
		r.xadd(stream_name, data)
		print(f"Data added to stream: {data}")
	except requests.exceptions.RequestException as e:
		print(f"Error fetching data from API: {e}")

# Example of continuous polling
while True:
	fetch_and_stream_data()
	time.sleep(5) # Poll every 5 seconds


import redis
import time

r = redis.Redis(host='localhost', port=6379, db=0)
stream_name = 'api_data_stream'
group_name = 'my_consumer_group'
consumer_name = 'consumer_1'

# Create consumer group if it doesn't exist
try:
	r.xgroup_create(stream_name, group_name, id='0', mkstream=True)
except redis.exceptions.ResponseError as e:
	if "BUSYGROUP" not in str(e):
		raise

while True:
	try:
		# Read new entries from the stream using the consumer group
		messages = r.xreadgroup(group_name, consumer_name, {stream_name: '>'}, count=1, block=0)
		if messages:
			for stream, message_list in messages:
				for message_id, message_data in message_list:
					decoded_data = {k.decode(): v.decode() for k, v in message_data.items()}
					print(f"Consumer {consumer_name} received: ID={message_id.decode()}, Data={decoded_data}")
					# Process the data (e.g., store in another database, perform analytics)
					# Acknowledge the message
					r.xack(stream_name, group_name, message_id)
		time.sleep(1) # Small delay to prevent busy-waiting
	except Exception as e:
		print(f"Error consuming data: {e}")
		time.sleep(5)