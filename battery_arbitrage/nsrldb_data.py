import requests
import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("NSRLDB_API_KEY")


url = f"https://developer.nrel.gov/api/solar/psm3-5min-download.json?api_key={API_KEP}"

payload = "names=2012&leap_day=false&interval=60&utc=false&full_name=Honored%2BUser&email=honored.user%40gmail.com&affiliation=NREL&mailing_list=true&reason=Academic&attributes=dhi%2Cdni%2Cwind_speed_10m_nwp%2Csurface_air_temperature_nwp&wkt=MULTIPOINT(-106.22%2032.9741%2C-106.18%2032.9741%2C-106.1%2032.9741)"

headers = {
    'content-type': "application/x-www-form-urlencoded",
    'cache-control': "no-cache"
}

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)