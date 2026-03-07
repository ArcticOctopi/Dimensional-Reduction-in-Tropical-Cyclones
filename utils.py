import requests
import xarray as xr


def download_and_open(url,typeOfKey = 'isobaricInhPa', filename="temp.grib2"):

  
  print("Downloading filtered data...")
  response = requests.get(url, stream=True)
  response.raise_for_status()

  with open(filename, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
      f.write(chunk)

      # 3. Open with xarray using the cfgrib engine
  ds = xr.open_dataset(filename, 
                       filter_by_keys={'typeOfLevel': typeOfKey },
                       engine='cfgrib')
  return ds  