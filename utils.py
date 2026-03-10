import requests
import xarray as xr
import pandas as pd


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

def random_link_selection(num_of_frames, links_path):
    result = []
    with open(links_path) as f:
        links = pd.read_csv(f)
    random_frames = links.sample(num_of_frames)
    for link in random_frames.values:
        result.append('https://noaa-nws-hafs-pds.s3.amazonaws.com/' + link)
    
    return result