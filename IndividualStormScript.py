import xarray as xr
import math
from pyproj import CRS, Transformer
import pyproj
import numpy as np
import xesmf as xe
import matplotlib.pyplot as plt
# import fsspec 
# from dask.distributed import Client, progress
import re
from cfgrib import xarray_to_grib
import pickle
import gc
from datetime import date
import requests
import scipy
import utils
import logging
import pandas as pd
import add_best_track
import joblib
import TC_EmbeddingStormAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
'''
-Hurricane Melissa (2025, 13l)
    -Hurricane Milton (2024, 14l)
    -Hurricane Beryl (2024, 02l)
    -Hurricane Otis (2023, 18e)
    -Hurricane Ian (2022, 09l)
'''

def pull_storm_from_links(link_list: list, storm_year: str, storm_number: str) -> list:
    '''
        link_list: list of all model runs found on the AWS HAFS repository
        storm_year: year of desired storm, in string format 
        storm_number: the number of the named storm for that basin, eg 03w or 12l

        returns a list of all 6 hour-timesteps of that storm to be used to creating a pseudo-analysis of a desired storm.

    '''
    pattern = fr"(?=.*{storm_year})(?=.*{storm_number})(?=.*hfsa)"
    matches = ['https://noaa-nws-hafs-pds.s3.amazonaws.com/' + s for s in links[0].values if re.search(pattern, s)]    

    return matches

def frame_processing(s3_path: str, starting_timestep = '.f006', heading_timestep = '.f009' , ri_timestep = '.030'):
    '''
    Docstring for frame_processing
    
    Inputs:
        s3_path: path to aws bucket for a particular model run

        starting_timestep: in a '.f###' format where the string is the model timestep desired (defaults to the 6 hour timestep)
        heading_timestep: the model timestep to calculate storm heading (defaults to 9 hour timestep)
        ri_timestep: model used to calculate intensification (defaults to the 30 hour timestep)


    Checks for frame validity:
        Central minimum pressure of at least 1000mb
    
    If valid returns a Model Dataset consisting of:
        PBL temp, heights, rh, and u/v winds
        700 mb temp, heights, rh, and u/v winds
        central minimum pressure for the frame,
        maximum wind for the frame
    '''
    
    frame_path = s3_path
    next_frame_path = re.sub(starting_timestep, heading_timestep, frame_path)
    ri_frame_path = re.sub(starting_timestep, ri_timestep, frame_path)

    ## Arguments for downloading grib files
    atm_args = dict(typeOfKey = 'isobaricInhPa',filename="Model_Data/temp_gribs/atm_temp.grb2" )
    sfc_args = dict(typeOfKey = 'meanSea',filename="Model_Data/temp_gribs/sfc_temp.grb2" )
    next_step_sfc_args = dict(typeOfKey = 'meanSea',filename="Model_Data/temp_gribs/sfc_next_temp.grb2" )
    ri_args = dict(typeOfKey = 'meanSea',filename="Model_Data/temp_gribs/ri_temp.grb2" )

    # Current frame surface data
    try:
        ds_sfc = download_and_open(frame_path, **sfc_args)
    except Exception as e:
        print('Couldn\'t download surface dataset, skipping frame')
        print(e)
        logging.info('Couldn\'t download surface dataset, skipping frame')
        return None, None, None
    
    # Converting surface pressure from pascal's to mb
    ds_sfc['prmsl'] = ds_sfc['prmsl']/100

    # get current frame center coordinates and mslp
    
    center_coords_sfc = utils.get_sfc_center(ds_sfc, land_filter = False)
    
    if not center_coords_sfc:
        print('Could not determine surface centers, skipping frame.')
        
        return None, None, None
    
    # Opening the atmospheric data
    try:
        ds_atm = download_and_open(frame_path, **atm_args)
    except Exception as e:
        print('Couldn\'t download atmosphere dataset, skipping frame')
        logging.info('Couldn\'t download atmosphere dataset, skipping frame')
        return None, None, None

    # opening the surface data for the next timestep
    # necessary for calculating storm motion
    try:
        ds_sfc_next_step = download_and_open(next_frame_path, **next_step_sfc_args)
    except Exception as e:
        logging.info('Couldn\'t download next surface dataset, skipping frame')
        return None, None, None

    #Converting surface pressure from pascal's to mb
    ds_sfc_next_step['prmsl'] = ds_sfc_next_step['prmsl']/100
    
    # Selecting only the surface level, 700mb, and 200mb level
    ds_atm = ds_atm.sel(isobaricInhPa = [center_coords_sfc['mslp'], 700.0, 200.0], method = 'nearest')
    
    center_coords_atm = get_atm_center(ds_atm, center_coords_sfc)

    if not center_coords_atm:
        print('Cannot determine Atmospheric Level Centers, skipping frame')
        logging.info('Cannot determine Atmospheric Level Centers, skipping frame')
        return None, None, None
    
    # computing the necessary values for the follow on frame. Necessary
    # for determining storm motion
    
    center_coords_nxt_step = utils.get_sfc_center(ds_sfc_next_step, next_step = True, land_filter = False)
    if not center_coords_nxt_step:
        print('Can\'t determine follow on timestep storm center. Skipping frame')
        logging.info('Can\'t determine follow on timestep storm center. Skipping frame')
        return None, None, None

    # Opening the sfc data for 24 hours ahead of current frame
    # Necessary for determining RI
    # ds_ri_step = download_and_open(ri_frame_path, **ri_args)
    # ds_ri_step['prmsl'] = ds_ri_step['prmsl']/100
    # center_coords_ri_step = utils.get_sfc_center(ds_ri_step) need to create a separate function for this. don't need a lot of the filters

    # Calculating storm heading
    storm_heading = calc_heading(center_coords_sfc['center_lat'].values, center_coords_sfc['center_lon'].values, 
                                 center_coords_nxt_step['center_lat'].values, center_coords_nxt_step['center_lon'].values)
   
    
    # Calculating the maximum wind value across the whole frame
    max_wind = np.nanmax(np.sqrt(ds_atm['u'].isel(isobaricInhPa = 0).values**2 + ds_atm['v'].isel(isobaricInhPa = 0).values**2))
    print(f'Storm heading is {storm_heading:.2f}')
    frame = to_polar(ds_atm,
                     center_coords_atm,
                     storm_heading = storm_heading)
    
    future_results = utils.get_future_intensity(frame_path)
    

    # Saving to dictionaries to turn into a dataset later
    atm_700_object = {
                    'radius_coords': frame.radius.values,
                    'angle_coords': frame.angle.values,
                    'gh': frame['gh'].isel(isobaricInhPa = 1).values,
                    't': frame['t'].isel(isobaricInhPa = 1).values,
                    'r': frame['r'].isel(isobaricInhPa = 1).values,
                    'u': frame['u'].isel(isobaricInhPa = 1).values,
                    'v': frame['v'].isel(isobaricInhPa = 1).values,
                    'radial_wind': frame['radial_wind'].isel(isobaricInhPa = 1).values,
                    'azimuthal_wind': frame['azimuthal_wind'].isel(isobaricInhPa = 1).values,
                    'center_lat': center_coords_atm['700mb'][1].values,
                    'center_lon': center_coords_atm['700mb'][0].values,
                    'valid_time': frame.isel(isobaricInhPa = 0).valid_time.values,
                    'storm_heading': storm_heading
                    }
    atm_200_object = {
                    'radius_coords': frame.radius.values,
                    'angle_coords': frame.angle.values,
                    'gh': frame['gh'].isel(isobaricInhPa = 2).values,
                    't': frame['t'].isel(isobaricInhPa = 2).values,
                    'r': frame['r'].isel(isobaricInhPa = 2).values,
                    'u': frame['u'].isel(isobaricInhPa = 2).values,
                    'v': frame['v'].isel(isobaricInhPa = 2).values,
                    'radial_wind': frame['radial_wind'].isel(isobaricInhPa = 2).values,
                    'azimuthal_wind': frame['azimuthal_wind'].isel(isobaricInhPa = 2).values,
                    'center_lat': center_coords_atm['200mb'][1].values,
                    'center_lon': center_coords_atm['200mb'][0].values,
                    'valid_time': frame.isel(isobaricInhPa = 0).valid_time.values,
                    'storm_heading': storm_heading
                    }
    sfc_object = {
                    'radius_coords': frame.radius.values,
                    'angle_coords': frame.angle.values,
                    'gh': frame['gh'].isel(isobaricInhPa = 0).values,
                    't': frame['t'].isel(isobaricInhPa = 0).values,
                    'r': frame['r'].isel(isobaricInhPa = 0).values,
                    'u': frame['u'].isel(isobaricInhPa = 0).values,
                    'v': frame['v'].isel(isobaricInhPa = 0).values,
                    'radial_wind': frame['radial_wind'].isel(isobaricInhPa = 0).values,
                    'azimuthal_wind': frame['azimuthal_wind'].isel(isobaricInhPa = 0).values,
                    'center_pressure': center_coords_sfc['mslp'],
                    'max_wind': max_wind,
                    'center_lat': center_coords_atm['Surface_Layer'][1].values,
                    'center_lon': center_coords_atm['Surface_Layer'][0].values,
                    'valid_time': frame.valid_time.values,
                    'storm_heading': storm_heading,
                    '24hr_Max_Winds': np.squeeze(future_results['24hr Max Winds']),
                    '24hr_Min_Central_Pressure': np.squeeze(future_results['24hr Minimum Central Pressure']),
                    }
                    
    
    return sfc_object, atm_700_object, atm_200_object
    
def calc_heading(lat1, lon1, lat2, lon2):
    '''
        Determines storm motion
        Lat1/lon1 are your initial storm center
        Lat2/lon2 is your follow-on storm center
    '''

    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    lon1_rad = np.deg2rad(lon1)
    lon2_rad = np.deg2rad(lon2)

    d_lon = lon2_rad - lon1_rad

    y = np.sin(d_lon) * np.cos(lat2_rad)
    x = (np.cos(lat1_rad) * np.sin(lat2_rad) -
         np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(d_lon)) 
    
    initial_bearing_rad = np.arctan2(y, x)

    initial_bearing_deg = np.rad2deg(initial_bearing_rad)

    bearing = (initial_bearing_deg + 360) % 360

    return bearing 

def rotate_point_around_origin(x, y, angle):
    """
    Rotates a 2D point (x, y) around the origin (0, 0) by a given angle.

    Args:
        x (float): The x-coordinate or x meshgrid of the point(s).
        y (float): The y-coordinate or y meshgrid of the point(s).
        angle_radians (float): The angle of rotation in radians (counter-clockwise).

    Returns:
        a tuple or meshgrid representing the rotated point(s).
    """
    angle = -(np.deg2rad(angle))
    new_x = x * math.cos(angle) - y * math.sin(angle)
    new_y = x * math.sin(angle) + y * math.cos(angle)
    return new_x, new_y

def to_polar(
        ds: xr.Dataset,
        center_coords_list,
        storm_heading: float,
        R_max = 250.0, # Max range in km
        R_step = 0.5, # Range step in km
        A_step = 0.5  # Azimuth step in degrees
    ) -> xr.Dataset:

        '''
            Converts an xarray from a lat/lon grid to polar coordinates.
            Initally a polar grid is generated using set gridpoints.
            Lat/lon values are generated for each polar grid point 
            using a pyproj transformer and the storm center lat/lon
            as the reference. The values are then interpolated from 
            the original lat/lon grid of the dataset to the polar
            lat/lons. 
        '''
        R = 6378137 # radius of Earth in meters

        # Generating the Polar Grid
        new_range = np.arange(0.0, R_max + R_step, R_step)
        new_azimuth = np.arange(0.0, 360.0, A_step)
        T, R = np.meshgrid(new_azimuth, new_range, indexing = 'ij')
     
        X = R * np.cos(T) * 1000 # Convert km to meters
        Y = R * np.sin(T) * 1000
        for i, level in enumerate(ds.isobaricInhPa.values):
            if i == 0:
                origin_lat = center_coords_list['Surface_Layer'][1].values
                origin_lon = center_coords_list['Surface_Layer'][0].values
            if i == 1:
                origin_lat = center_coords_list['700mb'][1].values
                origin_lon = center_coords_list['700mb'][0].values
            if i == 2:
                origin_lat = center_coords_list['200mb'][1].values
                origin_lon = center_coords_list['200mb'][0].values

            ds_atm_lvl = ds.sel(isobaricInhPa = level)
            # Creating lat/lon values based on the polar gridpoints with storm center lat/lon
            local_proj_str = f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +units=m"
            transformer = pyproj.Transformer.from_crs(local_proj_str, "EPSG:4326", always_xy=True)

            # Rotating polar points to match storm heading
            X,Y = rotate_point_around_origin(X, Y, storm_heading)
            target_lons, target_lats = transformer.transform(X, Y)
            
            # Creating the polar dataset
            ds_out = xr.Dataset(
                                coords={
                                        "latitude": (("angle", "radius"), target_lats),
                                        "longitude": (("angle", "radius"), target_lons),
                                        "radius": new_range,
                                        "angle": new_azimuth
                                            }
                                )
            # Interpolating the data from the original datset to the new polar dataset
            regridder = xe.Regridder(ds_atm_lvl, ds_out, 'bilinear')
            polar_out = regridder(ds, ds_atm_lvl)

            # 1. Calculate the true geographic angle of your transformed points.
            # Since target_lons/lats are on Earth, we use the raw distance from origin.
            lon_rad = np.radians(polar_out['longitude'].values)
            lat_rad = np.radians(polar_out['latitude'].values)
            center_lon_rad = np.radians(origin_lon)
            center_lat_rad = np.radians(origin_lat)
            
            dlon = lon_rad - center_lon_rad
            
            # Calculate great-circle bearing (theta)
            y_bearing = np.sin(dlon) * np.cos(lat_rad)
            x_bearing = (np.cos(center_lat_rad)*np.sin(lat_rad) - 
                         np.sin(center_lat_rad)*np.cos(lat_rad)*np.cos(dlon))
            
            theta = np.arctan2(y_bearing, x_bearing)
            # Convert compass bearing
            phi = (np.pi/2.0) - theta

            # 2. Extract the interpolated geographic wind components
            u_geo = polar_out['u'] 
            v_geo = polar_out['v']

            # 3. Rotate to Radial and Azimuthal (Tangential)
            # Radial: positive outward (outflow)
            polar_out['radial_wind'] = u_geo * np.cos(phi) + v_geo * np.sin(phi)
            # Azimuthal: positive counter-clockwise (cyclonic in NH)
            polar_out['azimuthal_wind'] = -u_geo * np.sin(phi) + v_geo * np.cos(phi)

            # Add clear metadata attributes
            polar_out['radial_wind'].attrs = {'units': 'm/s', 'long_name': 'Radial Wind Component'}
            polar_out['azimuthal_wind'].attrs = {'units': 'm/s', 'long_name': 'Azimuthal Wind Component'}

            if i == 0:
                output = polar_out
            else:
                output = xr.concat([output, polar_out], dim = 'isobaricInhPa')
       
        return output

def download_and_open(url,typeOfKey = 'isobaricInhPa', filename="temp.grib2"):

  
  print("Downloading filtered data...")
  response = requests.get(url, stream=True)
  response.raise_for_status()

  with open(filename, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
      f.write(chunk)

  
  ds = xr.open_dataset(filename, 
                       filter_by_keys={'typeOfLevel': typeOfKey },
                       engine='cfgrib')
  return ds        

def get_atm_center(ds_atm, sfc_center):
    current_level_center = sfc_center
    current_lat = current_level_center['center_lat']
    current_lon = current_level_center['center_lon']
    center_points = {'Surface_Layer': [],'700mb': [], '200mb': []}
    for i, level in enumerate(ds_atm.isobaricInhPa):
        ds_atm_lvl = ds_atm.sel(latitude = slice(current_lat - 2, current_lat.values + 2),
                        longitude = slice(current_lon - 2, current_lon + 2))
        ds_atm_lvl = ds_atm_lvl.sel(isobaricInhPa = level)
        if ds_atm_lvl['gh'].isnull().any():
            print('NaN values in storm region. Skipping Frame')
            logging.info('NaN values in atmosphere storm region. Skipping Frame')
             
            return None
        
        hgt_threshold = np.quantile(ds_atm_lvl['gh'], 0.01)
        hgt_clusters = ds_atm_lvl['gh'].where(ds_atm_lvl['gh'] < hgt_threshold)
        ght_clusters_mask = ~hgt_clusters.isnull()

        clusters = scipy.ndimage.label(ght_clusters_mask)
        
        cluster_size = []
        for a in np.arange(clusters[1]):
            cluster_number = a + 1
            cluster_size.append(np.sum(np.where(clusters[0] == cluster_number, 1, 0)))

        try:
            largest_cluster = np.argmax(cluster_size) + 1
        except Exception as e:
            return None
        center_y, center_x = scipy.ndimage.center_of_mass(clusters[0], largest_cluster)

        current_lat = ds_atm_lvl.latitude.isel(latitude =int(np.round(center_y)))
        current_lon = ds_atm_lvl.longitude.isel(longitude =int(np.round(center_x)))

        if i == 0:  
            center_points['Surface_Layer'] = [current_lon, current_lat]
        elif i == 1:
            center_points['700mb'] = [current_lon, current_lat]
        elif i == 2:
            center_points['200mb'] = [current_lon, current_lat]

    return center_points


links_path = 'Model_Data/links/link_list.txt'

starting_timestep = '.f006'
heading_timestep = '.f009'
ri_timestep = '.f030'

save_rate = 50

StormName = 'Melissa'

with open(links_path) as f:
        links = pd.read_csv(f, header = None)

link_list = links[0].values
Hurricane_List = pull_storm_from_links(link_list = link_list, storm_year = "2025", storm_number = "13l")

starting_timestep = '.f006'
heading_timestep = '.f009'
ri_timestep = '.f030'

save_rate = 50

# with open('used_links.txt', 'a') as f:
#             for item in partial_loaded_list:
#                     f.write(item[0] + '\n')
frames_700mb = {
                    'radius_coords': [],
                    'angle_coords': [],
                    'gh': [],
                    't': [],
                    'r': [],
                    'u': [],
                    'v': [],
                    'radial_wind': [],
                    'azimuthal_wind': [],
                    'valid_time': [],
                    'center_lat': [],
                    'center_lon': [],
                    'storm_heading': []
    }

frames_200mb = {
                    'radius_coords': [],
                    'angle_coords': [],
                    'gh': [],
                    't': [],
                    'r': [],
                    'u': [],
                    'v': [],
                    'radial_wind': [],
                    'azimuthal_wind': [],
                    'valid_time': [],
                    'center_lat': [],
                    'center_lon': [],
                    'storm_heading': []
    }
frames_sfc = {
                    'radius_coords': [],
                    'angle_coords': [],
                    'gh': [],
                    't': [],
                    'r': [],
                    'u': [],
                    'v': [],
                    'radial_wind': [],
                    'azimuthal_wind': [],
                    'center_pressure': [],
                    'max_wind': [],
                    'valid_time': [],
                    'center_lat': [],
                    'center_lon': [],
                    'storm_heading': [],
                    'frame_links': [],
                    '24hr_Max_Winds': [],
                    '24hr_Min_Central_Pressure': []

    }

# Running through each frame in the list
frame_number = 0
Valid_frames = False
starting_frame = 1
for i, frame_path in enumerate(Hurricane_List):
   
    with open('used_links.txt', 'a') as f:  
            f.write(frame_path + '\n')
            
    print(f"Link number {i}")
    print(frame_path)
    frame_path = re.sub(ri_timestep, starting_timestep, frame_path)
    sfc_frame, atm_700_frame, atm_200_frame = frame_processing(frame_path, 
                                                               starting_timestep = starting_timestep,
                                                               heading_timestep = heading_timestep,
                                                               ri_timestep = ri_timestep)
    frame = True
    if not sfc_frame:
        frame = False

    
        
    # If a frame is valid, save it.
    if frame:
        frame_number = frame_number + 1
        print(f'Adding frame number {frame_number}')
        if not Valid_frames:
            Valid_frames = True
        frames_700mb['radius_coords'].append(atm_700_frame['radius_coords'])
        frames_700mb['angle_coords'].append(atm_700_frame['angle_coords'])
        frames_700mb['gh'].append(atm_700_frame['gh'])
        frames_700mb['t'].append(atm_700_frame['t'])
        frames_700mb['r'].append(atm_700_frame['r'])
        frames_700mb['u'].append(atm_700_frame['u'])
        frames_700mb['v'].append(atm_700_frame['v'])
        frames_700mb['radial_wind'].append(atm_700_frame['radial_wind'])
        frames_700mb['azimuthal_wind'].append(atm_700_frame['azimuthal_wind'])
        frames_700mb['valid_time'].append(atm_700_frame['valid_time'])
        frames_700mb['center_lat'].append(atm_700_frame['center_lat'])
        frames_700mb['center_lon'].append(atm_700_frame['center_lon'])
        
        
        frames_200mb['radius_coords'].append(atm_200_frame['radius_coords'])
        frames_200mb['angle_coords'].append(atm_200_frame['angle_coords'])
        frames_200mb['gh'].append(atm_200_frame['gh'])
        frames_200mb['t'].append(atm_200_frame['t'])
        frames_200mb['r'].append(atm_200_frame['r'])
        frames_200mb['u'].append(atm_200_frame['u'])
        frames_200mb['v'].append(atm_200_frame['v'])
        frames_200mb['radial_wind'].append(atm_200_frame['radial_wind'])
        frames_200mb['azimuthal_wind'].append(atm_200_frame['azimuthal_wind'])
        frames_200mb['valid_time'].append(atm_200_frame['valid_time'])
        frames_200mb['center_lat'].append(atm_200_frame['center_lat'])
        frames_200mb['center_lon'].append(atm_200_frame['center_lon'])
       

        frames_sfc['radius_coords'].append(sfc_frame['radius_coords'])
        frames_sfc['angle_coords'].append(sfc_frame['angle_coords'])
        frames_sfc['gh'].append(sfc_frame['gh'])
        frames_sfc['t'].append(sfc_frame['t'])
        frames_sfc['r'].append(sfc_frame['r'])
        frames_sfc['u'].append(sfc_frame['u'])
        frames_sfc['v'].append(sfc_frame['v'])
        frames_sfc['radial_wind'].append(sfc_frame['radial_wind'])
        frames_sfc['azimuthal_wind'].append(sfc_frame['azimuthal_wind'])
        frames_sfc['center_pressure'].append(sfc_frame['center_pressure'])
        frames_sfc['max_wind'].append(sfc_frame['max_wind'])
        frames_sfc['valid_time'].append(sfc_frame['valid_time'])
        frames_sfc['center_lat'].append(sfc_frame['center_lat'])
        frames_sfc['center_lon'].append(sfc_frame['center_lon'])
        frames_sfc['storm_heading'].append(sfc_frame['storm_heading'])
        frames_sfc['frame_links'].append(frame_path)
        frames_sfc['24hr_Max_Winds'].append(sfc_frame['24hr_Max_Winds'])
        frames_sfc['24hr_Min_Central_Pressure'].append(sfc_frame['24hr_Min_Central_Pressure'])

        print(sfc_frame['24hr_Min_Central_Pressure'])
       
    # Generating a netcdf file every save_rate frames
    if frame_number%save_rate == 0 and Valid_frames:
        ending_frame = frame_number
       
        ds = xr.Dataset(
                        data_vars = {
                                    'gh': (['levels','valid_time', 'angle', 'radius' ], 
                                        np.stack([frames_sfc['gh'], frames_700mb['gh'], frames_200mb['gh']]),
                                        {'units': 'm', 'long_name': 'Geopotential Height'}),
                                    'rh': (['levels','valid_time','angle', 'radius'], 
                                        np.stack([frames_sfc['r'], frames_700mb['r'], frames_200mb['r']]),
                                        {'units': 'percent humidity', 'long_name': 'Relative Humidity'}),
                                    't': (['levels','valid_time', 'angle', 'radius'], 
                                        np.stack([frames_sfc['t'], frames_700mb['t'], frames_200mb['t']]),
                                        {'units': 'K', 'long_name': 'Air Temperature'}),
                                    'u': (['levels','valid_time', 'angle', 'radius'], 
                                        np.stack([frames_sfc['u'], frames_700mb['u'], frames_200mb['u']]),
                                        {'units': 'm/s', 'long_name': 'U wind'}),
                                    'v': (['levels','valid_time', 'angle', 'radius'], 
                                        np.stack([frames_sfc['v'], frames_700mb['v'], frames_200mb['v']]),
                                        {'units': 'm/s', 'long_name': 'V wind'}),
                                    'az_wind': (['levels','valid_time', 'angle', 'radius'], 
                                        np.stack([frames_sfc['azimuthal_wind'], frames_700mb['azimuthal_wind'], frames_200mb['azimuthal_wind']]),
                                        {'units': 'm/s', 'long_name': 'Azimuthal wind'}),
                                    'rad_wind': (['levels','valid_time', 'angle', 'radius'], 
                                        np.stack([frames_sfc['radial_wind'], frames_700mb['radial_wind'], frames_200mb['radial_wind']]),
                                        {'units': 'm/s', 'long_name': 'Radial wind'}),
                                    'center_pressure': (['valid_time'], 
                                                        frames_sfc['center_pressure'],
                                                        {'units': 'mb', 'long_name': 'Central Minimum Pressure in the Storm Environment'}),
                                    'center_pressure_24hrs': (['valid_time'], 
                                                        frames_sfc['24hr_Min_Central_Pressure'],
                                                        {'units': 'mb', 'long_name': 'Central Minimum Pressure in the Storm Environment 24 hours from initial frame'}),
                                    'max_wind_24hrs': (['valid_time'], 
                                                        frames_sfc['24hr_Max_Winds'],
                                                        {'units': 'mb', 'long_name': 'Maximum Windspeed in the Storm Environment 24 hours from initial frame'}),
                                    'max_wind': (['valid_time'], 
                                                frames_sfc['max_wind'],
                                                {'units': 'm/s', 'long_name': 'Maximum Wind Speed in the Storm Environment'}),
                                    'center_lat': (['levels','valid_time'], 
                                                np.stack([frames_sfc['center_lat'], frames_700mb['center_lat'], frames_200mb['center_lat']]),
                                                {'units': 'degrees latitude', 'long_name': 'Latitude of Storm Center'}),
                                    'center_lon': (['levels','valid_time'], 
                                                np.stack([frames_sfc['center_lon'], frames_700mb['center_lon'], frames_200mb['center_lon']]),
                                                {'units': 'degrees longitude', 'long_name': 'Longitude of Storm Center'}),
                                    'storm_heading': (['valid_time'], 
                                                frames_sfc['storm_heading'],
                                                {'units': 'degrees', 'long_name': 'Storm Heading relative to Lat/Lon Center'}),
                                    'frame_link': (['valid_time'], 
                                                frames_sfc['frame_links'],
                                                { 'long_name': 'Link to grib file hosted on AWS bucket'}),
                                },
                            coords= {
                                    'angle': ('angle',
                                            frames_sfc['angle_coords'][0],
                                            {'units': 'degreees', 'long_name': 'Azimuth around Storm Center'}), 
                                    'radius': ('radius',
                                            frames_sfc['radius_coords'][0],
                                            {'units': 'km', 'long_name': 'Range from Storm Center'}), 
                                    'valid_time': frames_sfc['valid_time'],
                                    'levels': (['sfc', 700, 200])
                                    },
                            attrs= {'history': f'Created on {date.today()}',
                                    'Overview': ('Data originally obtained from the HAFS Repository ' 
                                                    '(https://registry.opendata.aws/noaa-nws-hafs/) '
                                                    'hosted on the AWS Sustainability Data Initiative. '
                                                    'Data was regridded and interpolated from the original '
                                                    'Lat/Lon grid to a regular polar grid centered on storm '
                                                    'center. The center was determined by finding the lowest '
                                                    'surface pressure within the surface pressure field. '
                                                    'Storm heading was determined by finding the center location '
                                                    'in the following time step and calculating the heading between '
                                                    'those two points. Sfc field level data is determined by finding the '
                                                    'pressure level in the dataset nearest the central minimum pressure.')}        
                )
        ds.to_netcdf(f'/data/keeling/a/ianpark3/a597/ThesisStormtrack/Hurricane{StormName}{starting_frame}To{ending_frame}.nc')

        # creating fresh dictionaries.
        frames_700mb = {
                    'radius_coords': [],
                    'angle_coords': [],
                    'gh': [],
                    't': [],
                    'r': [],
                    'u': [],
                    'v': [],
                    'radial_wind': [],
                    'azimuthal_wind': [],
                    'valid_time': [],
                    'center_lat': [],
                    'center_lon': [],
                    'storm_heading': []
                        }

        frames_200mb = {
                    'radius_coords': [],
                    'angle_coords': [],
                    'gh': [],
                    't': [],
                    'r': [],
                    'u': [],
                    'v': [],
                    'radial_wind': [],
                    'azimuthal_wind': [],
                    'valid_time': [],
                    'center_lat': [],
                    'center_lon': [],
                    'storm_heading': []
                        }
        frames_sfc = {
                    'radius_coords': [],
                    'angle_coords': [],
                    'gh': [],
                    't': [],
                    'r': [],
                    'u': [],
                    'v': [],
                    'radial_wind': [],
                    'azimuthal_wind': [],
                    'center_pressure': [],
                    'max_wind': [],
                    'valid_time': [],
                    'center_lat': [],
                    'center_lon': [],
                    'storm_heading': [],
                    'frame_links': [],
                    '24hr_Max_Winds': [],
                    '24hr_Min_Central_Pressure': []

                        }

        starting_frame = ending_frame + 1
        Valid_frames = False
        
ending_frame = frame_number
if Valid_frames:
    ds = xr.Dataset(
                            data_vars = {
                                        'gh': (['levels','valid_time', 'angle', 'radius' ], 
                                            np.stack([frames_sfc['gh'], frames_700mb['gh'], frames_200mb['gh']]),
                                            {'units': 'm', 'long_name': 'Geopotential Height'}),
                                        'rh': (['levels','valid_time','angle', 'radius'], 
                                            np.stack([frames_sfc['r'], frames_700mb['r'], frames_200mb['r']]),
                                            {'units': 'percent humidity', 'long_name': 'Relative Humidity'}),
                                        't': (['levels','valid_time', 'angle', 'radius'], 
                                            np.stack([frames_sfc['t'], frames_700mb['t'], frames_200mb['t']]),
                                            {'units': 'K', 'long_name': 'Air Temperature'}),
                                        'u': (['levels','valid_time', 'angle', 'radius'], 
                                            np.stack([frames_sfc['u'], frames_700mb['u'], frames_200mb['u']]),
                                            {'units': 'm/s', 'long_name': 'U wind'}),
                                        'v': (['levels','valid_time', 'angle', 'radius'], 
                                            np.stack([frames_sfc['v'], frames_700mb['v'], frames_200mb['v']]),
                                            {'units': 'm/s', 'long_name': 'V wind'}),
                                        'az_wind': (['levels','valid_time', 'angle', 'radius'], 
                                            np.stack([frames_sfc['azimuthal_wind'], frames_700mb['azimuthal_wind'], frames_200mb['azimuthal_wind']]),
                                            {'units': 'm/s', 'long_name': 'Azimuthal wind'}),
                                        'rad_wind': (['levels','valid_time', 'angle', 'radius'], 
                                            np.stack([frames_sfc['radial_wind'], frames_700mb['radial_wind'], frames_200mb['radial_wind']]),
                                            {'units': 'm/s', 'long_name': 'Radial wind'}),
                                        'center_pressure': (['valid_time'], 
                                                            frames_sfc['center_pressure'],
                                                            {'units': 'mb', 'long_name': 'Central Minimum Pressure in the Storm Environment'}),
                                        'center_pressure_24hrs': (['valid_time'], 
                                                        frames_sfc['24hr_Min_Central_Pressure'],
                                                        {'units': 'mb', 'long_name': 'Central Minimum Pressure in the Storm Environment 24 hours from initial frame'}),
                                        'max_wind_24hrs': (['valid_time'], 
                                                        frames_sfc['24hr_Max_Winds'],
                                                        {'units': 'mb', 'long_name': 'Maximum Windspeed in the Storm Environment 24 hours from initial frame'}),
                                        'max_wind': (['valid_time'], 
                                                    frames_sfc['max_wind'],
                                                    {'units': 'm/s', 'long_name': 'Maximum Wind Speed in the Storm Environment'}),
                                        'center_lat': (['levels','valid_time'], 
                                                    np.stack([frames_sfc['center_lat'], frames_700mb['center_lat'], frames_200mb['center_lat']]),
                                                    {'units': 'degrees latitude', 'long_name': 'Latitude of Storm Center'}),
                                        'center_lon': (['levels','valid_time'], 
                                                    np.stack([frames_sfc['center_lon'], frames_700mb['center_lon'], frames_200mb['center_lon']]),
                                                    {'units': 'degrees longitude', 'long_name': 'Longitude of Storm Center'}),
                                        'storm_heading': (['valid_time'], 
                                                    frames_sfc['storm_heading'],
                                                    {'units': 'degrees', 'long_name': 'Storm Heading relative to Lat/Lon Center'}),
                                        'frame_link': (['valid_time'], 
                                                    frames_sfc['frame_links'],
                                                    { 'long_name': 'Link to grib file hosted on AWS bucket'}),
                                    },
                                coords= {
                                        'angle': ('angle',
                                                frames_sfc['angle_coords'][0],
                                                {'units': 'degreees', 'long_name': 'Azimuth around Storm Center'}), 
                                        'radius': ('radius',
                                                frames_sfc['radius_coords'][0],
                                                {'units': 'km', 'long_name': 'Range from Storm Center'}), 
                                        'valid_time': frames_sfc['valid_time'],
                                        'levels': (['sfc', 700, 200])
                                        },
                                attrs= {'history': f'Created on {date.today()}',
                                        'Overview': ('Data originally obtained from the HAFS Repository ' 
                                                        '(https://registry.opendata.aws/noaa-nws-hafs/) '
                                                        'hosted on the AWS Sustainability Data Initiative. '
                                                        'Data was regridded and interpolated from the original '
                                                        'Lat/Lon grid to a regular polar grid centered on storm '
                                                        'center. The center was determined by finding the lowest '
                                                        'surface pressure within the surface pressure field. '
                                                        'Storm heading was determined by finding the center location '
                                                        'in the following time step and calculating the heading between '
                                                        'those two points. Sfc field level data is determined by finding the '
                                                        'pressure level in the dataset nearest the central minimum pressure.')}        
                    )
    ds.to_netcdf(f'/data/keeling/a/ianpark3/a597/ThesisStormtrack/Hurricane{StormName}{starting_frame}To{ending_frame}.nc')

