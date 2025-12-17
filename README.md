# Characterization of Tropical Cyclones utilizing HAFS-A Model Output

Tropical cyclones have long been characterized by two values - minimum central pressure and maximum surface wind speed - and the official classification of tropical cyclone strength is based entirely on maximum surface wind speed. This is generally an acceptable way of classifying tropical cylones given the destructive power of hurricane force winds. However, examples such as Hurricanes Harvey, Helene and Sandy have demonstrated the potential for destruction that belied their categorical classification at the time.

Given the success the methods such as Empirical Orthogonal Functions and Principal Compontent Analysis have had in identifying large scale synoptic and climatic patterns, this project looks to see whether such data compression would be feasible on a storm-centric basis.

## Workflow:

  1. The first step in this workflow is to run the ListingFiles notebook. For my useage, I created a separate environment solely for this notebook. Boto3 seems to have compatability issues with some of the othe packages used  in this project, so the easiest solution seemed to be to put it in an environment by itself. The notebook itself requires no input and will generate a text file containing a list of links for the 24 hour timestep grib file of each HAFS-A model run.
  2. Next, run the Data Workflow notebook. This notebook has several variables and parameters that the user can tweek. The primary parameters affecting the dataset are the CMP_MAX and the DROP_VARIABLES. The CMP_MAX determines the necessary storm strength threshold for inclusion into the dataset. Central minimum pressure is one of the main methods of categorizing storm strength, so setting a CMP_MAX of 1000mb will exclude all frames whose minimum central pressure is above 1000mb. The DROP_VARIABLES variable determines which variables to exclude when downloading the data. The remaining dataset variables are listed in a comment immediately above.

## Packages required:
 (The environment yaml's may be incompatible with your system depending on your system configuration.)

    - xarray
    - Dask
    - cfgrib
    - boto3
    - s3fs
    - fsspec
    - pyproj
    - h5netcdf
    - matplotlib
    - scikit-learn
    - xesmf
    - eofs
