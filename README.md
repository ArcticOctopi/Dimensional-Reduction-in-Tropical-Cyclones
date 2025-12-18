# Characterization of Tropical Cyclones utilizing HAFS-A Model Output

Tropical cyclones have long been characterized by two values - minimum central pressure and maximum surface wind speed - and the official classification of tropical cyclone strength is based entirely on maximum surface wind speed. This is generally an acceptable way of classifying tropical cylones given the destructive power of hurricane force winds. However, examples such as Hurricanes Harvey, Helene and Sandy have demonstrated the potential for destruction that belied their categorical classification at the time.

Given the success the methods such as Empirical Orthogonal Functions and Principal Compontent Analysis have had in identifying large scale synoptic and climatic patterns, this project looks to see whether such data compression would be feasible on a storm-centric basis.

## Workflow:

  Note: There are three main differences to the data when comparing it to the presentation video. The first is there was an error in the storm rotation, causing storms to be rotated away from storm heading rather than towards. Second, there's now a check to ensure that all storms are in the Northern Hemisphere. The opposite rotation of Southern Hemisphere storms could potentially dilute the results. Third, the data is now stored in netcdf files.

  If you're using netcdf datasets generated previously make sure they live in a folder labeled CompleteData that's located inside the main project folder.

  1. The first step in this workflow is to run the ListingFiles notebook. For my useage, I created a separate environment solely for this notebook. Boto3 seems to have compatability issues with some of the othe packages used  in this project, so the easiest solution seemed to be to put it in an environment by itself. The notebook itself requires no input and will generate a text file containing a list of links for the 24 hour timestep grib file of each HAFS-A model run.
  2. Next, run the Data Workflow notebook. This notebook has several variables and parameters that the user can tweek. The primary parameters affecting the dataset are the CMP_MAX and the DROP_VARIABLES. The CMP_MAX determines the necessary storm strength threshold for inclusion into the dataset. Central minimum pressure is one of the main methods of categorizing storm strength, so setting a CMP_MAX of 1000mb will exclude all frames whose minimum central pressure is above 1000mb. The DROP_VARIABLES variable determines which variables to exclude when downloading the data. The remaining dataset variables are listed in a comment immediately above. This notebook will produce a series of netcdf files containing data variables at the 700mb level and sfc level(determined by selecting the level nearest the minimum central pressure of that frame) as well as Maximum Windspeed, Minimum Central Pressure and Center Lat/Lon for that frame.
  3. Finally, run the data analysis notebook. If using v1 netcdfs ensure version_1_data is set to true. The notebook first looks at the distribution of data, noting the composition of storm strength as a function of minimum central pressure. It then looks at the distribution of 700mb data. Temperature data is fairly normal so a standard scaler is used. Geopotential Height and Relative showed a left skewed distribution so I used a Yeo-Johnson transformation to make them more normal. The final data transformation that's performed before analyzing the data is the construction of a field of weights. Due to gridpoint spacing in polar coordinates, weights must be constructed so that features near the center of the field don't dominate the analysis.

  ## Results
  
  For the analysis, I first attempted to construct a multi-variate analysis including geopotential height, temperature, relative humidity at the 700 mb level. To verify retention of information, I computed a multiple linear regression consisting of the top 3 EOFs against frame max wind and frame minimum central pressure. This showed some marginal skill in estimating storm max wind and minimum central pressure for each frame. 

  Next I constructed single variate EOF's of each of Geopotential Height, Temperature and Relative Humidity. Then I performed the same multiple linear regression but using the top 3 principal components of each EOF (9 total) as the predictors. This shows much better skill in estimating storm frame max wind and storm frame minimum central pressure. Lastly I performed a shap analysis so see which EOF is the strongest contributor for each predictand. 

  Not surprisingly, Geopotential Height dominated for both predictands. However, I thought it was interesting that for Max Winds EOF 2 seemed to be the primary influencer while Central Pressure seemed to be more influenced by EOF1. It would also be interesting to see the impact the Temperature and Relative Humidity EOFs have on future storm development. Both are important aspects of tropical cyclone development and I could seem them having an impact on events like Rapid Intensification and Eyewall Replacement Cycles. 

  There would also be a couple changes I think could be made for the analysis portion. I mentioned previously that trying a rotated EOF transformation like Varimax could be important for isolating effects solely caused by the coordinate transformations in the EOF process. I also think further investigation is warranted in the choice of Normalizer for the data. I know ideally for an EOF you have normally distributed data. Given how much the inner core of a tropical cyclone drives the dynamics, maybe it's beneficial to have that inner core weighted stronger than the fringes.

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
    - shap
