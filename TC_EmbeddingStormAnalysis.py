import xarray as xr
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
import matplotlib as plt
import cartopy.crs as ccrs
import umap
import cartopy.feature as CFeature
import joblib
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors



class TC_embedding():
    def __init__(self, ds: xr.Dataset, embedding_fields: list):
        self.embedding_fields = embedding_fields
        # Initializing desired fields
        self.dataset = ds
        self.fields = {}
        self.normal_data = {}
        for name in embedding_fields:
            self.fields[f'{name}'] = self.dataset[name].isel(levels = 1)

        # Generating some statistics

        self.mslp = self.dataset['center_pressure'].values
        self.max_winds = self.dataset['max_wind'].values
        self.pressure_change = self.dataset['center_pressure_24hrs'].values - self.mslp
        self.wind_change = self.dataset['max_wind_24hrs'].values - self.max_winds
        self.fl_wind_max = np.max(np.sqrt(self.dataset['rad_wind'].isel(levels = 1).values ** 2 + self.dataset['az_wind'].isel(levels = 1).values ** 2), axis = (1,2))
        self.BestTrack_max_sfcW = self.dataset['best_track_winds'].values
        self.BestTrack_24hr_windchange = self.dataset['best_track_24winds'].values - self.BestTrack_max_sfcW
        self.center_lat = self.dataset.center_lat
        self.center_lon = self.dataset.center_lon
        self.category = []
        self.embedding = np.zeros((len(self.dataset['frame_number'].values), 2))

        # Global Neighbors variable for UMAP embeddings.
        self.NEIGHBORS = 5
        self.MIN_DIST = 0.15
        self.N_COMPONENTS = 2

        # Setting up a custom cmap for wind change.
        wind_change_bins = [-30,-14,-7, 0, 7, 14,50]
        wind_change_colors = [mcolors.CSS4_COLORS.get('red'),
                      mcolors.CSS4_COLORS.get('lightcoral'),
                      mcolors.CSS4_COLORS.get('mistyrose'), 
                      mcolors.CSS4_COLORS.get('lightsteelblue'), 
                      mcolors.CSS4_COLORS.get('royalblue'),
                      mcolors.CSS4_COLORS.get('navy')]
        self.WIND_CHANGE_CMAP = mcolors.ListedColormap(wind_change_colors)
        self.WIND_CHANGE_NORM = mcolors.BoundaryNorm(boundaries=wind_change_bins, ncolors=len(wind_change_colors))

        for wind in self.max_winds:
            if   wind < 33:
                self.category.append(0)
            elif 33 <= wind < 43:
                self.category.append(1)
            elif 43 <= wind < 50:
                self.category.append(2)
            elif 50 <= wind < 58:
                self.category.append(3)
            elif 58 <= wind < 70:
                self.category.append(4)
            elif 70 <= wind:
                self.category.append(5)
        

        # self.baseline_acc_current = {'classifier': 'Linear regression of minimum central pressure to surface max winds',
        #                      'Statistics': self.TC_classifier(feature = np.stack((self.mslp, self.fl_wind_max), axis = -1), target = self.max_winds)
        #                     }
        
        # self.baseline_acc_change = {'classifier': 'Linear regression of minimum central pressure to 24 hour intensity change.',
        #                      'Statistics': self.TC_classifier(feature = np.stack((self.mslp, self.fl_wind_max), axis = -1), target = self.wind_change)
                            # }
        
    def TC_classifier(self, feature, target):

        if len(np.shape(feature)) == 1:
            feature = feature.reshape(-1,1)

        X_train, X_test, y_train, y_test = train_test_split(feature, target)
        model = LinearRegression()
        fitted_model = model.fit(X_train, y_train)
        y_pred = fitted_model.predict(X_test)
        resids = y_pred - y_test
        standard_error = np.std(resids)/np.sqrt(len(y_pred))
        return {'RMSE': np.sqrt(mean_squared_error(y_test,y_pred)), 
                'Coefficient of Determination': fitted_model.score(X_test, y_test),
                'Standard Error': standard_error}
        
    def Show_baseline(self):
        print(self.baseline_acc_current)
        print(self.baseline_acc_change)

    def Normalize_data(self, method = 'MinMax'):

        '''
            Methods: 'MinMax', 'Gauss'
        '''
        if   method == 'MinMax':
            

            for key in self.fields.keys():
                self.normal_data[key] = self._minmax_data_normalizer(self.fields[key])
        elif   method == 'Gauss':
                    
        
                    for key in self.fields.keys():
                        self.normal_data[key] = self._standard_data_normalizer(self.fields[key])
            

    def _standard_data_normalizer(self, da):
        frame_number, angle, radius = da.shape
        flattened_arrary = da.values.reshape(-1,1)
        scaler = StandardScaler()
        data_normalized_flat = scaler.fit_transform(flattened_arrary)
        data_normalized = data_normalized_flat.reshape(frame_number, angle, radius)

        return data_normalized    
    
    def _minmax_data_normalizer(self, da):
        frame_number, angle, radius = da.shape
        flattened_arrary = da.values.reshape(-1,1)
        scaler = MinMaxScaler()
        data_normalized_flat = scaler.fit_transform(flattened_arrary)
        data_normalized = data_normalized_flat.reshape(frame_number, angle, radius)

        return data_normalized

    def Plot_locations(self, shading = None, cmap = 'RdBu'):
        '''
        shadings: 'mslp', 'max_wind', 'max_wind_change', 'px_change', 'categories
        '''

        if shading == 'mslp':
            shading = self.mslp
        elif shading == 'max_wind':
            shading = self.max_winds
        elif shading == 'max_wind_change':
            shading = self.wind_change
        elif shading == 'px_change':
            shading = self.pressure_change
        elif shading == 'categories':
            shading = self.category
        else:
            shading = None

        
        
        lats = self.dataset['center_lat'].isel(levels = 0).values
        lons = self.dataset['center_lon'].isel(levels = 0).values


        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,8), subplot_kw = {'projection': ccrs.PlateCarree()})

        im = ax.scatter(lons, lats, transform = ccrs.PlateCarree(), s = 3, c = shading, cmap = cmap)
        ax.add_feature(CFeature.COASTLINE)
        ax.set_title('Storm Locations')
        fig.colorbar(im, location = 'bottom')

    def Plot_strengths(self):

        bin_names = ['Tropical Storms', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']

        plt.hist(self.category, rwidth = 0.75)
        plt.xticks([0,1,2,3,4,5], bin_names)
        plt.title('Distribution of Storm Strength within the Dataset')

    def Fit_umap(self, neighbors = 4, metric = 'correlation', min_dist = 0.1, n_components = 2):

        if self.normal_data:
            num_fields = len(self.normal_data.keys())
            dataset_size = len(self.dataset['frame_number'].values)
            array = []
            first = True
            for field in self.normal_data.keys():
                if first:
                    array = self.normal_data[field].reshape(dataset_size, 720 * 501)
                    first = False
                else:
                    array = np.concat((array, self.normal_data[field].reshape(dataset_size, 720 * 501)), axis = 1)
            
            # print(f'Input Shape: {np.shape(array)}')
            # pca = IncrementalPCA().fit_transform(array)
            # print(f'PCA components shape: {np.shape(pca)}')
            # print(f'Loadings shape: {np.shape(loadings)}')
            # rotated_loadings, rotation_matrix = rotate_factors(loadings, method='varimax')
            # print(rotated_loadings)

            loadings = array


            self.embedder = umap.UMAP(n_components = n_components, 
                                      n_neighbors = neighbors, 
                                      random_state = 12, 
                                      init = 'pca',
                                      metric = metric,
                                      min_dist = min_dist) 
            self.embedding = self.embedder.fit_transform(loadings) 
            self.model = self.embedder.fit(loadings)
            return(self.embedding)    
        else:
            dataset_size = len(self.dataset['frame_number'].values)
            num_fields = len(self.fields.keys())
            array = []
            first = True
            for field in self.fields.keys():
                if first:
                    array = self.fields[field].values.reshape(dataset_size, 720 * 501)
                    first = False
                else:
                    array = np.concat((array, self.fields[field].values.reshape(dataset_size, 720 *501)), axis = 1) 

            
            # print(f'Input Shape: {np.shape(array)}')
            # pca = IncrementalPCA().fit_transform(array)
            # print(f'PCA components shape: {np.shape(pca)}')
            # loadings = pca
            # print(f'Loadings shape: {np.shape(loadings)}')
            # rotated_loadings, rotation_matrix = rotate_factors(loadings, method='varimax')
            # print(rotated_loadings)
            loadings = array

            self.embedder = umap.UMAP(n_components = 2, n_neighbors = neighbors, random_state = 12, init = 'pca')
            self.embedding = self.embedder.fit_transform(loadings)
            self.model = self.embedder.fit(loadings)

            self.embed_metric = metric
            self.embed_min_dist = min_dist 
            return(self.embedding) 
    
    def transform(self):
        num_fields = len(self.normal_data.keys())
        dataset_size = len(self.dataset['valid_time'].values)
        array = []
        first = True
        for field in self.normal_data.keys():
            print(field)
            if first:
                array = self.normal_data[field].reshape(dataset_size, 720 * 501)
                first = False
            else:
                array = np.concat((array, self.normal_data[field].reshape(dataset_size, 720 * 501)), axis = 1)
        
        loadings = array
        
        print(np.shape(array))
        self.embedding = self.model.transform(loadings)
        return(self.embedding)


    def Fit_pumap(self, neighbors = 4, n_components = 2):
    
            if self.normal_data:
                num_fields = len(self.normal_data.keys())
                dataset_size = len(self.dataset['frame_number'].values)
                array = []
                first = True
                for field in self.normal_data.keys():
                    if first:
                        array = self.normal_data[field].reshape(dataset_size, 720 * 501)
                        first = False
                    else:
                        array = np.concat((array, self.normal_data[field].reshape(dataset_size, 720 * 501)), axis = 1)

                # print(f'Input Shape: {np.shape(array)}')
                # pca = IncrementalPCA().fit_transform(array)
                # print(f'PCA components shape: {np.shape(pca)}')
                # loadings = pca
                # print(f'Loadings shape: {np.shape(loadings)}')
                # rotated_loadings, rotation_matrix = rotate_factors(loadings, method='varimax')
                # print(loadings)

                loadings = array
                
                self.embedder = umap.ParametricUMAP(n_components = n_components, 
                                                    n_neighbors = neighbors,
                                                    metric='correlation',
                                                    init = 'pca',
                                                    
                                                                                            
                                                    ) 
               
                self.embedding = self.embedder.fit_transform(loadings) 
                self.model = self.embedder.fit(loadings)
                return(self.embedding)    
            else:
                dataset_size = len(self.dataset['frame_number'].values)
                num_fields = len(self.fields.keys())
                array = []
                first = True
                for field in self.fields.keys():
                    if first:
                        array = self.fields[field].values.reshape(dataset_size, 720 * 501)
                        first = False
                    else:
                        array = np.concat((array, self.fields[field].values.reshape(dataset_size, 720 *501)), axis = 1) 
                
                # print(f'Input Shape: {array}')
                # pca = IncrementalPCA(n_components=100).fit_transform(array)
                # print(f'PCA components shape: {pca}')
                # loadings = pca
                # print(f'Loadings shape: {np.shape(loadings)}')
                # rotated_loadings, rotation_matrix = rotate_factors(loadings, method='varimax')
                # print(rotated_loadings)

                loadings = array
                self.embedder = umap.ParametricUMAP(n_components = n_components, 
                                                    n_neighbors = neighbors,  
                                                    metric = 'correlation',
                                                    init = 'pca'                                      
                                                    ) 
                self.embedding = self.embedder.fit_transform(loadings)
                self.model = self.embedder.fit(loadings)
    
                return(self.embedding) 

    def save_embedding(self, embedding_path: str):
        '''
            Checks to make sure an embedder has actually been created and 
            then saves it to a joblib file.

            embedding path: the location and desired name of the save file. No extension tag required.
        '''

        if self.model:
            joblib.dump(self.model, embedding_path + '.joblib')

    def load_embedding(self, embedding_path:str):

        '''
            Takes a previously created embedding, saved in a joblib file, and loads it.
        '''

        self.model = joblib.load(embedding_path)
    
            
    def show_inverse(self, corners):

        p_tl, p_bl, p_tr, p_br = corners  

        x = np.linspace(0,1,10)  
        y = np.linspace(0,1,10)

        X_grid, Y_grid = np.meshgrid(x,y)
        X_grid = X_grid[..., np.newaxis]
        Y_grid = Y_grid[..., np.newaxis]
        test_pts = (
                    (1 - X_grid) * (1 - Y_grid) * p_tl
                    + X_grid * (1 - Y_grid) * p_tr
                    + X_grid * Y_grid * p_br
                    + (1 - X_grid) * Y_grid * p_bl
                )  
        inv_transformed_points = self.embedder.inverse_transform([[12.5,16]])

        fig = plt.figure(figsize=(12,6))
        gs = GridSpec(10, 20, fig)
        scatter_ax = fig.add_subplot(gs[:, :10])
        digit_axes = np.zeros((10, 10), dtype=object)
        for i in range(10):
            for j in range(10):
                digit_axes[i, j] = fig.add_subplot(gs[i, 10 + j])

        # Use umap.plot to plot to the major axis
        # umap.plot.points(mapper, labels=labels, ax=scatter_ax)
        scatter_ax.scatter(self.embedding_[:, 0], self.embedding_[:, 1],
                           c=self.BestTrack_max_sfcW, cmap='Spectral', s=0.1)
        scatter_ax.set(xticks=[], yticks=[])

        # Plot the locations of the text points
        scatter_ax.scatter(test_pts[:, 0], test_pts[:, 1], marker='x', c='k', s=15)

        # Plot each of the generated digit images
        for i in range(10):
            for j in range(10):
                digit_axes[i, j].imshow(inv_transformed_points[i*10 + j].reshape(28, 28))
                digit_axes[i, j].set(xticks=[], yticks=[])

    def plot_embedding(self, title = None):
        fig = plt.figure(figsize = (16,12), layout = 'constrained')
        subfigs = fig.subfigures(nrows=2, ncols=1)
        axs_row1 = subfigs[0].subplots(nrows = 1, ncols = 2)
        axs_row2 = subfigs[1].subplots(nrows = 1, ncols = 2)


        im_wind = axs_row1[0].scatter(self.embedding[:,0], self.embedding[:,1], 
                                    c = self.BestTrack_max_sfcW, cmap = 'Spectral',
                                    vmin = np.min(self.BestTrack_max_sfcW), vmax = np.max(self.BestTrack_max_sfcW))
        im_windchange = axs_row1[1].scatter(self.embedding[:,0], self.embedding[:,1], c = self.BestTrack_24hr_windchange, cmap = self.WIND_CHANGE_CMAP, norm = self.WIND_CHANGE_NORM)
        subfigs[0].suptitle('Shading: Best Track Data')

        axs_row2[0].scatter(self.embedding[:,0], self.embedding[:,1], 
                                    c = self.max_winds,  cmap = 'Spectral',
                                    vmin = np.min(self.BestTrack_max_sfcW), vmax = np.max(self.BestTrack_max_sfcW))
        axs_row2[1].scatter(self.embedding[:,0], self.embedding[:,1],
                                            c = self.wind_change, cmap = self.WIND_CHANGE_CMAP, 
                                            norm = self.WIND_CHANGE_NORM)# vmin = np.min(embed_4.wind_change), vmax = np.max(embed_4.wind_change))
        subfigs[1].suptitle('Shading: HAFS-A Data')

        fig.colorbar(im_windchange, ax = axs_row2[1], location = "bottom", label = "24 hour Windspeed Change (m/s)")
        fig.colorbar(im_wind, ax = axs_row2[0], location = "bottom", label = "Max Windspeed (m/s)")

        if title:
        
            fig.suptitle(f'''{title}
                            TC Embedding | fields: GH, Radial Winds, Azimuthal Winds, Temperature | neighbors: {self.NEIGHBORS} 
                             N = {len(self.embedding[:,0])} | Metric = "Correlation" | Min_dist ={self.MIN_DIST}''')
        else:
            fig.suptitle(f'''TC Embedding | fields: GH, Radial Winds, Azimuthal Winds, Temperature | neighbors: {self.NEIGHBORS} 
                             N = {len(self.embedding[:,0])} | Metric = "Correlation" | Min_dist ={self.MIN_DIST}''')
        
    def plot_composites(self, x_low = None, x_high = None, y_low = None, y_high = None, cluster_filter = None):

        level = 1
        if not any(cluster_filter):
            filter_y = (self.embedding[:,1] > y_low) & (self.embedding[:,1] < y_high)
            filter_x = (self.embedding[:,0] > x_low) & (self.embedding[:,0] < x_high)
            cluster_filter = filter_x * filter_y
        else:
            cluster_filter = cluster_filter

        # Creating Composite Fields
        composite_filter = cluster_filter
        angle = self.dataset.angle.values
        radius = self.dataset.radius.values + 90
        radius = np.where( radius > 360, radius - 360, radius)

        ds_gph = self.dataset['gh'].isel(levels = level).values
        ds_t = self.dataset['t'].isel(levels = level).values
        ds_rad_winds = self.dataset['rad_wind'].isel(levels = level).values
        ds_az_winds = self.dataset['az_wind'].isel(levels = level).values
        max_winds = self.dataset['max_wind'].isel(levels = 0).values


        cluster_wind_change = self.BestTrack_24hr_windchange[composite_filter]
        avg_cluster_wind_change = np.nanmean(cluster_wind_change)
        std_cluster_wind_change = np.nanstd(cluster_wind_change)
        cluster_winds = self.BestTrack_max_sfcW[composite_filter]
        avg_cluster_wind = np.nanmean(cluster_winds)
        std_cluster_wind = np.nanstd(cluster_winds)
        cluster_size =np.sum(composite_filter)
        cluster_lats = self.center_lat[composite_filter] #0 for selecting the surface center lat
        cluster_lons = self.center_lon[composite_filter] #0 for selecting the surface center lon


        sample_pop_ul = avg_cluster_wind + std_cluster_wind*2
        sample_pop_ll = avg_cluster_wind - std_cluster_wind*2
        print(f'Upper sample limit: {sample_pop_ul}')
        print(f'Lower sample limit: {sample_pop_ll}')
        

        #### GH Composite ####
        mean_field_gh = np.nanmean(self.dataset['gh'].isel(levels = 1).where((self.dataset.max_wind.isel(levels = 1) < sample_pop_ul) & (self.dataset.max_wind.isel(levels = 1) > sample_pop_ll)), axis = 0)
        field_anomalies_gh = ds_gph - mean_field_gh
        cluster_field_anomalies_gh = field_anomalies_gh[composite_filter, :, :]
        cluster_field_mean_anom_gh = np.mean(cluster_field_anomalies_gh, axis = 0)
        cluster_field_mean_gh = np.mean(ds_gph[composite_filter,:,:], axis = 0)

        #### T Composite #####
        mean_field_t = np.nanmean(self.dataset['t'].isel(levels = 1).where((self.dataset.max_wind.isel(levels = 1) < sample_pop_ul) & (self.dataset.max_wind.isel(levels = 1) > sample_pop_ll)), axis = 0)
        field_anomalies_t = ds_t - mean_field_t
        cluster_field_anomalies_t = field_anomalies_t[composite_filter, :, :]
        cluster_field_mean_anom_t = np.mean(cluster_field_anomalies_t, axis = 0)
        cluster_field_mean_t = np.mean(ds_t[composite_filter,:,:], axis = 0)

        #### Rad Winds Composite #####
        mean_field_rad_winds = np.nanmean(self.dataset['rad_wind'].isel(levels = 1).where((self.dataset.max_wind.isel(levels = 1) < sample_pop_ul) & (self.dataset.max_wind.isel(levels = 1) > sample_pop_ll)), axis = 0)
        cluster_field_anomalies_rad_winds = ds_rad_winds[composite_filter, :, :] - mean_field_rad_winds
        cluster_field_mean_anom_rad_winds = np.mean(cluster_field_anomalies_rad_winds, axis = 0)
        cluster_field_mean_rad_winds = np.mean(ds_rad_winds[composite_filter,:,:], axis = 0)

        #### Az Winds Composite #####
        mean_field_az_winds = np.nanmean(self.dataset['az_wind'].isel(levels = 1).where((self.dataset.max_wind.isel(levels = 1) < sample_pop_ul) & (self.dataset.max_wind.isel(levels = 1) > sample_pop_ll)), axis = 0)
        cluster_field_anomalies_az_winds = ds_az_winds[composite_filter, :, :] - mean_field_az_winds
        cluster_field_mean_anom_az_winds = np.mean(cluster_field_anomalies_az_winds, axis = 0)
        cluster_field_mean_az_winds = np.mean(ds_az_winds[composite_filter,:,:], axis = 0)

        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=(12, 12), subplot_kw= {'projection':'polar'})
        Theta, R = np.meshgrid(angle, radius)
        rad_anom_im = ax[0,0].contourf(angle, radius, cluster_field_mean_anom_rad_winds.T, cmap = 'RdBu')
        # ax[0,0].scatter(sig_upper_x_gh[::50], sig_upper_y_gh[::50], alpha = 0.1, color = 'black', marker = 'x', s = 10)
        # ax[0,0].scatter(sig_lower_x_gh[::50], sig_lower_y_gh[::50], alpha = 0.1, color = 'black', marker = 'x', s = 10)
        ax[0,0].grid(False)
        # ax[0,0].set_theta_direction(-1)
        # ax[0,0].set_theta_offset(np.pi/2)
        ax[0,0].set_title('Cluster Radial Winds vs Same Strength Storms ')
        fig.colorbar(rad_anom_im, ax = ax[0,0], location = 'bottom', label = 'Radial Wind Anomaly (m/s)')

        az_anom_im = ax[0,1].contourf(angle, radius, cluster_field_mean_anom_az_winds.T, cmap = 'viridis')
        # ax[0,1].scatter(sig_upper_x_t[::60], sig_upper_y_t[::60], alpha = 0.15, color = 'black', marker = '*', s = 5)
        # ax[0,1].scatter(sig_lower_x_t[::60], sig_lower_y_t[::60], alpha = 0.15, color = 'black', marker = '*', s = 5)
        ax[0,1].grid(False)
        # ax[0,1].set_theta_direction(-1)
        # ax[0,1].set_theta_offset(np.pi/2)
        ax[0,1].set_title('Cluster Azimuthal Wind vs Same Strength Storms ')
        fig.colorbar(az_anom_im, ax = ax[0,1], location = 'bottom', label = 'Azimuthal Winds vs Same Strength Storms (m/s)')

        ax00_inset = ax[0,0].inset_axes([1.15, .66, .5, .4])
        ax00_inset.scatter(self.embedding[:,0], self.embedding[:,1], color = 'k', s = 5, alpha = 0.05)
        ax00_inset.scatter(self.embedding[composite_filter,0], self.embedding[composite_filter,1], 
                        c = self.BestTrack_24hr_windchange[cluster_filter], cmap = self.WIND_CHANGE_CMAP, norm = self.WIND_CHANGE_NORM, s = 5,
                        )


        height_anom_im = ax[1,0].contourf(angle, radius, cluster_field_mean_anom_gh.T, cmap = 'coolwarm')
        # ax[1,0].scatter(sig_gh_x[::30], sig_gh_y[::30], alpha = 0.1, color = 'black', marker = 'x', s = 10)
        # ax[1,0].scatter(sig_lower_x_gh[::40], sig_lower_y_gh[::40], alpha = 0.1, color = 'black', marker = 'x', s = 10)
        ax[1,0].grid(False)
        # ax[1,0].set_theta_direction(-1)
        # ax[1,0].set_theta_offset(np.pi/2)
        ax[1,0].set_title('Cluster Geopotential Height vs Same Strength Storms ')
        fig.colorbar(height_anom_im, ax = ax[1,0], location = 'bottom', label = 'Geopotential Height (m)')

        ax10_inset = ax[1,0].inset_axes([.9, .5, 1, 1], projection =  ccrs.PlateCarree())
        ax10_inset.scatter(cluster_lons, cluster_lats, transform = ccrs.PlateCarree(), 
                        c = self.BestTrack_24hr_windchange[cluster_filter], cmap = self.WIND_CHANGE_CMAP, norm = self.WIND_CHANGE_NORM, s = 5,)
        ax10_inset.set_xlim(-180, 180)
        ax10_inset.set_ylim(-10,50)
        ax10_inset.add_feature(CFeature.COASTLINE)

        temp_anom_im = ax[1,1].contourf(angle, radius, cluster_field_mean_anom_t.T, cmap = 'inferno')
        # ax[1,1].scatter(sig_t_x[::20], sig_t_y[::20], alpha = 0.05, color = 'black', marker = '*', s = 1)
        # ax[1,1].scatter(sig_lower_x_t, sig_lower_y_t, alpha = 0.05, color = 'black', marker = '*', s = 1)
        ax[1,1].grid(False)
        # ax[1,1].set_theta_direction(-1)
        # ax[1,1].set_theta_offset(np.pi/2)
        ax[1,1].set_title('Cluster Temperature vs Same Strength Storms ')
        fig.colorbar(temp_anom_im, ax = ax[1,1], location = 'bottom', label = 'Temperature Anomaly (C)')

        fig.suptitle(f'''Cluster Average Windspeed: {avg_cluster_wind:.2f} m/s (std: {std_cluster_wind:.2f} m/s)
                         Cluster Average 24hr Storm Intensity Change: {avg_cluster_wind_change:.2f} m/s  (std: {std_cluster_wind_change:.2f} m/s)
                         Cluster Size: {cluster_size}''')