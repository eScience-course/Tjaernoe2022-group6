import xarray as xr, pandas as pd, numpy as np, math, time   # For data handling
import matplotlib.pyplot as plt, cartopy.crs as ccrs         # For plotting
import statsmodels.formula.api as sm                         # For regression
from joblib import Parallel, delayed                         # For parallizing
from datetime import datetime                                # For time computations
from ipywidgets import interact, interactive, fixed, widgets # For interactive plotting
import intake

# s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
#                        client_kwargs=dict(endpoint_url="https://rgw.met.no"))

# def read_file(file):
#     return s3.open(file)

# def create_xr(file):
#     return xr.open_dataset(file)

def get_bucket_data(variable='chlos', time_res='monthly', model='NorESM2-LM', experiment='hist', 
                    member_id='*', period='*', chunks=None, noChunks=True, last_n_files=0, 
                    parallel=False, filepath=None):
    '''
        Import model data from the s3 storage bucket. 
    Args:
        variable     [str]     :  Name of the variable to access
        time_res     [str]     :  Time resolution of data, can be 'daily' or 'monthly'
        model        [str]     :  Name of the model the output is from, e.g. 'NorESM2-LM' or 'CESM2'
        experiment   [str]     :  Name of the model run experiment, e.g. 'historical'
        member_id    [str]     :  Name of the model member id to get data from, default is all
        period       [str]     :  The period to import data from, default is all
        chunks       [int]     :  Number of timesteps for each chunk if data should be chunked
        noChunks     [bool]    :  Whether to chunk the dask array or not
        last_n_files [int]     :  Number of captured files to read counting from last, default is all
        filepath     [str]     :  Path to file on s3 for simple read instead
    Returns:
        dS           [DataSet] :  xarray.DataSet of the collected data
    '''
    
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
                               client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    base_path = 's3://escience2022/Ada/'
    
    if chunks is None and not noChunks:
        chunks = {'time': chunks} #math.ceil((365 * (5 + 10 * (17 - 1)) + 1) / 16)}
    
    if filepath is None:
        if time_res == 'monthly':
            name = 'Omon'
        else:
            name = 'Oday'
        if member_id == period:
            full_path = base_path + f'{time_res}/{variable}_{name}_{model}_{experiment}_*.nc'
        else:
            full_path = base_path + f'{time_res}/{variable}_{name}_{model}_{experiment}_{member_id}_{period}.nc'

        print('Request: ' + full_path)
    else:
        full_path = base_path + filepath
        remote_files = s3.glob(full_path)

        fileset = [s3.open(file) for file in remote_files[-last_n_files:]]

        # fileset = Parallel(n_jobs=2)(delayed(read_file)(file) for file in remote_files[-last_n_files:])

        #xrs = Parallel(n_jobs=2)(delayed(create_xr)(file) for file in fileset)
        #data = xr.concat(xrs, dim='time')

        if last_n_files != 0:
            dS = xr.open_mfdataset(fileset, concat_dim='time', combine='nested', chunks=chunks, parallel=parallel)
        else:
            dS = xr.open_mfdataset(fileset, concat_dim='time', combine='nested', chunks=chunks, parallel=parallel)
        return dS


def get_areacello(model='NorESM2-LM'):
    '''
        Downloads or reads ocean gridcell area data for the desired model
    Args:
        model     [str]     :  Name of the model, default is 'NorESM2-LM'
    Returns:
        areacello [DataSet] :  xarray.DataArray of model ocean grid cell area
    '''
    try:
        areacello = consistent_naming(xr.open_dataset(f'areacello_{model}.nc'))
        print('Found local areacello NetCDF')
    except:
        print('Areacello not stored locally. Getting from cloud...')
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=['areacello'], member_id=['r1i1p1f1'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
        area = ds_dict[list(ds_dict.keys())[0]]
        areacello = consistent_naming(area)
        areacello.to_netcdf(f'areacello_{model}.nc')
    return areacello.areacello.squeeze()

def clip_to_region2(_ds, minj=340, maxj=380, mini=110, maxi=145, model='NorESM2'):
    '''
        Clip dataset to a specific region by model indices
    Args:
        _ds       [DataSet]  : Xarray.DataSet or DataArray of data to slice
        minj      [int]      : Minimum x index
        maxj      [int]      : Maximum x index
        mini      [int]      : Minimum y index
        maxi      [int]      : Maximum y index
    Returns:
        clip      [DataSet]  : Clipped Xarray.DataSet or DataArray
    '''
    if model == 'NorESM2':
        clip = _ds.isel(j=slice(minj, maxj), i=slice(mini, maxi))
    elif model == 'CESM2':
        clip = _ds.isel(nlon=slice(mini, maxi), nlat=slice(minj, maxj))
    return clip

def clip_to_region(_ds, minlon=20, maxlon=60, minlat=70, maxlat=90):
    '''
        Clip dataset to a specific region by latitude and longitude
    Args:
        _ds       [DataSet]  : Xarray.DataSet or DataArray of data to slice
        minlon    [int]      : Minimum longitude
        maxlon    [int]      : Maximum longitude
        minlat    [int]      : Minimum latitude
        maxlat    [int]      : Maximum latitude
    Returns:
        clip      [DataSet]  : Clipped Xarray.DataSet or DataArray
    '''
    try:
        # If this works then longitude is called 'lon'
        current_minlon = _ds.lon.values.min()
        if current_minlon > -100:
            _ds = shift_longitude(_ds)
        name_lat = 'lat'
        name_lon = 'lon'
        #_ds = _ds.sel(lon=slice(minlon, maxlon))
        #_ds = _ds.sel(lat=slice(minlat, maxlat))
    except:
        current_minlon = _ds.longitude.values.min()
        if current_minlon > -100:
            _ds = shift_longitude(_ds)
        name_lat = 'latitude'
        name_lon = 'longitude'
        #_ds = _ds.sel(longitude=slice(minlon, maxlon), method='nearest')
        #_ds = _ds.sel(latitude=slice(minlat, maxlat), method='nearest')
        
    clip = _ds.where((_ds[name_lat] > minlat) & 
                     (_ds[name_lat] < maxlat) & 
                     (_ds[name_lon] > minlon) & 
                     (_ds[name_lon] < maxlon), 
                     drop=True)
    return clip

    
def clip_to_months(_ds, start_month, stop_month):
    '''
        Cut data from unwanted months
    Args:
        _ds         [DataSet]  : Xarray.DataSet or DataArray of data to slice
        start_month [int]      : First (included) month to keep
        stop_month  [int]      : Last (included) month to keep
    Returns:
        clip      [DataSet]  : Clipped Xarray.DataSet or DataArray
    '''
    return _ds.where(_ds.time.dt.month.isin([i for i in range(start_month, stop_month + 1)]))

def shift_longitude(_ds):
    '''
        Transfer longitude from 0-360 degrees standard to -180-180 degrees standard
    Args:
        _ds         [DataSet]  : Xarray.DataSet or DataArray of data to slice
    Returns:
        shifted     [DataSet]  : Longitude shifted Xarray.DataSet or DataArray
    '''
    shifted = _ds.copy()
    try:
        shifted.coords['longitude'] = (shifted['longitude'] + 180) % 360 - 180
    except:
        shifted.coords['lon'] = (shifted['lon'] + 180) % 360 - 180
    return shifted
    
def regional_average(data, model='NorESM2-LM', clip_coords=[20, 60, 70, 90]):
    '''
        Compute the weighted average across a spatial region
    Args:
        _ds         [DataSet]   : Xarray.DataSet or DataArray of data to slice
        model       [str]       : Name of the model the output is from. Used to import gridcell area information
        clip_coords [list-like] : Bounding [minlon, maxlon, minlat, maxlat] coordinates for the region
    Returns:
        mean        [DataSet]   : Xarray.DataSet or DataArray, with 2 less dimensions than input
    '''
    area = clip_to_region(get_areacello(model), *clip_coords).squeeze()
    mean = (data*area).sum(dim=('i','j'))/area.sum(dim=('i','j'))
    return mean

def time_anomaly(_ds, first_start, first_stop, last_start, last_stop):
    '''
        Compute the anomaly between two specified periods
    Args:
        _ds         [DataSet]  : Xarray.DataSet or DataArray of data
        first_start [int]      : Start of first period as model time index
        last_start  [int]      : End of first period as model time index
        first_stop  [int]      : Start of last period as model time index
        last_stop   [int]      : End of last period as model time index
    Returns:
        anomaly     [DataSet]  : Xarray.DataSet or DataArray with the anomaly
    '''
    anomaly = _ds.isel(time=slice(last_start, last_stop)).mean(dim='time') - _ds.isel(time=slice(first_start, first_stop)).mean(dim='time')
    return anomaly


def find_peak_dates(_da):
    '''
        Compute the dates each year of input where the variable peaks
    Args:
        _ds       [DataSet]   : Xarray.DataArray of variable data
    Returns:
        df        [DataFrame] : Pandas.DataFrame containing doy of the peaks each year  
    '''
    _dat = _da.copy()
    startyear, stopyear = _dat.isel(time=0).time.dt.year.values, _dat.isel(time=-1).time.dt.year.values
    _dat['time']  = _dat.time.to_dataframe().index.to_datetimeindex()
    _dat_grouped = _dat.groupby('time.year')
    _ds_dates = [_dat_grouped[year].idxmax(dim='time').values for year in _dat_grouped.groups.keys()]
    shape = np.shape(_ds_dates)
    df = xr.DataArray(_ds_dates).to_dataframe('date')
    df['dayofyear'] = pd.to_datetime(df['date']).dt.dayofyear
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    _xrds = xr.DataArray(df['dayofyear'].values.reshape(shape), dims=['time', 'j', 'i'], 
                  coords=dict(longitude=(["j", "i"], _dat.lon.values), 
                              latitude=(["j", "i"], _dat.lat.values), 
                              time=np.arange(startyear, stopyear+1)))
    return _xrds

def find_peak_dates2(_da):
    '''
        Compute the dates each year of input where the variable peaks
    Args:
        _ds       [DataSet]   : Xarray.DataArray of variable data
    Returns:
        df        [DataFrame] : Pandas.DataFrame containing doy of the peaks each year  
    '''
    _dat = _da.copy()
    _dat['time']  = _dat.time.to_dataframe().index.to_datetimeindex()
    _dat = _dat.groupby('time.year')
    _ds_dates = [_dat[year].idxmax(dim='time').values[0] for year in _dat.groups.keys()]
    df = xr.DataArray(_ds_dates).to_dataframe('date')
    df['dayofyear'] = pd.to_datetime(df['date']).dt.dayofyear
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df

def regression(data, x, y, summary=False):
    '''
        Computes the Ordinary Least Squares regression for the input DataFrame
    Args:
        data     [DataFrame]  : Pandas.DataFrame wiht columns corresponding to x & y
        x        [str]        : Name of the column in 'data' to use as x-values
        y        [str]        : Name of the column in 'data' to use as y-values
        summary  [bool]       : Return the summary of the OLS computation, default is False
    Returns:
        a        [float]      : Slope of the computed regression line
        b        [float]      : Intercept of the computed regression line
        summary  [str]        : If 'summary' is True, return a summary for the OLS
    '''
    reg = sm.ols(formula=f'{y} ~ {x}', data=data).fit()
    #print(reg.summary)
    b, a = reg.params
    if summary == True:
        summary = reg.summary()
        return a, b, summary
    else:
        return a, b

def scatter_dates(peaks, last_n_years=30, source='NorESM2-LM', ax=None, return_ax=False, reg_summary=False, color='#3C5C1B', anomaly=True):
    '''
        Create a scatter plot of the peak dates.
    Args:
        peaks         [DataFrame]  : Pandas.DataFrame object resulting from find_peak_dates()
        last_n_years  [int]        : Number of years to plot for, counting from last. Default is 30.
        source        [str]        : Name of the data source to use for plot title
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,4))
    
    if anomaly:
        ylabel = 'Day of Year (Anomaly)'
        to_plot = peaks - peaks.mean()
    else:
        ylabel = 'Day of Year'
        to_plot = peaks

    ax.scatter(to_plot.time[-last_n_years:], to_plot.values[-last_n_years:], color=color, alpha=0.7, s=3)
    
    _df = pd.DataFrame({'doy_anomaly': to_plot, 'year': to_plot.time})
    
    # Do regression
    if reg_summary:
        a, b, summary = regression(_df[-last_n_years:], 'year', 'doy_anomaly', summary=True)
    else:
        a, b = regression(_df[-last_n_years:], 'year', 'doy_anomaly')
        
    ax.plot(_df['year'][-last_n_years:], a * _df['year'][-last_n_years:] + b, 
            label=f'Trend: {round(a, 3)} d/y', linestyle='--', color=color)
    
    #Rolling mean
    mean5 = to_plot.rolling(time=5, center=True, min_periods=3).mean()
    mean10 = to_plot.rolling(time=10, center=True, min_periods=7).mean()
    ax.plot(_df['year'][-last_n_years:],  mean5.values[-last_n_years:], label=f'5-year mean', color=color, linewidth=1)
    ax.plot(_df['year'][-last_n_years:],  mean10.values[-last_n_years:], label=f'10-year mean', color=color, linewidth=3)
    
    #Variance
    std10 = to_plot.rolling(time=10, center=True, min_periods=7).std()
    
    ax.fill_between(_df['year'][-last_n_years:],  
                    mean10.values[-last_n_years:] - std10.values[-last_n_years:], 
                    mean10.values[-last_n_years:] + std10.values[-last_n_years:], 
                    label=f'Std', color=color, linewidth=1, alpha=0.3)
    #ax.plot(_df['year'][-last_n_years:],  min10.values[-last_n_years:], label=f'5-year mean', color=color, linewidth=1)
    #ax.plot(_df['year'][-last_n_years:],  max10.values[-last_n_years:], label=f'5-year mean', color=color, linewidth=1)
    
    ax.grid()
    ax.set_title(f'Barents sea: Peak phytoplankton blooming ({source})')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Peak chlorophyll date [{ylabel}]')
    ax.legend(loc='upper right')
    if return_ax or reg_summary:
        to_return = []
        if return_ax:
            to_return.append(fig, ax)
        if reg_summary:
            to_return.append(summary)
            return to_return
        
def barentsMap(minlat=70, maxlat=80, minlon=20, maxlon=60, nrows=1, ncols=1):
    '''
        Create a cartopy map instance for a specific region.
    Args:
        minlat     [float]  : Minimum latitude of the region
        maxlat     [float]  : Maximum latitude of the region
        minlon     [float]  : Minimum longitude of the region
        maxlon     [float]  : Maximum longitude of the region
    Returns:
        fig        [Figure] : Matplotlib.pyplot.Figure instance
        map        [Axes]   : Matplotlib.pyplot.Axes instance with given cartopy projection
    '''
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[7,5], projection=ccrs.NorthPolarStereo(central_longitude=40))
    try:
        for ax in axs:
            ax.coastlines(); ax.gridlines(draw_labels=True)
            ax.set_extent([minlon, maxlon, minlat, maxlat], crs=ccrs.PlateCarree())
    except:
        axs.coastlines(); axs.gridlines(draw_labels=True)
        axs.set_extent([minlon, maxlon, minlat, maxlat], crs=ccrs.PlateCarree())
    return fig, axs

def slider_map(_da, start, stop, freq='M', name='ESACCI', model=False, color='YlGn', levels=np.linspace(0, 3E-6, 20).round(7), 
               minlat=70, maxlat=80, minlon=20, maxlon=60):
    '''
        Create a plot with interactive slider to forward time
    Args:
        _ds       [DataSet]   : Xarray.DataArray of variable data
        start     [Datetime]  : Datetime.Datetime() instance of the desired start date for slider
        end       [Datetime]  : Datetime.Datetime() instance of the desired end date for slider
        freq      [str]       : Time resolution of the data, e.g 'M' or 'D'
        name      [str]       : Name of the data source to use for plot title
        model     [bool]      : Whether the data origins from model or observations
        color     [str]       : Name of a Matplotlib colormap to use for plotting
        levels    [list-like] : Intervals to use for colorbar bins
        minlat    [float]     : Minimum latitude of the region
        maxlat    [float]     : Maximum latitude of the region
        minlon    [float]     : Minimum longitude of the region
        maxlon    [float]     : Maximum longitude of the region
    '''
    if freq == 'M':
        timeformat = '%Y-%m'
    elif freq == 'D':
        timeformat = '%Y-%m-%d'
        
    def plot_map(date):
        '''
            Helper function to be called from IPyWidgets.interact to compute timestep
        Args:
            date   [Datetime]  : Datetime.Datetime instance to plot the data at
        '''
        fig, map = barentsMap(minlat=minlat, 
                              maxlat=maxlat, 
                              minlon=minlon, 
                              maxlon=maxlon)
        cmap = plt.get_cmap(color)
        if model:
            try:
                _da.sel(time=date.strftime(timeformat)).plot(ax=map, x='lon', y='lat', transform=ccrs.PlateCarree(), 
                                                           cmap=cmap, robust=True, levels=levels, cbar_kwargs={'location': 'bottom'})
            except:
                _da.sel(time=date.strftime(timeformat)).plot(ax=map, x='lon', y='lat', transform=ccrs.PlateCarree(), 
                                                           cmap=cmap, robust=True, levels=levels, cbar_kwargs={'location': 'bottom'})
            datestr = date.strftime(timeformat) 
        else:
            _da.sel(time=date, method='bfill').plot(ax=map, transform=ccrs.PlateCarree(), cmap=cmap, robust=True, 
                                                    cbar_kwargs={'spacing': 'proportional'})

            datestr = date.strftime(timeformat) 
        map.set_title(f'{name} - {datestr}\n')
    
    dates = pd.date_range(start, stop, freq=freq)
    options = [(date.strftime(timeformat), date) for date in dates]
    index = (0, len(options) - 1)
    date_slider = widgets.SelectionSlider(
                    options=options,
                    orientation='horizontal',
                    layout={'width': '800px'}
                )
    interact(plot_map, date=date_slider)

def consistent_naming(ds):
    """
    Author @Ada Gjermundsen
    The naming convention for coordinates and dimensions are not the same 
    for noresm raw output and cmorized variables. This function rewrites the 
    coords and dims names to be consistent and the functions thus work on all
    Choose the cmor naming convention.

    Parameters
    ----------
    ds : xarray.Dataset 

    Returns
    -------
    ds : xarray.Dataset

    """
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "region" in ds.dims:
        ds = ds.rename(
            {"region": "basin"}
        )  # are we sure that it is the dimension and not the variable which is renamed? Probably both
        # note in BLOM raw, region is both a dimension and a variable. Not sure how xarray handles that
        # in cmorized variables sector is the char variable with the basin names, and basin is the dimension
        # don't do this -> ds = ds.rename({'region':'sector'})
    if "x" in ds.dims:
        ds = ds.rename({"x": "i"})
    if "y" in ds.dims:
        ds = ds.rename({"y": "j"})
    if "ni" in ds.dims:
        ds = ds.rename({"ni": "i"})
    if "nj" in ds.dims:
        ds = ds.rename({"nj": "j"})
    if "nlat" in ds.dims:
        ds = ds.rename({"nlat": "j"})
    if "nlon" in ds.dims:
        ds = ds.rename({"nlon": "i"})
    if "nlat" in ds.coords:
        ds = ds.rename({"nlat": "j"})
    if "nlon" in ds.coords:
        ds = ds.rename({"nlon": "i"})
    if "depth" in ds.dims:
        ds = ds.rename({"depth": "lev"})
    if "nbnd" in ds.dims:
        ds = ds.rename({"nbnd": "bnds"})
    if "nbounds" in ds.dims:
        ds = ds.rename({"nbounds": "bnds"})
    if "bounds" in ds.dims:
        ds = ds.rename({"bounds": "bnds"})
    if "type" in ds.coords:
        ds = ds.drop("type")
    if 'latitude_bnds' in ds.variables:
        ds = ds.rename({'latitude_bnds':'lat_bnds'})
    if 'longitude_bnds' in ds.variables:
        ds = ds.rename({'longitude_bnds':'lon_bnds'})
    if 'nav_lat' in ds.coords:
        ds = ds.rename({'nav_lon':'lon','nav_lat':'lat'})
    if 'bounds_nav_lat' in ds.variables:
        ds = ds.rename({'bounds_nav_lat':'vertices_latitude','bounds_nav_lon':'vertices_longitude'})
    return ds