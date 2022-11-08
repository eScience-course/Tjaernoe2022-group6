import xarray as xr, pandas as pd, numpy as np, math, time   # For data handling
import s3fs, intake, cftime                                  # For reading bucket data
import matplotlib.pyplot as plt, cartopy.crs as ccrs         # For plotting
import statsmodels.formula.api as sm                         # For regression
from joblib import Parallel, delayed                         # For parallizing
from datetime import datetime                                # For time computations
from ipywidgets import interact, interactive, fixed, widgets # For interactive plotting

# s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
#                        client_kwargs=dict(endpoint_url="https://rgw.met.no"))

# def read_file(file):
#     return s3.open(file)

# def create_xr(file):
#     return xr.open_dataset(file)

def get_bucket_data(variable, time_res='monthly', model='NorESM2-LM', experiment='hist', 
                    member_id='*', period='*', chunks=None, noChunks=True, last_n_files=0, parallel=False):
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
    Returns:
        dS           [DataSet] :  xarray.DataSet of the collected data
    '''
    
    if chunks is None and not noChunks:
        chunks = {'time': chunks}#math.ceil((365 * (5 + 10 * (17 - 1)) + 1) / 16)}
        
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
                           client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    base_path = 's3://escience2022/Ada/'
    if time_res == 'monthly':
        name = 'Omon'
    else:
        name = 'Oday'
    if member_id == period:
        full_path = base_path + f'{time_res}/{variable}_{name}_{model}_{experiment}_*.nc'
    else:
        full_path = base_path + f'{time_res}/{variable}_{name}_{model}_{experiment}_{member_id}_{period}.nc'
    
    print('Request: ' + full_path)
    
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
        areacello = xr.open_dataset(f'areacello_{model}.nc').areacello
        print('Found local areacello NetCDF')
        return areacello
    except:
        print('Areacello not stored locally. Getting from cloud...')
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], table_id=['Ofx'], 
                         variable_id=['areacello'], member_id=['r1i1p1f1'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
        area = ds_dict[list(ds_dict.keys())[0]]
        areacello = area.areacello
        areacello.to_netcdf(f'areacello_{model}.nc')
        return areacello

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
        
    clip = _ds.where((_ds[name_lat] > minlat) & (_ds[name_lat] < maxlat) & (_ds[name_lon] > minlon) & (_ds[name_lon] < maxlon), drop=True)
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
    area = clip_to_region(get_areacello(model), *clip_coords)
    mean = (data*area).sum(dim=('i','j'))/area.sum(dim=('i','j'))
    #return data.weighted(area.fillna(0)).mean()
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


def find_peak_dates2(_ds):
    _ds = _ds.groupby('time.year')
    _ds_dates = [_ds[year].idxmax(dim='time').values[0] for year in _ds.groups.keys()]
    df = xr.DataArray(_ds_dates).to_dataframe('date')
    df['dayofyear'] = df['date'].dt.dayofyear
    return df

def find_peak_dates(_da):
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
    print(reg.summary)
    b, a = reg.params
    if summary == True:
        summary = reg.summary()
        return a, b, summary
    else:
        return a, b

def scatter_dates(peaks, last_n_years=30, source='NorESM2-LM'):
    '''
        Create a scatter plot of the peak dates.
    Args:
        peaks         [DataFrame]  : Pandas.DataFrame object resulting from find_peak_dates()
        last_n_years  [int]        : Number of years to plot for, counting from last. Default is 30.
        source        [str]        : Name of the data source to use for plot title
    '''
    fig, ax = plt.subplots(figsize=(12,4))
    peaks['doy_anomaly'] = peaks['dayofyear'] - peaks['dayofyear'].mean()

    peaks[-last_n_years:].plot.scatter(x='year', y='doy_anomaly', color='g', ax=ax)
    a, b = regression(peaks[-last_n_years:], 'year', 'doy_anomaly')
    ax.plot(peaks['year'][-last_n_years:], a * peaks['year'][-last_n_years:] + b, label=f'slope = {round(a, 3)} d/y')
    ax.grid()
    fig.suptitle(f'Peak phytoplankton blooming dates ({source})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Day of year anomaly')
    ax.legend(loc='upper right')
    
def barentsMap(minlat=70, maxlat=80, minlon=20, maxlon=60):
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
    fig = plt.figure(1, figsize=[7,5])
    map = plt.subplot(projection=ccrs.NorthPolarStereo(central_longitude=40))
    map.coastlines(); map.gridlines(draw_labels=True)
    map.set_extent([minlon, maxlon, minlat, maxlat], crs=ccrs.PlateCarree())
    return fig, map

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
        fig, map = barentsMap(minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon)
        cmap = plt.get_cmap(color)
        if model:
            _da.sel(time=date.strftime(timeformat)).plot(ax=map, x='longitude', y='latitude', transform=ccrs.PlateCarree(), 
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