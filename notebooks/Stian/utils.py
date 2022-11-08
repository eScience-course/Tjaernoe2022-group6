import xarray as xr, pandas as pd
import s3fs, intake, cftime, math, time
from joblib import Parallel, delayed
import statsmodels.formula.api as sm
import numba
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
from ipywidgets import interact, interactive, fixed, interact_manual, widgets
import numpy as np

# s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
#                        client_kwargs=dict(endpoint_url="https://rgw.met.no"))

# def read_file(file):
#     return s3.open(file)

# def create_xr(file):
#     return xr.open_dataset(file)

def get_bucket_data(variable, time_res='monthly', thing='Omon', model='NorESM2-LM', experiment='hist', 
                    chunks = None, noChunks = True, 
                    member_id='*', period='*', last_n_files=0, parallel=False):
    
    if chunks is None and not noChunks:
        chunks = {'time':math.ceil((365 * (5 + 10 * (17 - 1)) + 1) / 16)}
        
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
                           client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    base_path = 's3://escience2022/Ada/'
    if member_id == period:
        full_path = base_path + f'{time_res}/{variable}_{thing}_{model}_{experiment}_*.nc'
    else:
        full_path = base_path + f'{time_res}/{variable}_{thing}_{model}_{experiment}_{member_id}_{period}.nc'
    
    print('Request: ' + full_path)
    
    remote_files = s3.glob(full_path)
    
    fileset = [s3.open(file) for file in remote_files[-last_n_files:]]
    
    # fileset = Parallel(n_jobs=2)(delayed(read_file)(file) for file in remote_files[-last_n_files:])

    #xrs = Parallel(n_jobs=2)(delayed(create_xr)(file) for file in fileset)
    #data = xr.concat(xrs, dim='time')
    
    if last_n_files != 0:
        return xr.open_mfdataset(fileset, concat_dim='time', combine='nested', 
                                chunks=chunks, 
                                 parallel=parallel)
    else:
        return xr.open_mfdataset(fileset, concat_dim='time', combine='nested', 
                                chunks=chunks, 
                                 parallel=parallel
                                )
        
    
def get_areacello(model='NorESM2-LM'):
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

def clip_to_region2(_ds, minj=340, maxj=380, mini=110, maxi=145):
    return _ds.isel(j=slice(minj, maxj), i=slice(mini, maxi))

def shift_longitude(_ds):
    dset = _ds.copy()
    try:
        dset.coords['longitude'] = (dset['longitude'] + 180) % 360 - 180
    except:
        dset.coords['lon'] = (dset['lon'] + 180) % 360 - 180
    return dset

def clip_to_region(_ds, minlon=20, maxlon=60, minlat=70, maxlat=90):
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
        
    _ds = _ds.where((_ds[name_lat] > minlat) & (_ds[name_lat] < maxlat) & (_ds[name_lon] > minlon) & (_ds[name_lon] < maxlon), drop=True)
    return _ds
    
def clip_to_months(_ds, start_month, stop_month):
    return _ds.where(_ds.time.dt.month.isin([i for i in range(start_month, stop_month + 1)]))
    
def regional_average(data, model='NorESM2-LM', clip_coordinates=[20, 60, 70, 90]):
    area = clip_to_region(get_areacello(model), *clip_coordinates)
    #return data.weighted(area.fillna(0)).mean()
    return (data*area).sum(dim=('i','j'))/area.sum(dim=('i','j'))

def time_anomaly(_ds, first_start, first_stop, last_start, last_stop):
    return _ds.isel(time=slice(last_start, last_stop)).mean(dim='time') - _ds.isel(time=slice(first_start, first_stop)).mean(dim='time')


def find_peak_dates2(_ds):
    _ds = _ds.groupby('time.year')
    _ds_dates = [_ds[year].idxmax(dim='time').values[0] for year in _ds.groups.keys()]
    df = xr.DataArray(_ds_dates).to_dataframe('date')
    df['dayofyear'] = df['date'].dt.dayofyear
    return df

def find_peak_dates(_da):
    _dat = _da.copy()
    _dat['time']  = _dat.time.to_dataframe().index.to_datetimeindex()
    _dat = _dat.groupby('time.year')
    _ds_dates = [_dat[year].idxmax(dim='time').values[0] for year in _dat.groups.keys()]
    df = xr.DataArray(_ds_dates).to_dataframe('date')
    df['dayofyear'] = pd.to_datetime(df['date']).dt.dayofyear
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df

def regression(data, x, y):
    reg = sm.ols(formula=f'{y} ~ {x}', data=data).fit()
    print(reg.summary)
    b, a = reg.params
    return a, b

def scatter_dates(peaks, last_n_years=30, source='NorESM2-LM'):
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
    fig = plt.figure(1, figsize=[7,5])
    map = plt.subplot(projection=ccrs.NorthPolarStereo(central_longitude=40))
    map.coastlines(); map.gridlines(draw_labels=True)
    map.set_extent([minlon, maxlon, minlat, maxlat], crs=ccrs.PlateCarree())
    return fig, map

def slider_map(_da, start, stop, freq='M', name='ESACCI', model=False, color='YlGn', levels=np.linspace(0, 3E-6, 20).round(7), 
               minlat=70, maxlat=80, minlon=20, maxlon=60):
    if freq == 'M':
        timeformat = '%Y-%m'
    elif freq == 'D':
        timeformat = '%Y-%m-%d'
    def plot_map(date):
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