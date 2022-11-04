import xarray as xr
import s3fs, intake, cftime, math, time
from joblib import Parallel, delayed
import numba

# s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
#                        client_kwargs=dict(endpoint_url="https://rgw.met.no"))

# def read_file(file):
#     return s3.open(file)

# def create_xr(file):
#     return xr.open_dataset(file)

def get_bucket_data(variable, time_res='monthly', thing='Omon', model='NorESM2-LM', experiment='hist', 
                    member_id='*', period='*', last_n_files=0):
    
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
                           client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    base_path = f's3://escience2022/Ada/'
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
                                chunks={'time':math.ceil((365 * (5 + 10 * (last_n_files - 1)) + 1) / 16)})
    else:
        return xr.open_mfdataset(fileset, concat_dim='time', combine='nested', 
                                chunks={'time':math.ceil((365 * (5 + 10 * (17 - 1)) + 1) / 16)})
        
    
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

def clip_to_region(_ds, mini=340, maxi=380, minj=110, maxj=145):
    return _ds.isel(j=slice(mini, maxi), i=slice(minj, maxj))

def clip_to_months(_ds, start_month, stop_month):
    return _ds.where(_ds.time.dt.month.isin([i for i in range(start_month, stop_month + 1)]))
    
def regional_average(data, model='NorESM2-LM', clip_coordinates=[300, 380, 75, 160]):
    area = clip_to_region(get_areacello(model), *clip_coordinates)
    #return data.weighted(area.fillna(0)).mean()
    return (data*area).sum(dim=('i','j'))/area.sum(dim=('i','j'))

def time_anomaly(_ds, first_start, first_stop, last_start, last_stop):
    return _ds.isel(time=slice(last_start, last_stop)).mean(dim='time') - _ds.isel(time=slice(first_start, first_stop)).mean(dim='time')


def find_peak_date(_ds):
    _ds_yearly = _ds.group_by('time.year')