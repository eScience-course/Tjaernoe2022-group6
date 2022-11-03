import xarray as xr
import s3fs, intake, cftime

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
    return xr.open_mfdataset(fileset, combine='by_coords')
    
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

def clip_to_region(_ds, mini=300, maxi=380, minj=75, maxj=160):
    return _ds.isel(j=slice(mini, maxi), i=slice(minj, maxj))
    
def regional_average(data, model='NorESM2-LM', clip_coordinates=[300, 380, 75, 160]):
    area = clip_to_region(get_areacello(model), *clip_coordinates)
    return (data*area).sum(dim=('i','j'))/area.sum(dim=('i','j'))

def time_anomaly(_ds, first_start, first_stop, last_start, last_stop):
    return _ds.isel(time=slice(last_start, last_stop)).mean(dim='time') - _ds.isel(time=slice(first_start, first_stop)).mean(dim='time')