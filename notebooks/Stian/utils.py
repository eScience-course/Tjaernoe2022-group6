import xarray as xr
import s3fs, intake, cftime

def get_monthly_bucket_data(variable, thing='Omon', model='NorESM2-LM', experiment='hist', member_id='*', period='*'):
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
                           client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    base_path = 's3://escience2022/Ada/monthly/'
    if member_id == period:
        full_path = base_path + variable + '_' + thing + '_' + model + '_' + experiment + '_' + '*' + '.nc'
    else:
        full_path = base_path + variable + '_' + thing + '_' + model + '_' + experiment + '_' + member_id + '_' + period + '.nc'
    remote_files = s3.glob(full_path)
    print(full_path)
    fileset = [s3.open(file) for file in remote_files]
    return xr.open_mfdataset(fileset, combine='by_coords')
    
def get_areacello(model='NorESM2-LM'):
    cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(cat_url)
    cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], table_id=['Ofx'], 
                     variable_id=['areacello'], member_id=['r1i1p1f1'])
    ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    areacello = ds_dict[list(ds_dict.keys())[0]]
    return areacello

def clip_to_region(_ds, mini=300, maxi=380, minj=75, maxj=160):
    return _ds.sel(j=slice(300, 380), i=slice(75, 160))
    
def regional_average(data, area):
    return (data*area).sum(dim=('i','j'))/area.sum(dim=('i','j'))
    