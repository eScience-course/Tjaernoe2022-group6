import xarray as xr
import s3fs

def get_data_from_bucket():
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
                           client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    base_path = 's3://escience2022/Ada/'
    
    full_path = base_path + 'monthly/chlos_Omon_CESM2_esm-ssp585_r1i1p1f1_gn_*.nc'
    
    fileset = [s3.open(file) for file in full_path]
    return xr.open_mfdataset(fileset, combine='by_coords')
    
