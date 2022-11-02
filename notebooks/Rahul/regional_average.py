import xarray as xr
import cftime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import s3fs
import glob



def regional_average_barent(files_dir):
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD",
                           secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    remote_files = 's3://'+files_dir
    remote_files_ls = s3.glob(remote_files)
    print(len(remote_files_ls))  
    
    
    
    