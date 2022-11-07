import xarray as xr
xr.set_options(display_style='html')
import intake
import cftime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import s3fs
import glob
#%matplotlib inline

#......use 'volcello' for DMS and clos data......##


def get_areacello(model,min_lat,max_lat,min_lon,max_lon,area):
    
    if (model=='NorESM2-LM'):
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=[area], member_id=['r1i1p1f1'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    
        areacello = ds_dict[list(ds_dict.keys())[0]]
        areacello = areacello.squeeze()
        
        BSarea = areacello.get(area).where((areacello.latitude>=min_lat) & (areacello.latitude<=max_lat) 
                                       & (areacello.longitude <= max_lon)  & (areacello.longitude >= min_lon))    
    if (model=='CNRM-ESM2-1'):
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=[area], member_id=['r1i1p1f2'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    
        areacello = ds_dict[list(ds_dict.keys())[0]]
        areacello = areacello.squeeze()
        
        BSarea = areacello.get(area).where((areacello.lat>=min_lat) & (areacello.lat<=max_lat) 
                                       & (areacello.lon <= max_lon)  & (areacello.lon >= min_lon))
    if (model=='CESM2'):
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=[area], member_id=['r1i1p1f1'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    
        areacello = ds_dict[list(ds_dict.keys())[0]]
        areacello = areacello.squeeze()
        
        BSarea = areacello.get(area).where((areacello.lat>=min_lat) & (areacello.lat<=max_lat) 
                                       & (areacello.lon <= max_lon)  & (areacello.lon >= min_lon))
    return BSarea


def regional_average(inp):
                                                 
    files_dir= inp[0]
    model= inp[1]
    min_lat= inp[2]
    max_lat= inp[3]
    min_lon= inp[4]
    max_lon= inp[5]   
    var= inp[6]
    cel_type=  inp[7]                                                 
    area=cel_type
    
    cell_area=get_areacello(model,min_lat,max_lat,min_lon,max_lon,area)

    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", 
                       secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0", client_kwargs=dict(endpoint_url="https://rgw.met.no"))


    remote_files = 's3:/'+ files_dir
    remote_files = s3.glob(remote_files)
    from_1950=remote_files[10:]
    fileset = [s3.open(file) for file in from_1950]
    ds = xr.open_mfdataset(fileset, combine='by_coords')
    
    av=weighted_temporal_mean(ds, var)
    dss=av.groupby("time.year").sum(dim='time')
    
    BSsst = dss.where((dss.latitude>=min_lat) & (dss.latitude<=max_lat) & 
                      (dss.longitude <= max_lon)  & (dss.longitude >=min_lon))
    
    if var =='chlos':
       BSsst = (cell_area*10*BSsst).sum(dim=('i','j'))/(cell_area*10).sum(dim=('i','j'))
    else:
        BSsst = (cell_area*BSsst).sum(dim=('i','j'))/(cell_area).sum(dim=('i','j'))
    return BSsst


def weighted_temporal_mean(ds, var):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    # Return the weighted average
    return obs_sum / ones_out