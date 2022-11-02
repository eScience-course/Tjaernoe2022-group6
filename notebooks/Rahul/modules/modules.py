import xarray as xr
xr.set_options(display_style='html')
import intake
import cftime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import s3fs
import glob
%matplotlib inline



def get_areacello(model,min_lat,max_lat,min_lon,max_lon):
    
    if (model=='NorESM2-LM'):
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=['areacello'], member_id=['r1i1p1f1'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    
        areacello = ds_dict[list(ds_dict.keys())[0]]
        areacello = areacello.squeeze()
        
        BSarea = areacello.areacello.where((areacello.latitude>=65.9) & (areacello.latitude<=81.9) 
                                       & (areacello.longitude <= 68.6)  & (areacello.longitude >= 16.6))    
    if (model=='CNRM-ESM2-1'):
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=['areacello'], member_id=['r1i1p1f2'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    
        areacello = ds_dict[list(ds_dict.keys())[0]]
        areacello = areacello.squeeze()
        
        BSarea = areacello.areacello.where((areacello.lat>=65.9) & (areacello.lat<=81.9) 
                                       & (areacello.lon <= 68.6)  & (areacello.lon >= 16.6))
    if (model=='CESM2'):
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=['areacello'], member_id=['r1i1p1f1'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    
        areacello = ds_dict[list(ds_dict.keys())[0]]
        areacello = areacello.squeeze()
        
        BSarea = areacello.areacello.where((areacello.lat>=65.9) & (areacello.lat<=81.9) 
                                       & (areacello.lon <= 68.6)  & (areacello.lon >= 16.6))
    return BSarea

