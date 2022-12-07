import xarray as xr
import intake
import cftime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import s3fs
import glob
#%matplotlib inline

#......use 'volcello' for DMS and clos data......##

def open_file(model,var):
    
    'Returns a list of files of a particulal variable of a model run from the storage '
    
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", 
                       secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0", client_kwargs=dict(endpoint_url="https://rgw.met.no"))

    if model=='NorESM2-LM':
        if var == 'chlos':
            file_dir ='s3://escience2022/Ada/monthly/chlos_Omon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
        elif var=='dmsos':
            file_dir ='s3://escience2022/Ada/monthly/dmsos_Omon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
        elif var=='emidms':
            file_dir ='s3://escience2022/Ada/monthly/emidms_AERmon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
        elif var == 'siconc':
            file_dir='s3://escience2022/Ada/monthly/siconc_SImon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
        elif var == 'tos':
            file_dir='s3://escience2022/Ada/monthly/tos_Omon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
        
    if (model=='CNRM-ESM2-1'):
        
        if var == 'chlos':
            file_dir ='s3://escience2022/Ada/monthly/chlos_Omon_CNRM-ESM2-1_historical_r2i1p1f2_gn*.nc'
        elif var=='dmsos':
            file_dir ='s3://escience2022/Ada/monthly/dmsos_Omon_CNRM-ESM2-1_historical_r2i1p1f2_gn*.nc'
        elif var=='emidms':
            file_dir ='s3://escience2022/Ada/monthly/emidms_AERmon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
        elif var == 'siconc':
            file_dir='s3://escience2022/Ada/monthly/siconc_SImon_CNRM-ESM2-1_historical_r2i1p1f2_gn*.nc'
        elif var == 'tos':
            file_dir='s3://escience2022/Ada/monthly/tos_Omon_CNRM-ESM2-1_historical_r2i1p1f2_gn*.nc'
        
    
    if (model=='CESM2'):
        
        if var == 'chlos':
            file_dir ='s3://escience2022/Ada/monthly/chlos_Omon_CESM2_historical_r4i1p1f1_gn*.nc'
        elif var=='dmsos':
            file_dir ='s3://escience2022/Ada/monthly/dmsos_Omon_NorESM2-LM_historical_r1i1p1f1_gn*.nc'
        elif var=='emidms':
            file_dir ='s3://escience2022/Ada/monthly/emidms_AERmon_NorESM2-LM_historical_r1i1p1f1_gn*.nc'
        elif var == 'siconc':
            file_dir='s3://escience2022/Ada/monthly/siconc_SImon_CESM2_historical_r4i1p1f1_gn*.nc'
        elif var == 'tos':
            file_dir='s3://escience2022/Ada/monthly/tos_Omon_CESM2_historical_r4i1p1f1_gn*.nc'
        
    remote_files = s3.glob(file_dir)
    fileset = [s3.open(file) for file in remote_files]
    return fileset

def get_areacello(model,region):
    
    "Return the area of the cells of given region"
    
    min_lat= region[0]
    max_lat= region[1]
    min_lon= region[2]
    max_lon= region[3] 
    
    if (model=='NorESM2-LM'):
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=['areacello'], member_id=['r1i1p1f1'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    
        areacello = ds_dict[list(ds_dict.keys())[0]]
        areacello = areacello.squeeze()                     
        
        BSarea = areacello.areacello.where((areacello.latitude>=min_lat) & (areacello.latitude<=max_lat) 
                                       & (areacello.longitude <= max_lon)  & (areacello.longitude >= min_lon))    
    if (model=='CNRM-ESM2-1'):
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=['areacello'], member_id=['r2i1p1f2'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    
        areacello = ds_dict[list(ds_dict.keys())[0]]
        areacello = areacello.squeeze()
        
        BSarea = areacello.areacello.where((areacello.lat>=min_lat) & (areacello.lat<=max_lat) 
                                       & (areacello.lon <= max_lon)  & (areacello.lon >= min_lon))
    if (model=='CESM2'):
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=['areacello'], member_id=['r4i1p1f1'])
        ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    
        areacello = ds_dict[list(ds_dict.keys())[0]]
        areacello = areacello.squeeze()
        
        BSarea = areacello.areacello.where((areacello.lat>=min_lat) & (areacello.lat<=max_lat) 
                                       & (areacello.lon <= max_lon)  & (areacello.lon >= min_lon))
    return BSarea

def get_polar_region(ds,model):  
    
    #......................get an xarray only for the polar region.......................#
    
    cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(cat_url)
    
    if (model=='NorESM2-LM'):
        cat = col.search(source_id=['NorESM2-LM'], activity_id = ['CMIP'], experiment_id=['piControl'], 
                     table_id=['Ofx'], variable_id=['areacello'], member_id=['r1i1p1f1'])
    if (model=='CNRM-ESM2-1'):
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=[area], member_id=['r2i1p1f2'])
        
    if (model=='CESM2'):
        cat = col.search(source_id=[model], activity_id = ['CMIP'], experiment_id=['piControl'], 
                         table_id=['Ofx'], variable_id=[area], member_id=['r4i1p1f1'])
    
    ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    areacello = ds_dict[list(ds_dict.keys())[0]]
    areacello = areacello.squeeze()
    areacello = areacello.where(areacello.latitude>60, drop = True)
    da=ds.sel(i=areacello.i).sel(j=areacello.j)
    
    return da
    

def regional_average(model,var):
    
    #........................calculate the regional average for each year.................................#
    
    min_lat=60      #............................................#
    max_lat=90      #                                           .#
    min_lon=0       #        Polar Region                       .#
    max_lon=360     #............................................# 
    
    region=[min_lat,max_lat,min_lon,max_lon]  #region defined                                          
    
    cell_area=get_areacello(model,region)  #get cell area
    
    fileset=open_file(model,var)                                               #get a list of files to open
    da = xr.open_mfdataset(fileset, combine='by_coords')
    
    if var!='siconc':
        ds= get_polar_region(da,model)
    else:
        ds=da
    
    dss=weighted_yearly_mean(ds, var)
    
    if (model=='NorESM2-LM'):
        BSsst = dss.where((dss.latitude>=min_lat) & (dss.latitude<=max_lat) & 
                      (dss.longitude <= max_lon)  & (dss.longitude >=min_lon))
    
    if (model=='CNRM-ESM2-1'):
        BSsst = dss.where((dss.lat>=min_lat) & (dss.lat<=max_lat) & 
                      (dss.lon <= max_lon)  & (dss.lon>=min_lon))
        
    if (model=='CESM2'):
        BSsst = dss.where((dss.lat>=min_lat) & (dss.lat<=max_lat) & 
                      (dss.lon <= max_lon)  & (dss.lon >=min_lon))
    
    if var =='chlos':
       BSsst = (cell_area*BSsst).sum(dim=('i','j'))/(cell_area).sum(dim=('i','j'))  #check if it is correct?
    if var =='dmsos':
       BSsst = (cell_area*BSsst).sum(dim=('i','j'))/(cell_area).sum(dim=('i','j'))  #check if it is correct?
    else:
        BSsst = (cell_area*BSsst).sum(dim=('i','j'))/(cell_area).sum(dim=('i','j'))
    return BSsst

def weighted_yearly_mean(ds, var):
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
    ones = xr.where(cond, 0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).groupby("time.year").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones*wgts).groupby("time.year").sum(dim="time")

    # Return the weighted average
    return obs_sum / ones_out


def weighted_seasonal_mean(ds,var):  #to calculate mean of a particular season. The output is the mean of each season separately over a period of time.
    
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.season") / month_length.groupby("time.season").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.season").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).groupby("time.season").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones*wgts).groupby("time.season").sum(dim="time")

    # Return the weighted average
    return (obs_sum/ ones_out).to_dataset(name = var)

def weighted_seasonal_mean(ds,var):  
    
    #.....to calculate mean of a particular season. The output is the mean of each season separately over a period of time....#
    
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.month") / month_length.groupby("time.month").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.month").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).groupby("time.month").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones*wgts).groupby("time.month").sum(dim="time")

    # Return the weighted average
    return (obs_sum/ ones_out).to_dataset(name = var)

def monthly_average(ds,var,region,model):
    
    min_lat=region[0]
    max_lat=region[1]
    min_lon=region[2]
    max_lon=region[3]
    
    ds=weighted_monthly_mean(ds,var)
    ds = ds.where((ds.latitude>=min_lat) & (ds.latitude<=max_lat) & 
                      (ds.longitude <= max_lon)  & (ds.longitude >=min_lon))
                                         
    cell_area=get_areacello(model,region)  #get cell area
    monthly_mean = (cell_area*10*ds).sum(dim=('i','j'))/(cell_area*10).sum(dim=('i','j'))
    
    return monthly_mean

def weighted_monthly_mean(ds,var):  
    
    #....To calculate mean of a particular season. The output is the mean of each season separately over a period of time.....#
    
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.month") / month_length.groupby("time.month").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.month").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).groupby("time.month").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones*wgts).groupby("time.month").sum(dim="time")

    # Return the weighted average
    return (obs_sum/ ones_out).to_dataset(name = var)

def seasonal_avg_timeseries(model,var):   #if I want to plot trends of a particular season
    """Calculates timeseries over seasonal averages from timeseries of monthly means
    The weighted average considers that each month has a different number of days.
    Using 'QS-DEC' frequency will split the data into consecutive three-month periods, 
    anchored at December 1st. 
    I.e. the first value will contain only the avg value over January and February 
    and the last value only the December monthly averaged value
    
    Parameters
    ----------
    ds : xarray.DataArray i.e.  ds[var]
        
    Returns
    -------
    ds_out: xarray.DataSet with 4 timeseries (one for each season DJF, MAM, JJA, SON)
            note that if you want to include the output in an other dataset, e.g. dr,
            you should use xr.merge(), e.g.
            dr = xr.merge([dr, seasonal_avg_timeseries(dr[var], var)])
    """
    
    fileset=open_file(model,var)
    da = xr.open_mfdataset(fileset, combine='by_coords')
    if var!= 'siconc':
        ds=get_polar_region(da,model)
    else:
        ds=da
   
    month_length = ds.time.dt.days_in_month
    sesavg = (ds * month_length).resample(time="QS-DEC").sum() / month_length.where(ds.notnull()).resample(time="QS-DEC").sum()
    
    djf=sesavg.isel(time=slice(0,None,4))
    mam=sesavg.isel(time=slice(1,None,4))
    jja=sesavg.isel(time=slice(2,None,4))
    son=sesavg.isel(time=slice(3,None,4))
    
    return [djf,mam,jja,son]

def slice_assign(ds_ls):
    
    "get the dataset for the last 65 years and add latitude and longitude to the datarray"
    
    ds=ds_ls.sel(time=slice("1950-01-01", "2014-10-31"))
    dss=ds.assign_coords(time=np.arange(1950,1950+len(ds.time.values)))
    da=dss.polyfit('time',deg=1)

    da1=da.assign_coords(latitude=ds.latitude)
    da2=da1.assign_coords(longitude=ds.longitude)
    return da2
    
def anomaly_seasonal(model,var):
    
    ##........
    
    fileset = open_file(model,var)

    ds = xr.open_mfdataset(fileset, combine='by_coords')
    
    now=ds.sel(time=slice("1980-01-01", "2014-10-31"))
    
    ref_year=ds.sel(time=slice("1950-01-01", "1979-10-31"))
    
    weighted_now= weighted_seasonal_mean(now,var)
    weighted_ref= weighted_seasonal_mean(ref_year,var)
    
    return [weighted_now,weighted_ref]


def anomaly (var):
    
    ## Put the variable name as stored in the NorESM data. This function will output the anomaly and anomaly/climatology in a list.
    #anaomaly is calculated as follows:
         #climatology is calculated usng data from 1950 to 1979
         #Present day trend is calculated for data from 1980 to 2014
    
    
    fileset=open_file(var)
    da = xr.open_mfdataset(fileset, combine='by_coords')
    
    if var!='siconc':
        ds= get_polar_region(da)
    else:
        ds=da
   
    weight= weighted_temporal_mean(ds,var)
    
    aa=weight#.groupby("time.year").sum(dim='time')

    now1=aa.isel(year = slice(30,None))  #remove first 30 years
    now=now1.mean('year')

    clim1= aa.isel(year = slice(None,30))  #remove last 30 years
    clim=clim1.mean('year')

    fractional_anm=(now-clim)/clim
    anm= (now-clim)
    
    anomaly=[anm,fractional_anm]
    return anomaly

def check_data(n,var):
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", 
                       secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0", client_kwargs=dict(endpoint_url="https://rgw.met.no"))


    if var == 'chlos':
        file_dir ='s3://escience2022/Ada/monthly/chlos_Omon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
    if var=='dmsos':
        file_dir ='s3://escience2022/Ada/monthly/dmsos_Omon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
    if var=='emidms':
        file_dir ='s3://escience2022/Ada/monthly/emidms_AERmon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
    if var == 'siconc':
        file_dir='s3://escience2022/Ada/monthly/siconc_SImon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
    if var == 'tos':
        file_dir='s3://escience2022/Ada/monthly/tos_Omon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc'
        
    remote_files = s3.glob(file_dir)
    fileset = [s3.open(file) for file in remote_files[n:]]
    da = xr.open_mfdataset(fileset, combine='by_coords')
    
    return da