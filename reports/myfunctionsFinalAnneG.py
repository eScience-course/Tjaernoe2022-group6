import cartopy.crs as ccrs
from cftime import DatetimeNoLeap
import xarray as xr
import s3fs
import numpy as np
import intake


def plotmap(ax,dr,season, levels):
    cf=ax.contourf(dr.lon,
              dr.lat,
              dr.sel(season=season),
              levels=levels,
              transform=ccrs.PlateCarree(),
                 
                 
              )
    ax.set_extent([-180, 180, 90, 50], ccrs.PlateCarree())
    ax.set_title(season)
    ax.gridlines(draw_labels=True) 
    ax.coastlines()
    return cf 



def read_omip_bucketdata(modelname, realm, var):
    print(var)
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0", client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    remote_files = 's3://escience2022/Ada/monthly/%s_%s_%s_omip1_r1i1p1f1_g*_19*.nc'%(var, realm, modelname)
    remote_files = s3.glob(remote_files)
    remote_files2 = 's3://escience2022/Ada/monthly/%s_%s_%s_omip1_r1i1p1f1_g*_200*.nc'%(var, realm, modelname)
    remote_files = remote_files + s3.glob(remote_files2)
    #print(remote_files)
    fileset = [s3.open(file) for file in remote_files]
    ds = xr.open_mfdataset(fileset, combine='by_coords')
    #print(ds)
    # only select the last 62 years
    ds = ds.sel(time = slice("1948-01-01","2009-12-31"))
    # rewrite time variable
    #print(ds)
    #months = ds.time_bnds.isel(bnds=0).dt.month.values
    #years = np.arange(1948, 2010)
    #years = np.repeat(years,12)
    #dates = [DatetimeNoLeap(year, month, 15) for year, month in zip(years, months)]
    #ds = ds.assign_coords(time=dates)
    return ds

def make_monthlymean(ds):
    return ds.groupby('time.month').mean('time')

def make_seasonmean(ds):
    return ds.groupby('time.season').mean('time')

def getareacello(modelname):
    cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col =intake.open_esm_datastore(cat_url)
    cat = col.search(source_id=['%s'%modelname], 
    activity_id=['CMIP'],experiment_id=['piControl'], 
    table_id=['Ofx'], variable_id=['areacello'], member_id=['r1i1p1f1'])
    ds_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})

    areacello = ds_dict[list(ds_dict.keys())[0]]
    areacello = areacello.squeeze()
    
    return areacello 

def regional_avg(ds, areacello, var, lat_min, lat_max, lon_min, lon_max):
    area = select_region(areacello, lat_min, lat_max, lon_min, lon_max)
    ds = select_region(ds, lat_min, lat_max, lon_min, lon_max)
    return (area.areacello*ds[var]).sum(dim=('i','j'))/area.areacello.sum(dim=('i','j'))
    
def select_region(da, lat_min, lat_max, lon_min, lon_max):
        return da.where((da.latitude>= lat_min) & (da.latitude<=lat_max) & (da.longitude >= lon_min)  & (da.longitude <= lon_max))
                     
                                                 
                                                 
def plot_variables(axs, var, title, ds_ice, ds_ocn, areacello, coordinates, i, colors):
    if var == 'siconc':
        ds = ds_ice
    else:
        ds = ds_ocn
        
    dsStart=make_monthlymean(ds[var].sel(time=slice('1948-01-01','1967-12-31'))).to_dataset(name = var)
    dsEnd=make_monthlymean(ds[var].sel(time=slice('1988-01-01', '2007-12-31'))).to_dataset(name = var)
    dsStartBB = regional_avg(dsStart, areacello, var,  coordinates[0], coordinates[1], coordinates[2], coordinates[3])
    dsEndBB = regional_avg(dsEnd, areacello, var, coordinates[0], coordinates[1], coordinates[2], coordinates[3])
    
    ax = axs[i]
         
    ax.plot(dsStartBB.month, dsStartBB, color=colors[0], label='1948-1968')
    ax.plot(dsEndBB.month, dsEndBB, color=colors[1], label='1988-2008' )
    
    ax.set_ylabel(ds[var].units)
    ax.set_xlabel('Month')        
    ax.set_xticks(np.arange(1,13,2)) 
    ax.legend(loc='upper left' )    
    
    
    ax.set_title(title)
    


    