import cartopy.crs as ccrs
from cftime import DatetimeNoLeap
import xarray as xr
import s3fs
import numpy as np

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



def read_omip_bucketdata(modelname, var):
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0", client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    remote_files = 's3://escience2022/Ada/monthly/%s_Omon_%s_omip1_r1i1p1f1_gn_20*.nc'%(var, modelname)
    remote_files = s3.glob(remote_files)
    fileset = [s3.open(file) for file in remote_files]
    ds = xr.open_mfdataset(fileset, combine='by_coords')
    # only select the last 62 years
    ds = ds.sel(time = slice("2010-01-01",None))
    # rewrite time variable
    months = ds.time_bnds.isel(bnds=0).dt.month.values
    years = np.arange(1948, 2010)
    years = np.repeat(years,12)
    dates = [DatetimeNoLeap(year, month, 15) for year, month in zip(years, months)]
    ds = ds.assign_coords(time=dates)
    return ds

def make_monthlymean(ds):
    return ds.groupby('time.month').mean('time')