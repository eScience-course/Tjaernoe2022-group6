import xarray as xr
import cftime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import s3fs
import glob
import modules as md

min_lat=65
max_lat=90
min_lon=16
max_lon=68

model='NorESM2-LM'
area=md.get_areacello(model,min_lat,max_lat,min_lon,max_lon)

print(area)