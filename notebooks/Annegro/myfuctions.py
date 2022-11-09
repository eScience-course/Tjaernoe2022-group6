import cartopy.crs as ccrs


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
    