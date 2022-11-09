def plotmap(): 
    ax1.contourf(dr_out.lon,
              dr_out.lat,
              dr_out.isel(season=),
              levels=np.arange(0, 101, 5),
              transform=ccrs.PlateCarree(),
                 
                 
              )
    
    f.suptitle('H')    
  
f.colorbar(W , label='sea ice concentration [%]') #, ax=ax1)
ax1.set_extent([-180, 180, 90, 50], ccrs.PlateCarree())
ax1.set_title('DJF')
ax1.gridlines(draw_labels=True) 
ax1.coastlines()
plt.tight_layout()
    