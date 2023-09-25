# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

path_to_SUMup_folder = 'C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Data/SUMup/SUMup 2023 beta/'

ds_density = xr.open_dataset(path_to_SUMup_folder + 'density/netcdf/SUMup_2023_density_greenland.nc')

metadata_fields = ['profile_key', 'latitude','longitude','elevation',
                   'profile_name','reference','reference_short','method']
# separating data and metadata
ds_meta = ds_density[metadata_fields].rename({'profile_key':'profile'})
ds_density = ds_density.drop(metadata_fields)

# %% plotting latitude lontgitudes
ds_meta[['latitude','longitude']].plot.scatter(x='longitude',y='latitude')

# %% plotting the 10 first profiles
for ind in ds_meta.profile[:10]:
    plt.figure()
    ds_density[
            ['density', 'midpoint', 'profile']
        ].where(ds_density.profile==ind).plot.scatter(
            x='density', 
            y='midpoint',
            yincrease=False,
            )
    plt.title(
        "Profile %s (nr. %i)\nfrom %s"%(
            ds_meta.profile_name.sel(profile=ind).values,
            ind,
            ds_meta.reference_short.sel(profile=ind).values
            ))

# %% filtering by location
# lets select the profiles tha are south of 67 deg N
ds_meta_south = ds_meta.where(ds_meta.latitude < 67, drop=True)

print(ds_meta_south)

plt.figure()
ds_meta[['latitude','longitude']].plot.scatter(x='longitude',y='latitude',label='all data')
ds_meta_south[['latitude','longitude']].plot.scatter(x='longitude',y='latitude', label='selected')
plt.legend()
# we can isolate the density data for those southern profiles
ds_density_south = ds_density.where(
    ds_density.profile.isin(ds_meta_south.profile), 
    drop = True)
# but we could also continue with ds_density and only use profiles from ds_meta_south

# %% plotting the 10 first southern profiles
for ind in ds_meta_south.profile[:10]:
    plt.figure()
    ds_density[
            ['density', 'midpoint', 'profile']
        ].where(ds_density.profile==ind).plot.scatter(
            x='density', 
            y='midpoint',
            yincrease=False,
            )
    plt.title(
        "Profile %s (nr. %i)\nfrom %s"%(
            ds_meta.profile_name.sel(profile=ind).values,
            ind,
            ds_meta.reference_short.sel(profile=ind).values
            ))
    
# %% finding the closest profile to given coordinates
# easiest if you use the following function
def nearest_latlon_profile(ds, points, return_value=True):
    # inspired from https://github.com/blaylockbk/pyBKB_v3/blob/master/demo/Nearest_lat-lon_Grid.ipynb

    if 'lat' in ds: ds = ds.rename(dict(lat='latitude', lon='longitude'))
    if isinstance(points, tuple): points = [points]
        
    xs = []; distances = []  # distance between the pair of points
    
    for point in points:
        assert len(point) == 2, "``points`` should be a tuple or list of tuples (lat, lon)"
        
        p_lat, p_lon = point
        # Find absolute difference between requested point and the grid coordinates.
        abslat = np.abs(ds.latitude - p_lat)
        abslon = np.abs(ds.longitude - p_lon)

        # Create grid of the maximum values of the two absolute grids
        c = np.maximum(abslon, abslat)

        # Find location where lat/lon minimum absolute value intersects
        x = np.where(c == np.min(c))[0][0]
        xs.append(x)

        # Matched Grid lat/lon
        g_lat = ds.latitude.isel(profile=x).values
        g_lon = ds.longitude.isel(profile=x).values
               
        R = 6373.0  # approximate radius of earth in km

        lat1 = np.deg2rad(p_lat); lon1 = np.deg2rad(p_lon)
        lat2 = np.deg2rad(g_lat); lon2 = np.deg2rad(g_lon)
        dlon = lon2 - lon1; dlat = lat2 - lat1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * c
        distances.append(distance)
        print(point, 'closest to profile',x,' (%0.4f°N, %0.4f°E) %0.4f km away'%(
            g_lat, g_lon, R * c))
    return ds.profile.isel(profile=xs).values
    
coord_list = [(71, -45), (68, -45), (77, -55), (76, -38)]
ind_list = nearest_latlon_profile(ds_meta, coord_list)

# plotting coordinates
plt.figure()
ds_meta[['latitude','longitude']].plot.scatter(x='longitude',y='latitude')
plt.scatter(np.array(coord_list)[:,1],
            np.array(coord_list)[:,0], marker='^', label='target')
# note that normally ds_meta.sel(profile=ind) should work but there is currently
# problems with duplicate entries. The following should work anyway.
ds_meta.where(
    ds_meta.profile.isin(ind_list),
    drop = True
    )[
      ['latitude','longitude']
      ].plot.scatter(x='longitude',
                     y='latitude',
                     label='closest',
                     marker='d',
                     color='green')
plt.legend()

for ind in ind_list:
    plt.figure()
    ds_density[
            ['density', 'midpoint', 'profile']
        ].where(ds_density.profile==ind).plot.scatter(
            x='density', 
            y='midpoint',
            yincrease=False,
            )
    plt.title(
        "Profile %s (nr. %i)\nfrom %s"%(
            ds_meta.profile_name.sel(profile=ind).values,
            ind,
            ds_meta.reference_short.sel(profile=ind).values
            ))
    
# %% Selecting data from a given source
df_meta = ds_meta.to_dataframe()
ref_target = 'GEUS snow and firn data (2023)'
# finding the profiles that are in 

df_meta_geus = df_meta.loc[df_meta.reference_short==ref_target]
tmp=ds_density.where(ds_density.profile.isin(df_meta_geus.index), drop=True).to_dataframe()[['profile','date']].drop_duplicates().set_index('profile')
df_meta_geus['date'] = tmp.loc[df_meta_geus.index]
# plotting them on a map
df_meta_geus[['latitude','longitude']].plot.scatter(x='longitude',y='latitude')
# if you want to add (overlapping) labels
# for k, v in df_meta_geus[['longitude','latitude','profile_name']].drop_duplicates(
#         subset='profile_name'
#         ).set_index('profile_name').iterrows():
#     plt.annotate(k, v)
plt.title('GEUS snow profiles')
# more advanced: switch to epsg:3413 projection and add background
try:
    import geopandas as gpd
    found = True
except:
    print('>>> Warning: geopandas package was not found.')
    print('Please install geopandas for reprojected maps.')
    print('Skipping this part.')
    found = False

if found:
    df_meta_geus = gpd.GeoDataFrame(df_meta_geus, 
                                    geometry=gpd.points_from_xy(
                                        df_meta_geus['longitude'], 
                                        df_meta_geus['latitude']))
    df_meta_geus = df_meta_geus.set_crs(4326).to_crs(3413)
    land = gpd.read_file('ancil/greenland_land_3413.shp')
    ice = gpd.read_file('ancil/greenland_ice_3413.shp')
    
    plt.figure()
    ax=plt.gca()
    land.plot(ax=ax, color='gray')
    ice.plot(ax=ax, color='lightblue')
    df_meta_geus.plot(ax=ax, color='k', marker='^')
    ax.set_title('GEUS snow profiles')
    ax.axis('off')

# %% Plotting all profiles
plt.close('all')
fig, ax = plt.subplots(3,5,sharex=True)
ax=ax.flatten()
count=0
for ind in df_meta_geus.index:
    print(df_meta_geus.loc[ind,'profile_name'])
    ds_density[
            ['density', 'midpoint']
        ].where(ds_density.profile==ind, drop=True).to_dataframe().plot(
            drawstyle="steps-mid",
            x='density', 
            y='midpoint',
            ax=ax[count],
            )
    ax[count].invert_yaxis()
    ax[count].set_xlim(200, 900)
    ax[count].grid()
    ax[count].get_legend().remove()
    ax[count].set_title(df_meta_geus.loc[ind,'profile_name'],
        fontsize=8)
    count += 1
    if count ==len(ax):
        fig, ax = plt.subplots(5,5,sharex=True,sharey=True)
        ax=ax.flatten()
        count=0

# %% plot only the cores
df_meta_geus_cores = df_meta_geus.loc[df_meta_geus.profile_name.str.lower().str.contains('core'),:]
plt.close('all')

fig, ax = plt.subplots(4,5,figsize=(15,10),
                       sharex=True,sharey=True)
ax=ax.flatten()
ax[0].invert_yaxis()
count=0
for ind in df_meta_geus_cores.index:
    print(df_meta_geus_cores.loc[ind,'profile_name'])
    ds_density[
            ['density', 'midpoint']
        ].where(ds_density.profile==ind, drop=True).to_dataframe().sort_values(by='midpoint').plot(
            drawstyle="steps-mid",
            x='density', 
            y='midpoint',
            ax=ax[count],
            )
    ax[count].set_xlim(200, 1000)
    ax[count].set_xlabel('Density (kg m-3)')
    ax[count].set_ylabel('Depth (m)')
    ax[count].grid()
    ax[count].get_legend().remove()
    ax[count].set_title(df_meta_geus.loc[ind,'profile_name'] + str(df_meta_geus.loc[ind,'date'].year),
        fontsize=11)
    count += 1
    if count ==len(ax):
        fig, ax = plt.subplots(4,5,sharex=True,sharey=True)
        ax=ax.flatten()
        ax[0].invert_yaxis()
        count=0
