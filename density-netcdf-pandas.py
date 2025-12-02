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

path_to_SUMup_folder = 'C:/Users/bav/GitHub/SUMup/SUMup-2024/'
path_to_SUMup_folder = 'data/'
df_density = xr.open_dataset(
    path_to_SUMup_folder+'SUMup_2024_density_greenland.nc',
    group='DATA').to_dataframe()

ds_meta = xr.open_dataset(
    path_to_SUMup_folder + 'SUMup_2024_density_greenland.nc',
    group='METADATA')

# more elegant decoding
for v in ['profile','reference','reference_short','method']:
    ds_meta[v] = ds_meta[v].str.decode('utf-8')

# ds_meta contain the meaning of profile_key, reference_key, method_key being
# used in df_density

# % creating a metadata frame
# that contains, for each unique location, the important information
# (lat/lon, reference...)
df_meta = df_density[
    ['profile_key','latitude','longitude','timestamp','reference_key','method_key']
    ].drop_duplicates()

df_meta['profile_name'] = ds_meta.profile.sel(profile_key= df_meta.profile_key.values)
df_meta['method'] = ds_meta.method.sel(method_key= df_meta.method_key.values)
df_meta['reference'] = ds_meta.reference.sel(reference_key= df_meta.reference_key.values)
df_meta['reference_short'] = ds_meta.reference_short.sel(reference_key= df_meta.reference_key.values)
df_meta = df_meta.set_index('profile_key')


# %% plotting latitude lontgitudes
df_meta[['latitude','longitude']].plot.scatter(x='longitude',y='latitude')

# %% plotting the 10 first profiles
for ind in df_meta.index[:10]:
    plt.figure()
    df_density.loc[df_density.profile_key==ind,
            ['density', 'midpoint', 'profile_key']
        ].plot.scatter(
            x='density',
            y='midpoint',
            )
    plt.gca().invert_yaxis()
    plt.title(
        "Profile %s (nr. %i)\nfrom %s"%(
            df_meta.loc[ind, 'profile_name'],
            ind,
            df_meta.loc[ind, 'reference_short'],
            ))

# %% filtering by location
# lets select the profiles tha are south of 67 deg N
df_meta_south = df_meta.loc[df_meta.latitude < 67,:].copy()

print(df_meta_south)

plt.figure()
df_meta[['latitude','longitude']].plot(ax=plt.gca(), x='longitude',y='latitude',
                                       marker='o',ls='None', label='all data')
df_meta_south[['latitude','longitude']].plot(ax=plt.gca(), x='longitude',y='latitude',
                                       marker='o',ls='None', label='selected')
plt.legend()
# we can isolate the density data for those southern profiles
df_density_south = df_density.loc[
    df_density.profile_key.isin(df_meta_south.index)].copy()
# but we could also continue with df_density and only use profiles from df_meta_south

# %% plotting the 10 first southern profiles
for ind in df_meta_south.index[:10]:
    plt.figure()
    df_density.loc[df_density.profile_key==ind,
            ['density', 'midpoint', 'profile_key']
        ].plot.scatter(
            x='density',
            y='midpoint',
            )
    plt.gca().invert_yaxis()
    plt.title(
        "Profile %s (nr. %i)\nfrom %s"%(
            df_meta.loc[ind, 'profile_name'],
            ind,
            df_meta.loc[ind, 'reference_short'],
            ))

# %% finding the closest profile to given coordinates
# easiest if you use the following function
def nearest_latlon_profile(df, points, return_value=True):
    # inspired from https://github.com/blaylockbk/pyBKB_v3/blob/master/demo/Nearest_lat-lon_Grid.ipynb

    if 'lat' in df: df = df.rename(dict(lat='latitude', lon='longitude'))
    if isinstance(points, tuple): points = [points]

    xs = []; distances = []  # distance between the pair of points

    for point in points:
        assert len(point) == 2, "``points`` should be a tuple or list of tuples (lat, lon)"

        p_lat, p_lon = point
        # Find absolute difference between requested point and the grid coordinates.
        abslat = np.abs(df.latitude - p_lat)
        abslon = np.abs(df.longitude - p_lon)

        # Create grid of the maximum values of the two absolute grids
        c = np.maximum(abslon, abslat)

        # Find location where lat/lon minimum absolute value intersects
        x = np.where(c == np.min(c))[0][0]
        xs.append(x)

        # Matched Grid lat/lon
        g_lat = df.iloc[x,:].latitude
        g_lon = df.iloc[x,:].longitude

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
    return df.iloc[xs,:].index

coord_list = [(71, -45), (68, -45), (77, -55), (76, -38)]
ind_list = nearest_latlon_profile(df_meta, coord_list)

# plotting coordinates
plt.figure()
df_meta[['latitude','longitude']].plot.scatter(ax=plt.gca(),x='longitude',y='latitude')
plt.gca().plot(np.array(coord_list)[:,1],
            np.array(coord_list)[:,0], marker='^',
            ls='None', label='target',
            color='tab:red')
# note that normally df_meta.sel(profile=ind) should work but there is currently
# problems with duplicate entries. The following should work anyway.
df_meta.loc[df_meta.index.isin(ind_list),
            ['latitude','longitude']
            ].plot(ax=plt.gca(),
                           x='longitude',
                           y='latitude',
                           label='closest',
                           marker='d',ls='None',
                           color='tab:orange')
plt.legend()

# plotting profiles
for ind in ind_list:
    plt.figure()
    df_density.loc[df_density.profile_key==ind,
            ['density', 'midpoint', 'profile_key']
        ].plot.scatter(
            x='density',
            y='midpoint',
            )
    plt.gca().invert_yaxis()
    plt.title(
        "Profile %s (nr. %i)\nfrom %s"%(
            df_meta.loc[ind, 'profile_name'],
            ind,
            df_meta.loc[ind, 'reference_short'],
            ))


# %% Selecting data from a given source
ref_target = 'Miège et al. (2013)'
# finding the profiles that are in

df_meta_selec = df_meta.loc[df_meta.reference_short==ref_target]

tmp = df_density.loc[df_density.profile_key.isin(df_meta_selec.index),
                     ['profile_key', 'timestamp']].drop_duplicates().set_index('profile_key')

# plotting them on a map
df_meta_selec[['latitude','longitude']].plot.scatter(x='longitude',y='latitude')
# if you want to add (overlapping) labels
# for k, v in df_meta_selec[['longitude','latitude','profile_name']].drop_duplicates(
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
    df_meta_selec = gpd.GeoDataFrame(df_meta_selec,
                                    geometry=gpd.points_from_xy(
                                        df_meta_selec['longitude'],
                                        df_meta_selec['latitude']))
    df_meta_selec = df_meta_selec.set_crs(4326).to_crs(3413)
    land = gpd.read_file('ancil/greenland_land_3413.shp')
    ice = gpd.read_file('ancil/greenland_ice_3413.shp')

    plt.figure()
    ax=plt.gca()
    land.plot(ax=ax, color='gray')
    ice.plot(ax=ax, color='lightblue')
    df_meta_selec.plot(ax=ax, color='k', marker='^')
    ax.set_title('GEUS snow profiles')
    ax.axis('off')

# %% Plotting recent profiles that have, for most sites a snowpit and a firn core
plt.close('all')
fig, ax = plt.subplots(2,8,sharex=True,sharey=True)
ax=ax.flatten()
count=0
df_meta_selec = df_meta_selec.loc[df_meta_selec.profile_name.str.contains('GC-Net_'),:]
df_meta_selec['site'] = df_meta_selec.profile_name.str.split('_').str[1].str.split('_').str[0].str.split(' ').str[0]
for year in range(2022,2025):
    print(year)
    for site in np.unique(df_meta_selec.loc[df_meta_selec.timestamp.dt.year==year,'site']):
        tmp_site = df_meta_selec.loc[
            (df_meta_selec.site==site)&(df_meta_selec.timestamp.dt.year==year), :]
        print('    ',site)
        for ind in tmp_site.index:
            print('        ', df_meta_selec.loc[ind,'profile_name'])
            data = df_density.loc[df_density.profile_key==ind,
                    ['density', 'midpoint']
                ].sort_values(by='midpoint')
            if 'core' in df_meta_selec.loc[[ind],'profile_name'].to_list()[0].lower():
                method= 'core'
            else:
                method='snowpit'
            data.plot(
                    drawstyle="steps-mid",
                    x='density',
                    y='midpoint',
                    ax=ax[count],
                    label=method,
                    )

        ax[count].set_xlim(250, 600)
        ax[count].set_ylim(2, 0)
        ax[count].grid()
        ax[count].legend()
        ax[count].set_title(site+' '+str(year))
        # if isinstance(df_meta_selec.loc[ind,'profile_name'],str):
        #     ax[count].set_title(df_meta_selec.loc[ind,'profile_name'],
        #         fontsize=8)
        # else:
        #     ax[count].set_title(np.unique(df_meta_selec.loc[ind,'profile_name'].values),
        #     fontsize=8)
        count += 1
        if count ==len(ax):
            fig, ax = plt.subplots(2,8,sharex=True,sharey=True)
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
    df_density.loc[df_density.profile_key==ind,
            ['density', 'midpoint']
        ].sort_values(by='midpoint').plot(
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
    ax[count].set_title(df_meta_geus.loc[ind,'profile_name'],
        fontsize=11)
    count += 1
    if count ==len(ax):
        fig, ax = plt.subplots(4,5,sharex=True,sharey=True)
        ax=ax.flatten()
        ax[0].invert_yaxis()
        count=0

# %% Selecting data from a given source
ref_target = 'Miège et al. (2013)'
# finding the profiles that are in

df_meta_selec = df_meta.loc[df_meta.reference_short==ref_target]

tmp = df_density.loc[df_density.profile_key.isin(df_meta_selec.index),
                     ['profile_key', 'timestamp']].drop_duplicates().set_index('profile_key')

# plotting them on a map
df_meta_selec[['latitude','longitude']].plot.scatter(x='longitude',y='latitude')
# if you want to add (overlapping) labels
# for k, v in df_meta_selec[['longitude','latitude','profile_name']].drop_duplicates(
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
    df_meta_selec = gpd.GeoDataFrame(df_meta_selec,
                                    geometry=gpd.points_from_xy(
                                        df_meta_selec['longitude'],
                                        df_meta_selec['latitude']))
    df_meta_selec = df_meta_selec.set_crs(4326).to_crs(3413)
    land = gpd.read_file('ancil/greenland_land_3413.shp')
    ice = gpd.read_file('ancil/greenland_ice_3413.shp')

    plt.figure()
    ax=plt.gca()
    land.plot(ax=ax, color='gray')
    ice.plot(ax=ax, color='lightblue')
    df_meta_selec.plot(ax=ax, color='k', marker='^')
    ax.set_title('GEUS snow profiles')
    ax.axis('off')

#  Plotting recent profiles that have, for most sites a snowpit and a firn core
plt.close('all')
fig, ax = plt.subplots(1,len(df_meta_selec.index),sharex=True,sharey=True)
ax=ax.flatten()
count=0

for ind in df_meta_selec.index:
    name  = df_meta_selec.loc[ind, 'profile_name']
    print('        ', df_meta_selec.loc[ind,'profile_name'])
    data = df_density.loc[df_density.profile_key==ind,
            ['density', 'midpoint']
        ].sort_values(by='midpoint')

    data.plot(
            drawstyle="steps-mid",
            x='density',
            y='midpoint',
            ax=ax[count],
            label='__nolegend__'
            )
    
    ax[count].set_xlim(300, 950)
    ax[count].set_ylim(50, 0)
    ax[count].grid()
    ax[count].set_title(name)
    count+=1
    

