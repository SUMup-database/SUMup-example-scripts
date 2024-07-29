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
import pandas as pd

path_to_SUMup_folder = 'C:/Users/bav/GitHub/SUMup/SUMup-2024/SUMup 2024 beta/'

df_density = xr.open_dataset(path_to_SUMup_folder + 'SUMup_2024_density_greenland.nc', group='DATA').to_dataframe()

ds_meta = xr.open_dataset(path_to_SUMup_folder + 'SUMup_2024_density_greenland.nc', group='METADATA')
# ds_meta contain the meaning of profile_key, reference_key, method_key being
# used in df_density

# % creating a metadata frame 
# that contains, for each unique location, the important information
# (lat/lon, reference...)
df_density_meta = df_density[
    ['profile_key','latitude','longitude','timestamp','reference_key','method_key']
    ].drop_duplicates()

df_density_meta['profile'] = (ds_meta.profile
                           .loc[dict(profile_key= df_density_meta.profile_key.values)]
                                     .values)
df_density_meta['method'] = (ds_meta.method
                     .drop_duplicates(dim='method_key')  # this is due to a bug, will be fixed soon
                     .loc[dict(method_key= df_density_meta.method_key.values)]
                     .values)
df_density_meta['reference'] = (ds_meta.reference
                        .loc[dict(reference_key= df_density_meta.reference_key.values)]
                        .values)
df_density_meta['reference_short'] = (ds_meta.reference_short
                              .loc[dict(reference_key= df_density_meta.reference_key.values)]
                              .values)
df_density_meta = df_density_meta.set_index('profile_key')

# % loading stratigraphy
df_strat = xr.open_dataset(path_to_SUMup_folder + 'SUMup_2024_stratigraphy_greenland.nc', group='DATA').to_dataframe()

ds_strat_meta = xr.open_dataset(path_to_SUMup_folder + 'SUMup_2024_stratigraphy_greenland.nc', group='METADATA')

df_strat_meta = df_strat[
    ['profile_key','latitude','longitude','timestamp','reference_key','method_key']
    ].drop_duplicates()

df_strat_meta['profile'] = (ds_strat_meta.profile
                           .loc[dict(profile_key= df_strat_meta.profile_key.values)]
                                     .values)
df_strat_meta['method'] = (ds_strat_meta.method
                     .drop_duplicates(dim='method_key')  # this is due to a bug, will be fixed soon
                     .loc[dict(method_key= df_strat_meta.method_key.values)]
                     .values)
df_strat_meta['reference'] = (ds_strat_meta.reference
                        .loc[dict(reference_key= df_strat_meta.reference_key.values)]
                        .values)
df_strat_meta['reference_short'] = (ds_strat_meta.reference_short
                              .loc[dict(reference_key= df_strat_meta.reference_key.values)]
                              .values)
df_strat_meta = df_strat_meta.set_index('profile_key')

# %% matching profiles
df_density_meta_2 = df_density_meta.reset_index().copy()
df_strat_meta_2 = df_strat_meta.reset_index().copy()

df_density_meta_2['latitude'] = df_density_meta_2.latitude.round(3)
df_density_meta_2['longitude'] = df_density_meta_2.longitude.round(3)
df_strat_meta_2['latitude'] = df_strat_meta_2.latitude.round(3)
df_strat_meta_2['longitude'] = df_strat_meta_2.longitude.round(3)

# Merge the DataFrames on 'latitude', 'longitude', and 'timestamp'
merged_df = pd.merge(df_strat_meta_2, 
                     df_density_meta_2, on=['latitude', 'longitude', 'timestamp'],
                     suffixes=('_strat', '_density'))
# %%

# Number of profiles per figure
profiles_per_fig = 6

# Iterate over the profile keys in chunks of 6
for i in range(0, len(merged_df), profiles_per_fig):
    fig, axs = plt.subplots( 1,profiles_per_fig, figsize=(18, 8.3))  
    fig.subplots_adjust(wspace=0.2)  # Adjust the space between the plots
    
    for j in range(profiles_per_fig):
        if i + j >= len(merged_df):
            break
        
        p_key_strat = merged_df.profile_key_strat.iloc[i + j]
        p_key_density = merged_df.profile_key_density.iloc[i + j]
        
        df_strat_filtered = df_strat.loc[df_strat.profile_key == p_key_strat, ['start_depth', 'stop_depth', 'ice_fraction_perc']]
        df_strat_filtered.loc[df_strat_filtered.ice_fraction_perc == -999, 
                              'ice_fraction_perc'] = np.nan
        df_density_filtered = df_density.loc[df_density.profile_key == p_key_density, 
                                             ['start_depth', 'stop_depth', 'density']]
        
        ax1 = axs[j]
        
        # Plot ice_fraction_perc as filled regions
        for _, row in df_strat_filtered.iterrows():
            ax1.fill_betweenx([row['start_depth'], row['stop_depth']], 0, row['ice_fraction_perc'], color='blue', alpha=0.3)
        
        ax1.invert_yaxis()
        ax1.set_xlabel('Ice Fraction (%)')
        if j == 0:
            ax1.set_ylabel('Depth (m)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create a second y-axis for density
        ax2 = ax1.twiny()
        
        # Plot density as step-like lines
        step_x = np.ravel(np.column_stack((df_density_filtered['density'], df_density_filtered['density'])))
        step_y = np.ravel(np.column_stack((df_density_filtered['start_depth'], df_density_filtered['stop_depth'])))
        ax2.step(step_x, step_y, where='post', color='black', linewidth=1.5)
        
        if j==0:
            ax2.set_xlabel('Density (kg/m³)')
        ax2.tick_params(axis='x', labelcolor='black')
        ax2.set_ylim(max(ax2.get_ylim()[0],ax1.get_ylim()[0]),0)
        ax1.set_ylim(max(ax2.get_ylim()[0],ax1.get_ylim()[0]),0)
        
        # Set subplot title
        ax1.set_title(df_strat_meta.loc[p_key_strat, 'profile'] + '\n' + \
                      df_density_meta.loc[p_key_density, 'profile'])
    
    plt.show()

#%%
ind_unique = merged_df[['latitude', 'longitude']].drop_duplicates().index
merged_df.loc[ind_unique, 'profile_strat']
ind_select = 7
lat= merged_df.loc[ind_select,'latitude']
lon= merged_df.loc[ind_select,'longitude']
msk = (merged_df.latitude == lat) & (merged_df.longitude == lon)
merged_df_select = merged_df.loc[msk]
# Iterate over the profile keys in chunks of 6
for i in range(0, len(merged_df_select), profiles_per_fig):

# %%
msk = ds_strat_meta.profile.str.lower().str.contains('dye') & ds_strat_meta.profile.str.contains('2')
df_strat_meta = ds_strat_meta.profile.to_dataframe()
for p in df_meta.loc[msk].index[-2:]:
    print(df_meta.loc[p])
    break
    df_density.loc[df_density.profile_key == p, :]
    
    ds_strat_meta.where(ds_strat_meta.profile == df_meta.loc[p].profile, drop=True)
    df_density.loc[df_density.profile_key == p, :]
    
    
    plt.figure()
    
    df_strat.loc[]
    plt.plot()

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
    
# %% finding all profiles within a certain radius of a given set of coordinates
from scipy.spatial import distance
from math import sin, cos, sqrt, atan2, radians

def get_distance(point1, point2):
    R = 6370
    lat1 = radians(point1[0])  #insert value
    lon1 = radians(point1[1])
    lat2 = radians(point2[0])
    lon2 = radians(point2[1])

    dlon = lon2 - lon1
    dlat = lat2- lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

query_point = [[66.47771, -46.28606]] # DYE-2
all_points = df_density_meta[['latitude', 'longitude']].values
df_density_meta['distance_from_query_point'] = distance.cdist(all_points, query_point, get_distance)

min_dist = 15 # in km
df_density_meta_selec = df_density_meta.loc[df_density_meta.distance_from_query_point<min_dist, :]   


all_points = df_strat_meta[['latitude', 'longitude']].values
df_strat_meta['distance_from_query_point'] = distance.cdist(all_points, query_point, get_distance)
df_strat_meta_selec = df_strat_meta.loc[df_strat_meta.distance_from_query_point<min_dist, :]   


# plotting coordinates
plt.figure()
df_density_meta[['latitude','longitude']].plot.scatter(ax=plt.gca(),
                                                       x='longitude',y='latitude')
plt.gca().plot(np.array(coord_list)[:,1],
            np.array(coord_list)[:,0], marker='^', 
            ls='None', label='target',
            color='tab:red')
# note that normally df_meta.sel(profile=ind) should work but there is currently
# problems with duplicate entries. The following should work anyway.
df_density_meta_selec[
            ['latitude','longitude']
            ].plot(ax=plt.gca(),
                           x='longitude',
                           y='latitude',
                           label='closest',
                           marker='d',ls='None',
                           color='tab:orange')
plt.legend()

# %% plotting profiles by year
list_year = np.unique(df_density_meta_selec.timestamp.dt.year)
fig, axs = plt.subplots( 1,len(list_year), figsize=(18, 8.3), sharey=True)  
fig.subplots_adjust(wspace=0.2)  # Adjust the space between the plots
for year, ax in zip(list_year, axs):
    print(year)
    ax1 = ax # for ice content
    
    list_key_strat = df_strat_meta_selec.loc[df_strat_meta_selec.timestamp.dt.year == year,:].index
    if len(list_key_strat) == 0:
        ax1.plot( [np.nan, np.nan], [0, 20])
    for p_key_strat in list_key_strat:
        df_strat_filtered = df_strat.loc[df_strat.profile_key == p_key_strat, ['start_depth', 'stop_depth', 'ice_fraction_perc']]
        print('strat',p_key_strat, df_strat_meta.loc[p_key_strat, 'profile'])
        
        df_strat_filtered.loc[df_strat_filtered.ice_fraction_perc == -999, 
                              'ice_fraction_perc'] = np.nan
        # Plot ice_fraction_perc as filled regions
        for _, row in df_strat_filtered.iterrows():
            ax1.fill_betweenx([row['start_depth'], row['stop_depth']], 0,
                              row['ice_fraction_perc'], color='blue', alpha=0.3)
            
    ax1.invert_yaxis()
    ax1.set_xlabel('Ice Fraction (%)')
    if j == 0:
        ax1.set_ylabel('Depth (m)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # plotting density
    ax2 = ax1.twiny() # for density
    for p_key_density in df_density_meta_selec.loc[df_density_meta_selec.timestamp.dt.year == year,:].index:
        print('density',p_key_density, df_density_meta.loc[p_key_density, 'profile'])

        df_density_filtered = df_density.loc[df_density.profile_key == p_key_density, 
                                             ['start_depth', 'stop_depth', 'density']]


                
        # Plot density as step-like lines
        step_x = np.ravel(np.column_stack((df_density_filtered['density'], df_density_filtered['density'])))
        step_y = np.ravel(np.column_stack((df_density_filtered['start_depth'], df_density_filtered['stop_depth'])))
        ax2.step(step_x, step_y, where='post', color='black', alpha=0.5, linewidth=1.5)
        
    if year==list_year[0]:
        ax2.set_xlabel('Density (kg/m³)')
    ax2.tick_params(axis='x', labelcolor='black')
    # ax2.set_ylim(max(ax2.get_ylim()[0],ax1.get_ylim()[0]),0)
    # ax1.set_ylim(max(ax2.get_ylim()[0],ax1.get_ylim()[0]),0)
    ax2.set_title(str(year))
    ax2.set_xlim(200,920)
    ax1.set_xlim(0,100)
        
ax2.set_ylim(20,0)
ax1.set_ylim(20,0)
    
    
plt.show() 
    
# %% Selecting data from a given source
ref_target = 'GEUS snow and firn data (2023)'
# finding the profiles that are in 

df_meta_geus = df_meta.loc[df_meta.reference_short==ref_target]

tmp = df_density.loc[df_density.profile_key.isin(df_meta_geus.index), 
                     ['profile_key', 'timestamp']].drop_duplicates().set_index('profile_key')

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
    df_density.loc[df_density.profile_key==ind,
            ['density', 'midpoint']
        ].plot(
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
