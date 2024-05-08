# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import geopandas as gpd

path_to_SUMup_folder = 'C:/Users/bav/GitHub/SUMup/SUMup-2024/'
df_sumup = xr.open_dataset(path_to_SUMup_folder+'SUMup 2024 beta/SUMup_2024_temperature_greenland.nc', 
                           group='DATA').to_dataframe()
ds_meta = xr.open_dataset(path_to_SUMup_folder+'SUMup 2024 beta/SUMup_2024_temperature_greenland.nc', 
                           group='METADATA')
df_sumup.method_key = df_sumup.method_key.replace(np.nan,-9999)
df_sumup['method'] = ds_meta.method.sel(method_key = df_sumup.method_key.values).astype(str)
df_sumup['name'] = ds_meta.name.sel(name_key = df_sumup.name_key.values).astype(str)
df_sumup['reference'] = (ds_meta.reference
                         .drop_duplicates(dim='reference_key')
                         .sel(reference_key=df_sumup.reference_key.values)
                         .astype(str))
df_sumup['reference_short'] = (ds_meta.reference_short
                         .drop_duplicates(dim='reference_key')
                         .sel(reference_key=df_sumup.reference_key.values)
                         .astype(str))
df_ref = ds_meta.reference.to_dataframe()

# selecting Greenland metadata measurements
df_meta = df_sumup.loc[df_sumup.latitude>0, 
                  ['latitude', 'longitude', 'name_key', 'name', 'method_key',
                   'reference_short','reference', 'reference_key']
                  ].drop_duplicates()



# %% plotting latitude lontgitudes
ice = gpd.GeoDataFrame.from_file(path_to_SUMup_folder + "doc/GIS/greenland_ice_3413.shp")
land = gpd.GeoDataFrame.from_file(path_to_SUMup_folder + "doc/GIS/greenland_land_3413.shp")

plt.figure()
land.to_crs(4326).plot(ax=plt.gca())
ice.to_crs(4326).plot(ax=plt.gca(),color='lightblue')
df_meta.loc[df_meta.latitude>0, ['latitude','longitude']].plot.scatter(
    ax=plt.gca(), x='longitude',y='latitude', color='k', marker='.')

# ant_land = gpd.GeoDataFrame.from_file(path_to_SUMup_folder + "doc/GIS/Medium_resolution_vector_polygons_of_the_Antarctic_coastline.shp")

# plt.figure()
# ant_land.to_crs(4326).plot(ax=plt.gca())
# df_meta.loc[df_meta.latitude<0, ['latitude','longitude']].plot.scatter(
#     ax=plt.gca(), x='longitude',y='latitude', color='k', marker='.')

# %% Plotting in EPSG:3413 
# Note: the repojection of all the data points take a very long time
df_gr =df_meta.loc[df_meta.latitude>0]

df_gr = (
    gpd.GeoDataFrame(df_gr, geometry=gpd.points_from_xy(df_gr.longitude, df_gr.latitude))
    .set_crs(4326)
    .to_crs(3413)
)

df_gr['x_3413'] = df_gr.geometry.x.values
df_gr['y_3413'] = df_gr.geometry.y.values

plt.figure()
ax=plt.gca()
land.plot(ax=ax)
ice.plot(ax=ax,color='lightblue')
df_gr.plot(ax=ax,  
        color='k', marker='.', legend=False) 
ax.set_xticks([])
ax.set_yticks([])

# # for antarctica
# df_ant =df_meta.loc[df_meta.latitude<0]

# df_ant = (
#     gpd.GeoDataFrame(df_ant, geometry=gpd.points_from_xy(df_ant.longitude, df_ant.latitude))
#     .set_crs(4326)
#     .to_crs(3031)
# )

# df_ant['x_3413'] = df_ant.geometry.x.values
# df_ant['y_3413'] = df_ant.geometry.y.values

# plt.figure()
# ax=plt.gca()
# ant_land.to_crs(3031).plot(ax=ax, color='lightblue')
# df_ant.plot(ax=ax,  
#         color='k', marker='.', legend=False) 
# ax.set_xticks([])
# ax.set_yticks([])

# %% finding the closest profile to given coordinates
# easiest if you use the following function
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

query_point = [[77.1667, -61.1333]] # Camp Century
all_points = df_meta[['latitude', 'longitude']].values
df_meta['distance_from_query_point'] = distance.cdist(all_points, query_point, get_distance)
min_dist = 20 # in km
df_meta_selec = df_meta.loc[df_meta.distance_from_query_point<min_dist, :]   

# plotting coordinates
plt.figure()
df_meta[['latitude','longitude']].plot.scatter(ax=plt.gca(),
                                               x='longitude',y='latitude',
                                               marker='.',
                                               label='all points')
plt.gca().plot(np.array(query_point)[:,1],
            np.array(query_point)[:,0], marker='^', 
            ls='None', label='target',
            color='tab:red')
df_meta_selec.plot(ax=plt.gca(), x='longitude', y='latitude',
              label='closest', marker='d',ls='None', color='tab:orange')
plt.legend()

# %% plotting individual temperature records
import matplotlib
cmap = matplotlib.cm.get_cmap('tab10')

plt.figure()
for count, ref in enumerate(df_meta_selec.reference_short.unique()):
    label = ref
    for n in df_meta_selec.loc[df_meta_selec.reference_short==ref, 'name_key'].drop_duplicates().values:
        df_sumup.loc[
            df_sumup.name_key == n, :
            ].plot(ax=plt.gca(), x='timestamp', y='temperature',
                      color = cmap(count),
                      marker='o',ls='None',
                      label=label, alpha=0.7, legend=False
                      )
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),title='Sources:')
        plt.ylabel('temperature (°C)')
        label='_nolegend_'
plt.title('Observations within '+str(min_dist)+' km of '+str(query_point))

# %% plotting 2D
df_sumup_selec = df_sumup.loc[df_sumup.name.isin(df_meta_selec.name)]
plt.figure()
sc = plt.scatter(df_sumup_selec.timestamp,
                 -df_sumup_selec.depth, 50, 
                 df_sumup_selec.temperature)
plt.colorbar(sc)
plt.grid()
plt.title('Observations within '+str(min_dist)+' km of '+str(query_point))