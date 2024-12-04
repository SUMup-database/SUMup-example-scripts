# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd

path_to_sumup = 'C:/Users/bav/GitHub/SUMup/SUMup-2024/SUMup 2024 beta/SUMup_2024_SMB_csv/'
df_sumup = (pd.read_csv(path_to_sumup+'/SUMup_2024_SMB_greenland.csv',
                        low_memory=False)
            .rename(columns=dict(name='name_key', method='method_key', reference='reference_key')))


df_methods = pd.read_csv(path_to_sumup+'/SUMup_2024_SMB_methods.tsv',
                         sep='\t').set_index('key').method
df_names = pd.read_csv(path_to_sumup+'/SUMup_2024_SMB_names.tsv',
                         sep='\t').set_index('key').name
df_references = pd.read_csv(path_to_sumup+'/SUMup_2024_SMB_references.tsv',
                         sep='\t').set_index('key')

df_sumup.loc[df_sumup.method_key.isnull(), 'method_key'] = -9999
df_sumup['method'] = df_methods.loc[df_sumup.method_key].values
df_sumup['name'] = df_names.loc[df_sumup.name_key].values
df_sumup['reference'] = df_references.loc[df_sumup.reference_key, 'reference'].values
df_sumup['reference_short'] = df_references.loc[df_sumup.reference_key, 'reference_short'].values

df_references.reference.loc[df_references.index.duplicated()]


# selecting Greenland metadata measurements
df_meta = df_sumup.loc[df_sumup.latitude>0,
                  ['latitude', 'longitude', 'name_key', 'name', 'method_key',
                   'reference_short','reference', 'reference_key']
                  ].drop_duplicates()

# %% plotting in EPSG:4326
ice = gpd.GeoDataFrame.from_file(path_to_sumup + "../../doc/GIS/greenland_ice_3413.shp")
land = gpd.GeoDataFrame.from_file(path_to_sumup + "../../doc/GIS/greenland_land_3413.shp")

plt.figure()
land.to_crs(4326).plot(ax=plt.gca())
ice.to_crs(4326).plot(ax=plt.gca(),color='lightblue')
df_meta.plot(ax=plt.gca(), x='longitude', y='latitude',
        color='k', marker='.',ls='None', legend=False)

# %% Source selection and plotting in EPSG:3413
# Note: the repojection of all the data points take a very long time

# selecting Lewis airborne radar data using reference key
# first we can look at which references start with "Lewis"
print(df_references.loc[df_references.reference.str.startswith('Lewis')])
# we then use the key for Lewis et al. (2017)
df_selec = df_meta.loc[df_meta.reference_short == 'Lewis et al. (2017)', :].copy()

gdf = (
    gpd.GeoDataFrame(df_selec, geometry=gpd.points_from_xy(df_selec.longitude, df_selec.latitude))
    .set_crs(4326)
    .to_crs(3413)
)

df_selec['x_3413'] = gdf.geometry.x.values
df_selec['y_3413'] = gdf.geometry.y.values

plt.figure()
ax=plt.gca()
land.plot(ax=ax)
ice.plot(ax=ax,color='lightblue')
gdf.plot(ax=ax,
        color='k', marker='.', legend=False)
ax.set_xticks([])
ax.set_yticks([])
plt.title('\n'.join(df_selec.reference_short.drop_duplicates()))
# ref_list = df_10m["reference_short"].unique()
# df_10m["ref_id"] = [np.where(ref_list == x)[0] for x in df_10m["reference_key"]]

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

query_point = [[75.1, -42.32]] # NGRIP
all_points = df_meta[['latitude', 'longitude']].values
df_meta['distance_from_query_point'] = distance.cdist(all_points, query_point, get_distance)
min_dist = 20 # in km
df_meta_selec = df_meta.loc[df_meta.distance_from_query_point<min_dist, :]

# plotting coordinates
plt.figure()
df_meta[['latitude','longitude']].plot.scatter(ax=plt.gca(),
                                               x='longitude',y='latitude')
plt.gca().plot(np.array(query_point)[:,1],
            np.array(query_point)[:,0], marker='^',
            ls='None', label='target',
            color='tab:red')
df_meta_selec.plot(ax=plt.gca(), x='longitude', y='latitude',
              label='closest', marker='d',ls='None', color='tab:orange')
plt.legend()

# %% plotting individual smb records
import matplotlib
cmap = matplotlib.cm.get_cmap('tab10')


# warning: this accumualtion is not appropriate for very short measurements or for the ablation area
df_sumup['accumulation'] = df_sumup['smb'] / np.maximum(1,df_sumup.end_year-df_sumup.start_year)


plt.figure()
for count, ref in enumerate(df_meta_selec.reference_short.unique()):
    # each reference will be plotted in a different color
    label = ref
    for n in df_meta_selec.loc[df_meta_selec.reference_short==ref, 'name'].unique():
        # for each core or each point along a radar transect, a separate line needs
        # to be plotted
        df_stack = (df_sumup.loc[df_sumup.name==n,['start_year','end_year']]
                    .sort_values(by='start_year')
                    .stack().reset_index().drop(columns='level_1')
                    .rename(columns={0:'year'}))
        df_stack['accumulation'] = df_sumup.loc[df_stack.level_0, 'accumulation'].values
        df_stack.plot(ax=plt.gca(), x='year', y='accumulation',
                      color = cmap(count),
                      label=label, alpha=0.7, legend=False
                      )
        plt.legend(loc='upper left')
        plt.ylabel('SMB (m w.e. yr-1)')
        label='_nolegend_'
plt.title('Observations within '+str(min_dist)+' km of '+str(query_point))
