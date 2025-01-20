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

path_to_sumup = 'C:/Users/bav/GitHub/SUMup/SUMup-2024/SUMup 2024 beta/'
df_sumup = xr.open_dataset(path_to_sumup+'/SUMup_2024_SMB_greenland.nc',
                           group='DATA', lock=False).to_dataframe()
df_sumup = df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.start_year>1990)]
ds_meta = xr.open_dataset(path_to_sumup+'/SUMup_2024_SMB_greenland.nc',
                           group='METADATA', lock=False)

# decoding strings as utf-8
for v in ['name','reference','reference_short','method']:
    ds_meta[v] = ds_meta[v].str.decode('utf-8')

df_sumup.method_key = df_sumup.method_key.replace(np.nan,-9999)
df_sumup['method'] = ds_meta.method.sel(method_key = df_sumup.method_key.values)
df_sumup['name'] = ds_meta.name.sel(name_key = df_sumup.name_key.values)
df_sumup['reference'] = (ds_meta.reference
                          .sel(reference_key=df_sumup.reference_key.values))
df_sumup['reference_short'] = (ds_meta.reference_short
                         .sel(reference_key=df_sumup.reference_key.values))

df_ref = ds_meta.reference.to_dataframe()

# selecting Greenland metadata measurements
df_meta = df_sumup[
                  ['latitude', 'longitude', 'name_key', 'name', 'method_key',
                   'reference_short','reference', 'reference_key']
                  ].drop_duplicates()
# warning: this accumualtion is not appropriate for very short measurements or for the ablation area
df_sumup['accumulation'] = df_sumup.apply(
    lambda row: row['smb'] / max(1, (pd.to_datetime(row['end_date']) - pd.to_datetime(row['start_date'])).days / 365.25)
    if not (pd.isna(row['start_date']) or pd.isna(row['end_date']))
    else row['smb'] / max(1, row['end_year'] - row['start_year']),
    axis=1
)
# %% plotting in EPSG:4326
ice = gpd.GeoDataFrame.from_file("ancil/greenland_ice_3413.shp")
land = gpd.GeoDataFrame.from_file("ancil/greenland_land_3413.shp")

plt.figure()
land.to_crs(4326).plot(ax=plt.gca())
ice.to_crs(4326).plot(ax=plt.gca(),color='lightblue')
df_meta.plot(ax=plt.gca(), x='longitude', y='latitude',
        color='k', marker='.',ls='None', legend=False)

# %% Listing available source

print(df_ref.to_markdown())

# %% Source selection and plotting in EPSG:3413
# Note: the repojection of all the data points take a very long time


# selecting PARCA cores using key-word in the reference field
df_selec = df_meta.loc[df_meta.reference.str.startswith('Mosley') \
    | df_meta.reference.str.startswith('Bales'), :].copy()

# selecting Lewis airborne radar data using reference key
# df_selec = df_meta.loc[df_meta.reference_key == 138, :].copy()

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

query_point = [ #
                # [75.1, -42.32] # NGRIP
                [77.13, -61.03] # Camp Century
                # [77.45, -51.06] # NEEM
                # [72.5796, -37.6378] # GRIP
                # [72.5785, -38.4594] # GISP
                # [78.5270, -56.8370] # Humboldt
                # [73.8402, -49.4983] # NASA-U
                # [78.0167, -33.9667] # Tunu-N
               #[66.1812, -39.0435] # FA-13
               ]
all_points = df_meta[['latitude', 'longitude']].values
df_meta['distance_from_query_point'] = distance.cdist(all_points, query_point, get_distance)
min_dist = 10 # in km
df_meta_selec = df_meta.loc[df_meta.distance_from_query_point<min_dist, :]

#  plotting coordinates
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
from tqdm import tqdm

plt.figure()

for count, ref in enumerate(df_meta_selec.reference_short.unique()):
    # each reference will be plotted in a different color
    label = ref
    list_names=df_meta_selec.loc[df_meta_selec.reference_short == ref, 'name'].unique()
    if len(list_names)>100:
        # to speed up plotting, we only show 100 measurements
        list_names = list_names[np.linspace(0, len(list_names) - 1, 100, dtype=int)]
    for n in tqdm(list_names, desc=ref):
        # for each core or each point along a radar transect, a separate line needs
        # to be plotted
        df_stack = (df_sumup.loc[df_sumup.name==n,['start_year','end_year']]
                    .sort_values(by='start_year')
                    .stack().reset_index().drop(columns='level_1')
                    .rename(columns={0:'year'}))
        df_stack['accumulation'] = df_sumup.loc[df_stack.measurement_id, 'accumulation'].values
        for meas_id in np.unique(df_stack.measurement_id):
            df_meas=df_stack.loc[df_stack.measurement_id==meas_id,:]
            if df_meas.iloc[1,1]==df_meas.year.iloc[0]:
                df_meas.iloc[1,1]=df_meas.year.iloc[0]+1
            df_meas.plot(
                ax=plt.gca(), x='year', y='accumulation',
                          color = cmap(count),
                          label=label if meas_id==df_stack.measurement_id[0] else '_nolegend_',
                          alpha=0.7, legend=False
                          )
        plt.legend(loc='upper left')
        plt.ylabel('accumulation (m w.e. yr-1)')
        label='_nolegend_'
plt.title('Observations within '+str(min_dist)+' km of '+str(query_point))

# %% selecting a given method
# displaying available methods
df_methods = ds_meta.method.to_dataframe()
print(df_methods.to_markdown())

# selecting the method keys containing 'pit' or 'core'
keys_cores_pits = df_methods.loc[
    df_methods.method.str.contains('pit') | df_methods.method.str.contains('core')].index.values

# selecting the observations that have method_key in keys_cores_pits
df_selec = df_sumup.loc[df_sumup.method_key.isin(keys_cores_pits)]

# adding spelled-out versions of keys to df_selec
df_selec['method'] = ds_meta.method.sel(method_key = df_selec.method_key.values).astype(str)
df_selec['name'] = ds_meta.name.sel(name_key = df_selec.name_key.values).astype(str)
df_selec['reference'] = (ds_meta.reference
                         .sel(reference_key=df_selec.reference_key.values)
                         .astype(str))
df_selec['reference_short'] = (ds_meta.reference_short
                         .sel(reference_key=df_selec.reference_key.values)
                         .astype(str))
print('Table contains the following columns')
print(df_selec.columns.values)

print('\n using the following methods')
print(df_selec.method.drop_duplicates().values)

print('\n and contains the following profiles')
print(df_selec.name.drop_duplicates().values)

print('\n from the following references')
print(df_selec.reference_short.drop_duplicates().values)
