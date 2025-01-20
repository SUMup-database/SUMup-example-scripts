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
from tqdm import tqdm

path_to_SUMup_folder = 'C:/Users/bav/GitHub/SUMup/SUMup-2024/'
df_sumup = xr.open_dataset(path_to_SUMup_folder+'SUMup 2024 beta/SUMup_2024_temperature_greenland.nc',
                           group='DATA').to_dataframe()
ds_meta = xr.open_dataset(path_to_SUMup_folder+'SUMup 2024 beta/SUMup_2024_temperature_greenland.nc',
                           group='METADATA')

# decoding strings as utf-8
for v in ['name','reference','reference_short','method']:
    ds_meta[v] = ds_meta[v].str.decode('utf-8')

# df_sumup.method_key = df_sumup.method_key.replace(np.nan,-9999)
df_sumup['method'] = ds_meta.method.sel(method_key = df_sumup.method_key.values)
df_sumup['name'] = ds_meta.name.sel(name_key = df_sumup.name_key.values)


df_sumup['reference'] = None  # Initialize the column
for key in tqdm(ds_meta.reference['reference_key'].values):
    df_sumup.loc[df_sumup['reference_key'] == key, 'reference'] = ds_meta.reference.sel(reference_key=key).item()

df_sumup['reference'] = (ds_meta.reference
                          .drop_duplicates(dim='reference_key')
                          .sel(reference_key=df_sumup.reference_key.values))
df_sumup['reference_short'] = (ds_meta.reference_short
                         .drop_duplicates(dim='reference_key')
                         .sel(reference_key=df_sumup.reference_key.values))
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

query_point = [
                [77.1667, -61.1333], # Camp Century
                # [66.4823, -46.2908], # DYE-2
               ]
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


# %% Plotting at time slices
plt.figure()
for date in ['2017-09-01','2024-09-01']:
    df_sumup_selec.loc[
        df_sumup_selec.timestamp == (pd.to_datetime(date)),
            :].plot(x='temperature',y='depth', marker='o', ls='None',
                    label=date, ax=plt.gca())
    plt.legend()
    plt.ylim(70,0)
    plt.grid()

df_sumup_selec.loc[
    df_sumup_selec.timestamp.isin(pd.to_datetime(['2017-09-01','2024-09-01'])),
        ['timestamp','depth','temperature']].to_csv('CC_temperature_2017_2024.csv',index=False)

# %% Interpoalting temperature at fixed depth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Assuming df_sumup is your DataFrame

# Set the depth for interpolation
target_depth = 20
min_depth_pairs = 2  # Minimum number of depth-temperature pairs required for interpolation

# Create a new DataFrame for interpolated temperatures
df_interpolated = pd.DataFrame()

# Interpolate temperature at 10 m depth for each name_key and timestamp
for n in df_meta_selec['name_key'].unique():
    df_subset = df_sumup[df_sumup['name_key'] == n]
    print(df_subset.name.drop_duplicates().values[0])

    # Check if there are enough depth-temperature pairs
    if df_subset.shape[0] < min_depth_pairs:
        if target_depth in df_subset.depth.values:
            temp_at_target_depth = df_subset.loc[df_subset.depth==target_depth,:]
            df_interpolated = pd.concat([df_interpolated, temp_at_target_depth], ignore_index=True)
        else:
            print(f"Target depth not in {df_subset.name.drop_duplicates().values[0]} and only {df_subset.shape[0]} values to interpolate from. Skipping.")
            continue


    # Set the index to a MultiIndex of timestamp and depth
    df_subset = df_subset.set_index(['timestamp', 'depth'])

    # Create a range of timestamps for interpolation
    timestamps = df_subset.index.levels[0]

    if df_subset.index.get_level_values(1).max()<(target_depth-2):
        continue
    # Interpolate for each timestamp
    for timestamp in timestamps:
        df_time_subset = df_subset.xs(timestamp, level='timestamp')[['temperature']]
        if df_time_subset.shape[0] < min_depth_pairs:
            continue        # Check if we can interpolate

        if df_time_subset.index[0] == target_depth:
            df_time_subset = df_time_subset.iloc[1:]

        # Add target depth if not present
        if target_depth not in df_time_subset.index.values:
            new_row = pd.Series({'temperature': np.nan}, name=target_depth)
            df_time_subset = pd.concat([df_time_subset, new_row.to_frame().T])  # Add new row

        # Interpolating temperature at 10 m depth
        interp_temp = df_time_subset['temperature'].interpolate(method='index') #.interpolate(method='index')

        # Check if the interpolated temperature exists for the target depth
        if target_depth in interp_temp.index:
            temp_at_target_depth = pd.DataFrame()
            temp_at_target_depth['depth'] = [target_depth]
            temp_at_target_depth['temperature'] = np.unique(interp_temp.loc[target_depth])
            temp_at_target_depth['name_key'] = n
            temp_at_target_depth['name'] = df_subset.name.drop_duplicates().values[0]
            temp_at_target_depth['timestamp'] = timestamp

            df_interpolated = pd.concat([df_interpolated, temp_at_target_depth], ignore_index=True)
        else:
            print(f"Temperature at {target_depth} m not found for {n} at {timestamp}.")

# %% Plotting interpolated temperature
plt.figure()
cmap = plt.get_cmap("tab10")  # Change to your desired colormap

for count, n in enumerate(df_interpolated['name_key'].unique()):
    df_temp = df_interpolated[df_interpolated['name_key'] == n]

    # Calculate the rolling mean
    if len(df_temp['temperature'])>7:
        df_temp['temperature'] = df_temp['temperature'].rolling(7).mean()

    # Remove NaN values introduced by rolling mean
    df_temp = df_temp.dropna(subset=['temperature'])

    # Plot the rolling mean
    plt.plot(df_temp['timestamp'], df_temp['temperature'],
             color=cmap(count), marker='o', linestyle='None',
             label=df_temp.name.iloc[0], alpha=0.7)

    # Fit a linear regression
    if (df_temp['timestamp'].max() - df_temp['timestamp'].min())>pd.to_timedelta('1000 days'):
        x = pd.to_numeric(df_temp['timestamp'])  # Convert timestamps to numeric values for regression
        y = df_temp['temperature']
        slope, intercept = np.polyfit(x, y, 1)  # Linear regression

        # Convert slope to °C per decade
        slope_per_decade = slope * (10 * 365.25 * 24 * 60 * 60 * 1e9)  # Convert nanoseconds to decades

        # Plot the regression line
        plt.plot(df_temp['timestamp'], intercept + slope * x,
                 color=cmap(count), linestyle='-', alpha=0.5,
                 label= f"{slope_per_decade:.2f} °C/decade")

if count>4:
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),ncols=2, title='Sources:')
else:
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), title='Sources:')
plt.ylabel(f'Temperature at {target_depth} m (°C)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout
plt.show()
