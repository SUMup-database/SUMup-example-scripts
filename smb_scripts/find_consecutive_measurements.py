# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: bav@geus.dk

This script identifies and processes continuous SMB (Surface Mass Balance) measurements,
often used for monitoring at weather stations or other observational sites. It aims to
aggregate and extract relevant information about consecutive SMB measurements for analysis.

The primary goals of this script are:
1. To identify groups of measurements from the same reference, grouped under the same name.
2. To check for temporal continuity within each group by comparing end_date with the subsequent start_date.
3. To extract information about consecutive groups, including:
   - Overall start and end dates.
   - The number of measurements in each consecutive group.
   - The average SMB within each group.
4. To provide the ability to filter and analyze SMB data for larger datasets, ensuring flexibility and efficiency.

The workflow includes:
- Loading the SMB dataset and associated metadata.
- Ensuring proper decoding of string fields (e.g., name, reference) for consistency.
- Filtering and identifying unique groups based on reference, reference_short, and name.
- For each group, identifying consecutive measurements based on matching end_date and start_date.
- Storing and optionally printing summary information about consecutive groups.

Input:
- `SUMup_2024_SMB_greenland.nc`: NetCDF file containing SMB data and metadata.

Output:
- Summary information about identified consecutive groups, including start and end dates, number of measurements, and average SMB.

Dependencies:
- numpy
- xarray

Notes:
- The script is optimized to handle large datasets efficiently, leveraging pandas and xarray functionalities.
- Only groups with at least 5 measurements are considered for consecutive group analysis.

"""

import numpy as np
import xarray as xr

path_to_sumup = 'C:/Users/bav/GitHub/SUMup/SUMup-2024/SUMup 2024 beta/'
df_sumup = xr.open_dataset(path_to_sumup+'/SUMup_2024_SMB_greenland.nc',
                           group='DATA').to_dataframe()
ds_meta = xr.open_dataset(path_to_sumup+'/SUMup_2024_SMB_greenland.nc',
                           group='METADATA')
decode_utf8 = np.vectorize(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
for v in ['name','reference','reference_short','method']:
    ds_meta[v] = xr.DataArray(decode_utf8(ds_meta[v].values), dims=ds_meta[v].dims)


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


# Get unique combinations of reference, reference_short, and name
unique_groups = df_sumup[['reference', 'reference_short', 'name']].drop_duplicates()


# Process consecutive groups
results = []
for _, group_keys in unique_groups.iterrows():
    print_info = True
    group = df_sumup[
        (df_sumup['reference'] == group_keys['reference']) &
        (df_sumup['reference_short'] == group_keys['reference_short']) &
        (df_sumup['name'] == group_keys['name'])
    ].sort_values('start_date').reset_index(drop=True)
    if len(group)<5:
        # print('less than 5 values')
        continue

    group['consecutive_group'] = (group['start_date'] != group['end_date'].shift()).cumsum()

    for consecutive_id, consecutive_group in group.groupby('consecutive_group'):
        if len(consecutive_group)>1:
            if print_info:
                print(group_keys[['name','reference_short']].values)
                print('    ','first_start_date\tlast_end_date\tnum_measurements\tmean_smb')
            print_info=False

            consecutive_info = {
                'overall_group': f"{group_keys['reference']} - {group_keys['name']}",
                'consecutive_group_id': consecutive_id,
                'overall_start_date': consecutive_group['start_date'].min().strftime('%Y-%m-%d'),
                'overall_end_date': consecutive_group['end_date'].max().strftime('%Y-%m-%d'),
                'number_of_measurements': len(consecutive_group),
                'average_smb': consecutive_group['smb'].mean()
            }
            print('   ', '\t'.join([str(consecutive_info[v]) for v in [
                'overall_start_date','overall_end_date',
                 'number_of_measurements','average_smb']]))
            results.append(consecutive_info)


# %% if you want to save the info
# Convert results to DataFrame
# consecutive_info = pd.DataFrame(results)

# Display results
# import ace_tools as tools; tools.display_dataframe_to_user(name="Optimized Consecutive Group Analysis with Drop Duplicates", dataframe=consecutive_info)
