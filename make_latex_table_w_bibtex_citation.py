# -*- coding: utf-8 -*-
"""
Create and print a LaTeX table summarizing Antarctic SMB observations.

What this script does:
- Loads SUMup SMB data and metadata from NetCDF
- Joins human-readable metadata (method, name, bibtex_key)
- Selects Antarctic observations (latitude < 0)
- Groups by method and bibtex reference
- Computes covered period and number of observations
- Prints and saves a LaTeX longtable using \\citet{}
"""

import xarray as xr
from pathlib import Path

path_to_sumup = "../2025/"
region = "greenland"

ncfile = Path(path_to_sumup) / f"SUMup_2025_SMB_{region}.nc"

df_sumup = xr.open_dataset(ncfile, group="DATA").to_dataframe()

ds_meta = xr.open_dataset(ncfile, group="METADATA")

for v in ["name", "reference", "reference_short", "method", "bibtex_key"]:
    ds_meta[v] = ds_meta[v].str.decode("utf-8")

df_sumup["method"] = ds_meta.method.sel(
    method_key=df_sumup.method_key.values
)
df_sumup["name"] = ds_meta.name.sel(name_key=df_sumup.name_key.values)
df_sumup["bibtex_key"] = ds_meta.bibtex_key.sel(
    reference_key=df_sumup.reference_key.values
)

g = df_sumup.groupby(["method", "bibtex_key"])

df_out = (
    g.agg(
        start_min=("start_year", "min"),
        end_max=("end_year", "max"),
        n_rows=("start_year", "size"),
    )
    .reset_index()
)

latex_table = (
    df_out.assign(
        Period=lambda d: d.start_min.astype(int).astype(str)
        + "--"
        + d.end_max.astype(int).astype(str),
        Reference=lambda d: r"\citet{" + d.bibtex_key.astype(str).str.replace(" ", ",") + "}",
    )
    [["Reference", "Period", "n_rows", "method"]]
    .rename(
        columns={
            "method": "Method",
            "n_rows": "Number of observations",
        }
    )
    .to_latex(
        index=False,
        escape=False,
        longtable=True,
        column_format="p{5cm}|p{3cm}|r|p{4cm}",
    )
)

print(latex_table)

out_file = Path(f"sumup_smb_{region}_table.tex")
out_file.write_text(latex_table, encoding="utf-8")
