# GIRAFE Firecube Plugin

Ingest [GIRAFE](https://doi.org/10.5676/EUM_SAF_CM/GIRAFE/V001) netCDF products into
a zarr store using [firecube](https://eumetsat.github.io/firecube/latest/). The plugin
provides two chunk layouts optimized for timeseries (Spaghetti) and
maps (Lasagna).

## Installation

```
pip install firecube-girafe
```

## Usage

Ingest using layout optimized for maps

```bash
firecube ingest girafe \
   --input-data /input/girafe \
   --target file:///output/girafe-maps.zarr \
   --product-name girafe \
   --write-mode direct \
   --option layout=maps
```

Ingest using layout optimized for timeseries

```bash
firecube ingest girafe \
   --input-data /input/girafe \
   --target file:///output/girafe-timeseries.zarr \
   --product-name girafe \
   --write-mode direct \
   --option layout=timeseries
```

You can also use a custom chunk layout

```bash
firecube ingest girafe \
   --input-data /input/girafe \
   --target file:///output/girafe-timeseries.zarr \
   --product-name girafe \
   --write-mode direct \
   --option zarr_chunk_shape='{"time": 10, "lat": 10, "lon": 10}'
```

After ingestion, compute something

```python
import xarray as xr

with xr.open_zarr(
    "/output/girafe-timeseries.zarr", group="default", consolidated=False
) as ds:
    res = ds.sel(lat=0, lon=0, method="nearest").mean("time")
    res["precipitation"].plot()
```
