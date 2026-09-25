"""Tests for GIRAFE Ingestor"""

import datetime as dt
from importlib.metadata import entry_points

import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner
from firecube.cli.main import cli
from firecube.ingestor.api import discover_ingestors

from firecube_girafe import GirafeIngestor


def test_entry_point_registers_the_ingestor() -> None:
    found = entry_points(group="firecube.plugins", name="girafe")
    assert found, "pyproject.toml declares no firecube.plugins entry point named girafe"
    for entry_point in found:
        entry_point.load()
    assert discover_ingestors()["girafe"] is GirafeIngestor
    assert GirafeIngestor.PRODUCT_NAME == "girafe"


def get_fake_dataset(date: dt.date) -> xr.Dataset:
    timestamp = dt.datetime.combine(date, dt.time(0))
    fake_data = date.day * np.ones((1, 2, 2))
    precip = xr.DataArray(
        fake_data,
        dims=("time", "lat", "lon"),
        attrs={
            "units": "mm",
            "standard_name": "lwe_thickness_of_precipitation_amount",
            "cell_method": "time: area: sum",
            "long_name": "daily accumulated precipitation",
        },
    )
    sampling_error = xr.DataArray(
        fake_data,
        dims=("time", "lat", "lon"),
        attrs={
            "units": "mm",
            "long_name": "sampling error of daily accumulated precipitation",
            "standard_name": "lwe_thickness_of_precipitation_amount standard_error",
        },
    )
    sampling_error_qf = xr.DataArray(
        fake_data,
        dims=("time", "lat", "lon"),
        attrs={
            "long_name": "Sampling error quality flag",
            "units": "1",
            "flag_values": np.array([0, 1, 2, 3], dtype=np.int8),
            "flag_meanings": (
                "ok "
                "default_value_for_decorrelation_scale_in_time "
                "default_value_for_decorrelation_scale_in_space "
                "default_value_for_decorrelation_scale_in_space_and_time"
            ),
        },
    )
    num_obs_fraction = xr.DataArray(
        fake_data,
        dims=("time", "lat", "lon"),
        attrs={
            "units": "1",
            "long_name": "Number of observations available for derivation of precip fraction",
            "standard_name": "number_of_observations",
        },
    )
    num_obs_rate = xr.DataArray(
        fake_data,
        dims=("time", "lat", "lon"),
        attrs={
            "units": "1",
            "long_name": "Number of observations available for derivation of precip rate",
            "standard_name": "number_of_observations",
        },
    )
    snow_flag = xr.DataArray(
        fake_data,
        dims=("time", "lat", "lon"),
        attrs={
            "long_name": "Surface snow and sea ice related quality flag",
            "units": "1",
            "flag_values": np.array([0, 1], dtype=np.int8),
            "flag_meanings": "ok bad_quality",
        },
    )
    time = xr.DataArray(
        [timestamp],
        dims="time",
        attrs={
            "long_name": "Product dataset time given as seconds since 2000-01-01T00:00:00",
            "standard_name": "time",
            "axis": "T",
            "bounds": "time_bnds",
        },
    )
    lat = xr.DataArray(
        [0.0, 1.0],
        dims="lat",
        attrs={
            "units": "degrees_north",
            "long_name": "Latitude",
            "standard_name": "latitude",
            "valid_range": np.array([-90.0, 90.0]),
            "reference_datum": "geographical coordinates, WGS84 projection",
            "axis": "Y",
            "bounds": "lat_bnds",
        },
    )
    lon = xr.DataArray(
        [0.0, 1.0],
        dims="lon",
        attrs={
            "units": "degrees_east",
            "long_name": "Longitude",
            "standard_name": "longitude",
            "valid_range": np.array([-180.0, 180.0]),
            "reference_datum": "geographical coordinates, WGS84 projection",
            "axis": "X",
            "bounds": "lon_bnds",
        },
    )
    time_bnds = xr.DataArray(
        [[timestamp, timestamp + dt.timedelta(days=1)]],
        dims=("time", "nv"),
        attrs={
            "long_name": "Time cell boundaries",
            "comment": (
                "Contains the start and end times for the time period "
                "that the data represent."
            ),
        },
    )
    lat_bnds = xr.DataArray(
        [[-0.5, 0.5], [0.5, 1.5]],
        dims=("lat", "nv"),
        attrs={
            "long_name": "Latitude cell boundaries",
            "valid_range": np.array([-90.0, 90.0]),
            "reference_datum": "geographical coordinates, WGS84 projection",
            "comment": "Contains the northern and southern boundaries of the grid cells.",
        },
    )
    lon_bnds = xr.DataArray(
        [[-0.5, 0.5], [0.5, 1.5]],
        dims=("lon", "nv"),
        attrs={
            "long_name": "Longitude cell boundaries",
            "valid_range": np.array([-180.0, 180.0]),
            "reference_datum": "geographical coordinates, WGS84 projection",
            "comment": "Contains the eastern and western boundaries of the grid cells.",
        },
    )
    rec_status = xr.DataArray(
        [0],
        dims="time",
        attrs={
            "long_name": "Record Status",
            "comment": (
                "Overall status of each record (timestamp) in this file. "
                "If a record is flagged as not ok, it is recommended not to use it."
            ),
            "flag_values": np.array([0, 1, 2], dtype=np.int8),
            "flag_meanings": "ok void bad_quality",
        },
    )
    time_fmt = "%Y-%m-%dT%H:%M:%SZ"
    time_cov_start = time_bnds.dt.strftime(time_fmt).isel(time=0, nv=0)
    time_cov_end = time_bnds.dt.strftime(time_fmt).isel(time=0, nv=1)
    # Platform and instrument may change over time
    attrs = {
        "date_created": dt.datetime.now(dt.UTC).strftime(time_fmt),
        "instrument": f"instrument-{date.day}",
        "platform": f"platform-{date.day}",
        "time_coverage_start": time_cov_start.item(),
        "time_coverage_end": time_cov_end.item(),
        "title": "Global Interpolated RAinFall Estimation (GIRAFE)",
    }
    return xr.Dataset(
        {
            "precipitation": precip,
            "sampling_error": sampling_error,
            "sampling_error_quality_flag": sampling_error_qf,
            "snow_flag": snow_flag,
            "num_obs_fraction": num_obs_fraction,
            "num_obs_rate": num_obs_rate,
            "record_status": rec_status,
            "time_bnds": time_bnds,
            "lat_bnds": lat_bnds,
            "lon_bnds": lon_bnds,
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs=attrs,
    )


@pytest.fixture
def dates():
    return [
        dt.date(2026, 1, 1),
        dt.date(2026, 1, 2),
        dt.date(2026, 1, 3),
        dt.date(2026, 1, 4),
        dt.date(2026, 1, 5),
    ]


@pytest.fixture
def encoding():
    return {
        "num_obs_fraction": {
            "zlib": True,
            "_FillValue": -99,
            "dtype": np.int32,
        },
        "num_obs_rate": {
            "zlib": True,
            "_FillValue": -99,
            "dtype": np.int32,
        },
        "precipitation": {
            "zlib": True,
            "_FillValue": -99,
            "dtype": np.float32,
        },
        "sampling_error": {"zlib": True, "_FillValue": -99},
        "sampling_error_quality_flag": {
            "dtype": np.int8,
            "zlib": True,
            "_FillValue": np.int8(-1),
        },
        "snow_flag": {
            "dtype": np.int8,
            "zlib": True,
            "_FillValue": np.int8(-1),
        },
        "time": {"dtype": np.int32, "units": "Seconds since 2000-01-01"},
        "time_bnds": {"dtype": np.int32},
        "lat_bnds": {"_FillValue": None, "dtype": np.float64},
        "lon_bnds": {"_FillValue": None, "dtype": np.float64},
        "lat": {"_FillValue": None, "dtype": np.float64},
        "lon": {"_FillValue": None, "dtype": np.float64},
    }


@pytest.fixture
def source_files(tmp_path, dates, encoding):
    filenames = []
    for date in dates:
        filename = tmp_path / f"PREdm{date:%Y%m%d}000000120IMPGS01GL.nc"
        ds = get_fake_dataset(date)
        ds.to_netcdf(filename, engine="netcdf4", encoding=encoding)
        filenames.append(filename)
    return filenames


@pytest.fixture
def target_file(tmp_path):
    return tmp_path / "girafe.zarr"


@pytest.fixture
def dataset_exp(dates):
    datasets = [get_fake_dataset(date) for date in dates]
    ds = xr.concat(datasets, dim="time", coords="minimal", data_vars="minimal")
    for key in (
        "date_created",
        "time_coverage_start",
        "time_coverage_end",
        "platform",
        "instrument",
    ):
        ds.attrs.pop(key)
    ds["firecube_timestamp_state"] = xr.DataArray(
        [1, 1, 1, 1, 1],
        dims="time",
        attrs={
            "firecube_meaning": {
                "0": "unknown",
                "1": "present",
                "2": "deleted_by_firecube",
                "3": "failed_batch",
            }
        },
    )
    return ds


@pytest.mark.usefixtures("source_files")
def test_ingest(tmp_path, target_file, dataset_exp):
    runner = CliRunner()
    cmd = [
        "ingest",
        "girafe",
        "--product-name",
        "girafe",
        "--input-data",
        str(tmp_path),
        "--target",
        f"file:///{target_file}",
        "--write-mode",
        "direct",
        "--option",
        "layout=maps",
    ]
    res = runner.invoke(cli, cmd, catch_exceptions=False)
    assert res.exit_code == 0
    with xr.open_zarr(target_file, group="default", consolidated=False) as ds:
        xr.testing.assert_identical(ds, dataset_exp)
