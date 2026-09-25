"""Generic Zarr ingestor for GIRAFE."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, ClassVar

import xarray as xr
from firecube.ingestor.api import (
    GenericZarrIngestor,
    PluginConfig,
    PluginContext,
    register_ingestor,
)


@dataclasses.dataclass
class GirafePluginConfig(PluginConfig):
    layout: str = "maps"


@register_ingestor("girafe")
class GirafeIngestor(GenericZarrIngestor):
    PRODUCT_NAME: ClassVar[str] = "girafe"
    time_dim_name: ClassVar[str] = "time"
    plugin_config_class = GirafePluginConfig
    time_dependent_attrs = (
        "date_created",
        "time_coverage_start",
        "time_coverage_end",
        "platform",
        "instrument",
    )
    layouts: ClassVar[dict] = {
        "maps": {"chunk_shape": {"time": 1, "lat": 180, "lon": 360}},
        "timeseries": {"chunk_shape": {"time": 300, "lat": 15, "lon": 15}},
    }

    def get_zarr_config(self, ctx):
        """Get zarr config.

        Start from the operator's options and fill only what they did not set.
        """
        zarr_config = super().get_zarr_config(ctx)
        layout = self.layouts[self.plugin_config.layout]
        for key, value in layout.items():
            if f"zarr_{key}" not in ctx.options:
                zarr_config[key] = value
        return zarr_config

    def build_dataset(
        self,
        group: str,
        items: list[Any],
        ctx: PluginContext,
    ) -> xr.Dataset | None:
        """Return one batch of source files as a single dataset, or ``None`` to skip it.

        Firecube calls this once per batch, in input order, and appends the
        returned dataset to the store before asking for the next batch. ``items``
        holds up to ``pipeline_batch_size`` files (10 by default). If this raises,
        the run stops at this batch; batches appended before it stay written.
        The dataset must be sorted along the time dimension with no repeated timestamps.
        """
        _ = group
        if not items:
            return None
        datasets = [self._read_dataset(ctx.materialize(item)) for item in items]
        return self._concatenate(datasets)

    def _read_dataset(self, path: Path) -> xr.Dataset:
        """Read one local data file."""
        with xr.open_dataset(path) as dataset:
            self._drop_time_dependent_attrs(dataset)
            # FIXME: Try without loading and use dask
            return dataset.load()

    def _drop_time_dependent_attrs(self, dataset: xr.Dataset) -> None:
        for key in self.time_dependent_attrs:
            dataset.attrs.pop(key)

    def _concatenate(self, datasets: list[xr.Dataset]) -> xr.Dataset:
        res = xr.concat(
            datasets,
            dim=self.time_dim_name,
            # variables without time dimension are stored once, not copied per step
            data_vars="minimal",
            # same for coordinates like bounds that get decoded as coords
            coords="minimal",
            # static values must match across files in one batch
            compat="equals",
            # grids must match
            join="exact",
        )
        return res.sortby(self.time_dim_name)
