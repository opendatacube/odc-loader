"""
Tests for the in-memory reader driver
"""

from __future__ import annotations

import json
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pytest
import xarray as xr
from dask import is_dask_collection
from dask.base import tokenize
from odc.geo.data import country_geom
from odc.geo.gcp import GCPGeoBox
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.xr import ODCExtensionDa, ODCExtensionDs, rasterize
from packaging.version import parse as parse_version

from odc.loader import chunked_load
from odc.loader._zarr import (
    Context,
    XrMemReader,
    XrMemReaderDask,
    XrMemReaderDriver,
    extract_zarr_spec,
    raster_group_md,
)
from odc.loader.types import (
    AuxDataSource,
    AuxLoadParams,
    FixedCoord,
    RasterBandMetadata,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterSource,
)

# pylint: disable=missing-function-docstring,use-implicit-booleaness-not-comparison,protected-access
# pylint: disable=too-many-locals,too-many-statements,redefined-outer-name,import-outside-toplevel


@pytest.fixture
def sample_ds() -> xr.Dataset:
    poly = country_geom("AUS", 3857)
    gbox = GeoBox.from_geopolygon(poly, resolution=10_000)
    xx = rasterize(poly, gbox).astype("int16")
    xx.attrs["units"] = "uu"
    xx.attrs["nodata"] = -33

    return xx.to_dataset(name="xx")


@pytest.fixture
def sample_ds_with_aux(sample_ds: xr.Dataset) -> xr.Dataset:
    ds = sample_ds.copy()
    ds["aux"] = xr.DataArray([1, 2, 3], dims=("index",), coords={"index": [0, 1, 2]})
    return ds


def test_mem_reader(sample_ds: xr.Dataset) -> None:
    fake_item = object()

    assert isinstance(sample_ds.odc, ODCExtensionDs)
    gbox = sample_ds.odc.geobox
    assert gbox is not None
    assert isinstance(gbox, GeoBox)

    driver = XrMemReaderDriver(sample_ds)

    assert driver.md_parser is not None

    md = driver.md_parser.extract(fake_item)
    assert isinstance(md, RasterGroupMetadata)
    assert len(md.bands) == 1
    assert ("xx", 1) in md.bands
    assert md.bands[("xx", 1)].data_type == "int16"
    assert md.bands[("xx", 1)].units == "uu"
    assert md.bands[("xx", 1)].nodata == -33
    assert md.bands[("xx", 1)].dims == ()
    assert len(md.aliases) == 0
    assert md.extra_dims == {}
    assert md.extra_coords == []

    ds = sample_ds.copy()
    xx = ds.xx
    yy = xx.astype("uint8", keep_attrs=False).rename("yy")
    yy = yy.expand_dims("band", 2)
    yy = xr.concat([yy, yy + 1, yy + 2], "band").assign_coords(band=["r", "g", "b"])
    yy.band.attrs["units"] = "CC"

    assert yy.odc.geobox == gbox

    ds["yy"] = yy
    ds["zz"] = yy.transpose("band", "y", "x")
    ds["aux"] = xr.DataArray([1, 2, 3], dims=("index",), coords={"index": [0, 1, 2]})

    driver = XrMemReaderDriver(ds)
    assert driver.md_parser is not None
    assert driver.dask_reader is not None
    assert driver.aux_reader is not None
    md = driver.md_parser.extract(fake_item)

    assert isinstance(md, RasterGroupMetadata)
    assert len(md.bands) == 4
    assert ("xx", 1) in md.bands
    assert ("yy", 1) in md.bands
    assert ("zz", 1) in md.bands
    assert ("aux", 1) in md.bands
    assert md.bands[("xx", 1)].data_type == "int16"
    assert md.bands[("xx", 1)].units == "uu"
    assert md.bands[("xx", 1)].nodata == -33
    assert md.bands[("xx", 1)].dims == ()
    assert md.bands[("yy", 1)].data_type == "uint8"
    assert md.bands[("yy", 1)].units == "1"
    assert md.bands[("yy", 1)].nodata is None
    assert md.bands[("yy", 1)].dims == ("y", "x", "band")
    assert md.bands[("zz", 1)].dims == ("band", "y", "x")
    assert md.bands[("aux", 1)].data_type == "int64"
    assert md.bands[("aux", 1)].units == "1"
    assert md.bands[("aux", 1)].nodata is None
    assert md.bands[("aux", 1)].dims == ()

    assert len(md.aliases) == 0
    assert md.extra_dims == {"band": 3}
    assert len(md.extra_coords) == 1

    (coord,) = md.extra_coords
    assert coord.name == "band"
    assert coord.units == "CC"
    assert coord.dim == "band"
    assert coord.values == ["r", "g", "b"]

    oo: ODCExtensionDa = ds.yy.odc
    assert isinstance(oo.geobox, GeoBox)

    env = driver.capture_env()
    ctx = driver.new_load(oo.geobox)
    assert isinstance(env, dict)

    def mk_src(n: str) -> RasterSource | AuxDataSource:
        meta = md.bands[n, 1]
        if isinstance(meta, RasterBandMetadata):
            return RasterSource(
                f"mem://{n}",
                meta=meta,
                driver_data=driver.md_parser.driver_data(fake_item, (n, 1)),
            )
        return AuxDataSource(
            f"mem://{n}",
            meta=meta,
            driver_data=driver.md_parser.driver_data(fake_item, (n, 1)),
        )

    srcs = {n: mk_src(n) for n, _ in md.bands}
    cfgs = {
        n: RasterLoadParams.same_as(src)
        for n, src in srcs.items()
        if isinstance(src, RasterSource)
    }

    with driver.restore_env(env, ctx) as _ctx:
        assert _ctx is not None

        loaders = {
            n: driver.open(src, ctx)
            for n, src in srcs.items()
            if isinstance(src, RasterSource)
        }
        assert set(loaders) == (set(srcs) - {"aux"})

        for n, loader in loaders.items():
            assert isinstance(loader, XrMemReader)
            roi, pix = loader.read(cfgs[n], gbox)
            assert roi == (slice(None), slice(None))
            assert isinstance(pix, np.ndarray)
            if n == "xx":
                assert pix.dtype == np.int16
                assert pix.shape == gbox.shape.yx
            elif n == "yy":
                assert pix.dtype == np.uint8
                assert pix.shape == (*gbox.shape.yx, 3)
            elif n == "zz":
                assert pix.shape == (3, *gbox.shape.yx)

        loader = loaders["yy"]
        roi, pix = loader.read(cfgs["yy"], gbox, selection=np.s_[:2])
        assert pix.shape == (*gbox.shape.yx, 2)

        loader = loaders["zz"]
        roi, pix = loader.read(cfgs["zz"], gbox, selection=np.s_[:2])
        assert pix.shape == (2, *gbox.shape.yx)


def test_raster_group_md() -> None:
    rgm = raster_group_md(xr.Dataset())
    assert rgm.bands == {}
    assert rgm.aliases == {}
    assert rgm.extra_dims == {}

    coord = FixedCoord("band", ["r", "g", "b"], dim="band")

    rgm = raster_group_md(
        xr.Dataset(), base=RasterGroupMetadata({}, {}, {"band": 3}, [])
    )
    assert rgm.extra_dims == {"band": 3}
    assert len(rgm.extra_coords) == 0

    rgm = raster_group_md(
        xr.Dataset(), base=RasterGroupMetadata({}, extra_coords=[coord])
    )
    assert rgm.extra_dims == {}
    assert rgm.extra_dims_full() == {"band": 3}
    assert len(rgm.extra_coords) == 1
    assert rgm.extra_coords[0] == coord


def test_memreader_zarr(sample_ds: xr.Dataset) -> None:
    assert isinstance(sample_ds.odc, ODCExtensionDs)
    assert "xx" in sample_ds

    zarr = pytest.importorskip("zarr")
    assert zarr is not None
    # FIXME: update test for Zarr v3, links to more information in:
    # https://github.com/opendatacube/odc-loader/pull/29#issuecomment-3258611734
    if parse_version(version("zarr")) > parse_version("2"):
        pytest.skip("Consolidated metadata is located elsewhere for Zarr v3")
    _gbox = sample_ds.odc.geobox
    assert _gbox is not None
    gbox = _gbox.approx if isinstance(_gbox, GCPGeoBox) else _gbox

    md_store: dict[str, bytes] = {}
    chunk_store: dict[str, bytes] = {}
    sample_ds.to_zarr(md_store, chunk_store, compute=False, consolidated=True)

    assert ".zmetadata" in md_store
    zmd = json.loads(md_store[".zmetadata"])["metadata"]

    src = RasterSource(
        "file:///tmp/no-such-dir/xx.zarr",
        subdataset="xx",
        driver_data=zmd,
    )
    assert src.driver_data is zmd
    cfg = RasterLoadParams.same_as(src)

    driver = XrMemReaderDriver()
    ctx = driver.new_load(gbox, chunks=None)
    rdr = driver.open(src, ctx)

    roi, xx = rdr.read(cfg, gbox)
    assert isinstance(xx, np.ndarray)
    assert xx.shape == gbox[roi].shape.yx
    assert gbox == gbox[roi]

    assert driver.dask_reader is not None

    tk = tokenize(src, cfg, gbox)

    ctx = driver.new_load(gbox, chunks={})
    assert isinstance(ctx, Context)

    rdr = driver.dask_reader.open(src, cfg, ctx, layer_name=f"xx-{tk}", idx=0)
    assert isinstance(rdr, XrMemReaderDask)
    assert rdr._xx is not None
    assert is_dask_collection(rdr._xx)

    fut = rdr.read(gbox)
    assert is_dask_collection(fut)

    roi, xx = fut.compute(scheduler="synchronous")
    assert isinstance(xx, np.ndarray)
    assert roi == (slice(None), slice(None))
    assert xx.shape == gbox.shape.yx


@pytest.mark.parametrize("chunks", [None, {"time": 1}])
def test_memreader_aux(
    sample_ds_with_aux: xr.Dataset,
    chunks: dict[str, int] | None,
) -> None:
    ds = sample_ds_with_aux
    assert isinstance(ds.odc, ODCExtensionDs)
    assert "aux" in ds

    gbox = ds.odc.geobox
    assert gbox is not None
    assert isinstance(gbox, GeoBox)

    driver = XrMemReaderDriver()
    assert driver.md_parser is not None

    template = driver.md_parser.extract(ds)
    assert isinstance(template, RasterGroupMetadata)
    assert len(template.bands) == 2
    assert ("xx", 1) in template.bands
    assert ("aux", 1) in template.bands
    assert ("xx", 1) in template.raster_bands
    assert ("aux", 1) in template.aux_bands

    assert template.bands[("xx", 1)].data_type == ds.xx.dtype
    assert template.bands[("aux", 1)].data_type == ds.aux.dtype

    cfgs: dict[str, RasterLoadParams | AuxLoadParams] = {
        "xx": RasterLoadParams.same_as(template.raster_bands[("xx", 1)]),
        "aux": AuxLoadParams.same_as(template.aux_bands[("aux", 1)]),
    }

    src_xx = RasterSource(
        "mem://xx",
        meta=template.raster_bands[("xx", 1)],
        driver_data=driver.md_parser.driver_data(ds, ("xx", 1)),
    )
    src_aux = AuxDataSource(
        "mem://aux",
        meta=template.aux_bands[("aux", 1)],
        driver_data=driver.md_parser.driver_data(ds, ("aux", 1)),
    )

    srcs: list[dict[str, RasterSource | AuxDataSource]] = [
        {"xx": src_xx, "aux": src_aux},
    ]
    tyx_bins = {(0, 0, 0): [0]}
    gbt = GeoboxTiles(gbox, gbox.shape.yx)
    tss = [datetime(2020, 1, 1)]
    env = driver.capture_env()

    oo = chunked_load(
        cfgs,
        template,
        srcs,
        tyx_bins,
        gbt,
        tss,
        env,
        driver,
        chunks=chunks,
    )
    assert isinstance(oo, xr.Dataset)
    assert "xx" in oo
    assert "aux" in oo
    assert oo.xx.dtype == ds.xx.dtype
    assert oo.aux.dtype == ds.aux.dtype
    assert oo.xx.shape == (1, *gbox.shape.yx)

    # again, aux only this time
    cfgs.pop("xx")
    oo = chunked_load(
        cfgs,
        template,
        srcs,
        tyx_bins,
        gbt,
        tss,
        env,
        driver,
        chunks=chunks,
    )
    assert isinstance(oo, xr.Dataset)
    assert "aux" in oo
    assert oo.aux.dtype == ds.aux.dtype
    assert oo.aux.shape == ds.aux.shape
    assert "xx" not in oo
    assert "x" not in oo.coords
    assert "y" not in oo.coords


def test_extract_zarr_spec() -> None:
    assert extract_zarr_spec({}) is None
    assert extract_zarr_spec({"something": "else"}) is None

    spec = {
        ".zgroup": {"zarr_format": 2},
        ".zattrs": {},
    }
    consolidated = {
        "zarr_consolidated_format": 1,
        "metadata": spec,
    }
    ref_fs = {
        ".zmetadata": json.dumps(consolidated).encode("utf-8"),
    }
    assert extract_zarr_spec(spec) == spec
    assert extract_zarr_spec(consolidated) == spec
    assert extract_zarr_spec(ref_fs) == spec
