"""Metadata and data loading model classes."""

from __future__ import annotations

import json
from dataclasses import astuple, dataclass, field, replace
from typing import (
    Any,
    ContextManager,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from odc.geo import Unset
from odc.geo.geobox import GeoBox, GeoBoxBase

T = TypeVar("T")

BandKey: TypeAlias = Tuple[str, int]
"""Asset Name, band index within an asset (1 based, 0 indicates "all the bands")."""

BandIdentifier: TypeAlias = Union[str, BandKey]
"""Alias or canonical band identifier."""

BandQuery: TypeAlias = Optional[Union[str, Sequence[str]]]
"""One|All|Some bands"""

ReaderSubsetSelection: TypeAlias = Any
GlobalLoadContext: TypeAlias = Any
LocalLoadContext: TypeAlias = Any

Band_DType: TypeAlias = Union[DTypeLike, Mapping[str, DTypeLike]]


@dataclass(eq=True, frozen=True)
class RasterBandMetadata:
    """
    Common raster metadata per band.

    We assume that all assets of the same name have the same "structure" across different items
    within a collection. Specifically, that they encode pixels with the same data type, use the same
    ``nodata`` value and have common units.

    These values are extracted from the ``eo:bands`` extension, but can also be supplied by the user
    from the config.
    """

    data_type: Optional[str] = None
    """Numpy compatible dtype string."""

    nodata: Optional[float] = None
    """Nodata marker/fill_value."""

    units: str = "1"
    """Units of the pixel data."""

    dims: Tuple[str, ...] = ()
    """Dimension names for this band.

    e.g. ("y", "x", "wavelength")
    """

    driver_data: Any = None
    """IO Driver specific extra data."""

    attrs: dict[str, Any] = field(default_factory=dict)
    """Additional raster band attributes."""

    def with_defaults(self, defaults: "RasterBandMetadata") -> "RasterBandMetadata":
        """
        Merge with another metadata object, using self as the primary source.

        If a field is None in self, use the value from defaults.
        """
        return RasterBandMetadata(
            data_type=with_default(self.data_type, defaults.data_type),
            nodata=with_default(self.nodata, defaults.nodata),
            units=with_default(self.units, defaults.units, "1"),
            dims=with_default(self.dims, defaults.dims, ()),
            driver_data=with_default(self.driver_data, defaults.driver_data),
            attrs=with_default(self.attrs, defaults.attrs),
        )

    def patch(self, **kwargs) -> "RasterBandMetadata":
        """
        Return a new object with updated fields.
        """
        return replace(self, **kwargs)

    def __dask_tokenize__(self):
        return astuple(self)

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the RasterBandMetadata object.
        """
        return {
            "data_type": self.data_type,
            "nodata": _maybe_json(self.nodata),
            "units": self.units,
            "dims": self.dims,
            "driver_data": _maybe_json(
                self.driver_data,
                allow_nan=False,
                roundtrip=True,
                on_error="SET, NOT JSON SERIALIZABLE",
            ),
            "attrs": self.attrs,
        }

    @property
    def extra_dims(self) -> tuple[str, ...]:
        """
        Non-spatial dimension names.
        """
        return _extra_dims(self.dims)

    @property
    def ydim(self) -> int:
        """Index of y dimension, typically 0."""
        return _ydim(self.dims)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.extra_dims) + 2

    @property
    def unit(self) -> str:
        """
        Alias for units.
        """
        return self.units

    @property
    def dtype(self) -> str | None:
        """
        Alias for data_type.
        """
        return self.data_type


@dataclass(eq=True, frozen=True)
class AuxBandMetadata:
    """
    Metadata for an auxiliary band.
    """

    data_type: Optional[str] = None
    """Numpy compatible dtype string."""

    nodata: Optional[float] = None
    """Nodata marker/fill_value."""

    units: str = "1"
    """Units of the data."""

    dims: Tuple[str, ...] = ()
    """Dimension names for this auxilliary band.

    e.g. ("time",) or ("index",) or ()
    """

    driver_data: Any = None
    """IO Driver specific extra data."""

    attrs: dict[str, Any] = field(default_factory=dict)
    """Additional auxilliary band attributes."""

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the AuxBandMetadata object.
        """
        return {
            "data_type": self.data_type,
            "nodata": _maybe_json(self.nodata),
            "units": self.units,
            "dims": self.dims,
            "driver_data": _maybe_json(
                self.driver_data,
                allow_nan=False,
                roundtrip=True,
                on_error="SET, NOT JSON SERIALIZABLE",
            ),
            "attrs": self.attrs,
        }

    @property
    def unit(self) -> str:
        """
        Alias for units.
        """
        return self.units

    @property
    def dtype(self) -> str | None:
        """
        Alias for data_type.
        """
        return self.data_type


@dataclass(eq=True)
class FixedCoord:
    """
    Encodes extra coordinate info.
    """

    name: str
    values: Sequence[Any]
    dtype: str = ""
    dim: str = ""
    units: str = "1"

    def __post_init__(self) -> None:
        if not self.dtype:
            self.dtype = np.array(self.values).dtype.str
        if not self.dim:
            self.dim = self.name

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the FixedCoord object.
        """

        return {
            "name": self.name,
            "values": [_maybe_json(v) for v in self.values],
            "dim": self.dim,
            "dtype": str(self.dtype),
            "units": str(self.units),
        }


@dataclass(eq=True, frozen=True)
class RasterGroupMetadata:
    """
    STAC Collection/Datacube Product abstraction.
    """

    bands: Mapping[BandKey, RasterBandMetadata | AuxBandMetadata]
    """
    Bands are assets that contain raster data.

    This controls which assets are extracted from STAC.
    """

    aliases: dict[str, list[BandKey]] = field(default_factory=dict)
    """
    Alias map ``alias -> [(asset, idx),...]``.

    Used to rename bands at load time.
    """

    extra_dims: dict[str, int] = field(default_factory=dict)
    """
    Expected extra dimensions other than time and spatial.

    Must be same size across items/datasets.
    """

    extra_coords: Sequence[FixedCoord] = ()
    """
    Coordinates for extra dimensions.

    Must be same values across items/datasets.
    """

    def patch(self, **kwargs) -> "RasterGroupMetadata":
        """
        Return a new object with updated fields.
        """
        return replace(self, **kwargs)

    def merge(self, other: "RasterGroupMetadata") -> "RasterGroupMetadata":
        """
        Merge with another metadata object, using self as the primary source.
        """
        if self == other:
            return self

        bands = {**self.bands, **other.bands}
        aliases = _merge_aliases(self.aliases, other.aliases)
        extra_dims = {**self.extra_dims, **other.extra_dims}
        extra_coords = _merge_unique(self.extra_coords, other.extra_coords)

        return RasterGroupMetadata(bands, aliases, extra_dims, tuple(extra_coords))

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the RasterGroupMetadata object.
        """
        # pylint: disable=protected-access
        return {
            "bands": {
                f"{name}.{idx}": v._repr_json_()
                for (name, idx), v in self.bands.items()
            },
            "aliases": self.aliases,
            "extra_dims": self.extra_dims,
            "extra_coords": [c._repr_json_() for c in self.extra_coords],
        }

    def extra_dims_full(self, band: BandIdentifier | None = None) -> dict[str, int]:
        dims = {**self.extra_dims}
        for coord in self.extra_coords:
            if coord.dim not in dims:
                dims[coord.dim] = len(coord.values)

        if band is not None:
            band_dims = self.bands[norm_key(band)].dims
            dims = {k: v for k, v in dims.items() if k in band_dims}

        return dims

    @property
    def raster_bands(self) -> dict[BandKey, RasterBandMetadata]:
        """
        Return a dictionary of raster bands.
        """
        return {
            k: v for k, v in self.bands.items() if isinstance(v, RasterBandMetadata)
        }

    @property
    def aux_bands(self) -> dict[BandKey, AuxBandMetadata]:
        """
        Return a dictionary of auxiliary bands.
        """
        return {k: v for k, v in self.bands.items() if isinstance(v, AuxBandMetadata)}


@dataclass(eq=True, frozen=True)
class RasterSource:
    """
    Captures known information about a single band.
    """

    uri: str
    """Asset location."""

    band: int = 1
    """One based band index (default=1)."""

    subdataset: Optional[str] = None
    """Used for netcdf/hdf5 sources."""

    geobox: Optional[GeoBoxBase] = None
    """Data footprint/shape/projection if known."""

    meta: Optional[RasterBandMetadata] = None
    """Expected raster dtype/nodata."""

    driver_data: Any = None
    """IO Driver specific extra data."""

    def patch(self, **kwargs) -> "RasterSource":
        """
        Return a new object with updated fields.
        """
        return replace(self, **kwargs)

    def strip(self) -> "RasterSource":
        """
        Copy with minimal data only.

        Removes geobox, as it's not needed for data loading.
        """
        return RasterSource(
            self.uri,
            band=self.band,
            subdataset=self.subdataset,
            geobox=None,
            meta=self.meta,
            driver_data=self.driver_data,
        )

    @property
    def ydim(self) -> int:
        """Index of y dimension, typically 0."""
        if self.meta is None:
            return 0
        return self.meta.ydim

    def __dask_tokenize__(self):
        return (self.uri, self.band, self.subdataset)

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the RasterSource object.
        """
        doc = {
            "uri": self.uri,
            "band": self.band,
        }

        if self.subdataset is not None:
            doc["subdataset"] = self.subdataset

        if self.meta is not None:
            doc.update(self.meta._repr_json_())  # pylint: disable=protected-access

        gbox = self.geobox
        if gbox is not None:
            doc["crs"] = str(gbox.crs)
            doc["shape"] = list(gbox.shape.yx)
            if isinstance(gbox, GeoBox):
                doc["transform"] = [*gbox.transform][:6]

        doc["driver_data"] = _maybe_json(
            self.driver_data,
            allow_nan=False,
            roundtrip=True,
            on_error="SET, NOT JSON SERIALIZABLE",
        )

        return doc


@dataclass(eq=True, frozen=True)
class AuxDataSource:
    """
    Captures known information about a single auxiliary band.
    """

    uri: str
    """Asset location."""

    subdataset: Optional[str] = None
    """Used for netcdf/hdf5 sources."""

    meta: Optional[AuxBandMetadata] = None
    """Expected raster dtype/nodata."""

    driver_data: Any = None
    """IO Driver specific extra data."""

    def patch(self, **kwargs) -> "AuxDataSource":
        """
        Return a new object with updated fields.
        """
        return replace(self, **kwargs)

    def strip(self) -> "AuxDataSource":
        """
        Compatibility with RasterSource.strip()
        """
        return self


MultiBandSource: TypeAlias = Mapping[str, RasterSource | AuxDataSource | None]
"""Mapping from band name on output to DataSource, raster or auxiliary."""


@dataclass
class RasterLoadParams:
    """
    Captures data loading configuration.
    """

    # pylint: disable=too-many-instance-attributes

    dtype: Optional[str] = None
    """Output dtype, default same as source."""

    fill_value: Optional[float] = None
    """Value to use for missing pixels."""

    src_nodata_fallback: Optional[float] = None
    """
    Fallback ``nodata`` marker for source.

    Used to deal with broken data sources. If file is missing ``nodata`` marker and
    ``src_nodata_fallback`` is set then treat source pixels with that value as missing.
    """

    src_nodata_override: Optional[float] = None
    """
    Override ``nodata`` marker for source.

    Used to deal with broken data sources. Ignore ``nodata`` marker of the source file even if
    present and use this value instead.
    """

    use_overviews: bool = True
    """
    Disable use of overview images.

    Set to ``False`` to always read from the main image ignoring overview images
    even when present in the data source.
    """

    resampling: str = "nearest"
    """Resampling method to use."""

    fail_on_error: bool = True
    """Quit on the first error or continue."""

    dims: Tuple[str, ...] = ()
    """Dimension names for this band."""

    meta: Optional[RasterBandMetadata] = None
    """Expected raster band metadata."""

    def patch(self, **kwargs) -> "RasterLoadParams":
        """
        Return a new object with updated fields.
        """
        return replace(self, **kwargs)

    @property
    def ydim(self) -> int:
        """Index of y dimension, typically 0."""
        return _ydim(self.dims)

    @property
    def extra_dims(self) -> tuple[str, ...]:
        """
        Non-spatial dimension names.
        """
        return _extra_dims(self.dims)

    @staticmethod
    def same_as(src: Union[RasterBandMetadata, RasterSource]) -> "RasterLoadParams":
        """Construct from source object."""
        if isinstance(src, RasterBandMetadata):
            meta = src
        else:
            meta = src.meta or RasterBandMetadata()

        dtype = meta.data_type
        if dtype is None:
            dtype = "float32"

        return RasterLoadParams(
            dtype=dtype, fill_value=meta.nodata, dims=meta.dims, meta=meta
        )

    @property
    def nearest(self) -> bool:
        """Report True if nearest resampling is used."""
        return self.resampling == "nearest"

    def __dask_tokenize__(self):
        return astuple(self)

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the RasterLoadParams object.
        """
        # pylint: disable=protected-access
        return {
            "dtype": _maybe_json(self.dtype, on_error=str),
            "fill_value": _maybe_json(self.fill_value),
            "src_nodata_fallback": self.src_nodata_fallback,
            "src_nodata_override": self.src_nodata_override,
            "use_overviews": self.use_overviews,
            "resampling": self.resampling,
            "fail_on_error": self.fail_on_error,
            "dims": list(self.dims),
            "meta": self.meta._repr_json_() if self.meta is not None else None,
        }


@dataclass(eq=True)
class AuxLoadParams:
    """
    Captures data loading configuration for auxiliary bands.
    """

    dtype: Optional[str] = None
    """Output dtype, default same as source."""

    fill_value: Optional[float] = None
    """Value used in-place of missing data."""

    meta: Optional[AuxBandMetadata] = None
    """Expected auxiliary band metadata."""

    def patch(self, **kwargs) -> "AuxLoadParams":
        """
        Return a new object with updated fields.
        """
        return replace(self, **kwargs)

    @staticmethod
    def same_as(src: Union[AuxBandMetadata, AuxDataSource]) -> "AuxLoadParams":
        """Construct from source object."""
        if isinstance(src, AuxBandMetadata):
            meta = src
        else:
            meta = src.meta or AuxBandMetadata()

        dtype = meta.data_type
        if dtype is None:
            dtype = "float32"

        return AuxLoadParams(dtype=dtype, fill_value=meta.nodata, meta=meta)

    def __dask_tokenize__(self):
        return astuple(self)

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the AuxLoadParams object.
        """
        # pylint: disable=protected-access
        return {
            "dtype": _maybe_json(self.dtype),
            "fill_value": _maybe_json(self.fill_value),
            "meta": self.meta._repr_json_() if self.meta is not None else None,
        }


class MDParser(Protocol):
    """
    Protocol for metadata parsers.

    - Parse group level metadata
      - data bands and their expected type
      - extra dimensions and coordinates
    - Extract driver specific data
    """

    def extract(self, md: Any) -> RasterGroupMetadata: ...
    def driver_data(self, md: Any, band_key: BandKey) -> Any: ...


class RasterReader(Protocol):
    """
    Protocol for raster readers.
    """

    # pylint: disable=too-few-public-methods

    def read(
        self,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        *,
        dst: Optional[np.ndarray] = None,
        selection: Optional[ReaderSubsetSelection] = None,
    ) -> tuple[tuple[slice, slice], np.ndarray]: ...


class DaskRasterReader(Protocol):
    """
    Protocol for raster readers that produce Dask sub-graphs.

    ``.read`` method should return a Dask future evaluating to a numpy array of
    pixels for a given geobox, alternatively dask future may evaluate to a
    subset of the geobox overlapping with the source. In this case Dask future
    should evaluate to a tuple: ``(yx_slice, pixels)``, such that
    ``dst_geobox[yx_slice].shape == pixels.shape[ydim:ydim+2]``.
    """

    # pylint: disable=too-few-public-methods

    def read(
        self,
        dst_geobox: GeoBox,
        *,
        selection: Optional[ReaderSubsetSelection] = None,
        idx: tuple[int, ...],
    ) -> Any: ...

    def open(
        self,
        src: RasterSource,
        cfg: RasterLoadParams,
        ctx: GlobalLoadContext,
        *,
        layer_name: str,
        idx: int,
    ) -> "DaskRasterReader": ...


class AuxReader(Protocol):
    """
    Protocol for auxiliary data readers.

    Read auxiliary data.

    :param srcs: auxiliary data sources grouped by time
    :param cfg: Loading configuration
    :param used_names: Names claimed by raster bands and their coordinates
    :param available_coords: Available coordinates, typically ``time`` is useful
    :param ctx: Load context
    :param dask_layer_name: Suggested dask layer name, when reading with dask
    :return: Auxiliary data loaded into a :py:class:`xarray.DataArray`
    """

    # pylint: disable=too-few-public-methods

    def read(
        self,
        srcs: Sequence[Sequence[AuxDataSource]],
        cfg: AuxLoadParams,
        used_names: set[str],
        available_coords: Mapping[str, xr.DataArray],
        ctx: GlobalLoadContext,
        *,
        dask_layer_name: str | None = None,
    ) -> xr.DataArray: ...


class ReaderDriver(Protocol):
    """
    Protocol for reader drivers.
    """

    def new_load(
        self,
        geobox: GeoBox,
        *,
        chunks: None | Dict[str, int] = None,
    ) -> GlobalLoadContext: ...

    def finalise_load(self, load_state: GlobalLoadContext) -> Any: ...

    def capture_env(self) -> dict[str, Any]: ...

    def restore_env(
        self, env: dict[str, Any], load_state: GlobalLoadContext
    ) -> ContextManager[LocalLoadContext]: ...

    def open(self, src: RasterSource, ctx: LocalLoadContext) -> RasterReader: ...

    @property
    def md_parser(self) -> MDParser | None: ...

    @property
    def dask_reader(self) -> DaskRasterReader | None: ...

    @property
    def aux_reader(self) -> AuxReader | None: ...


ReaderDriverSpec: TypeAlias = Union[str, ReaderDriver]


def is_reader_driver(x: Any) -> bool:
    """
    Check if x is a ReaderDriver.
    """
    expected_attributes = [x for x in dir(ReaderDriver) if not x.startswith("_")]
    return all(hasattr(x, a) for a in expected_attributes)


BAND_DEFAULTS = RasterBandMetadata("float32", None, "1")


def with_default(v: Optional[T], default_value: T, *other_defaults) -> T:
    """
    Replace ``None`` with default value.

    :param v: Value that might be None
    :param default_value: Default value of the same type as v
    :return: ``v`` unless it is ``None`` then return ``default_value`` instead
    """
    if v is None:
        return default_value
    if v in other_defaults:
        return default_value
    return v


def norm_nodata(nodata) -> Union[float, None]:
    if nodata is None:
        return None
    if isinstance(nodata, (int, float)):
        return nodata
    return float(nodata)


def norm_band_metadata(
    v: Union[RasterBandMetadata, Mapping[str, Any]],
    fallback: RasterBandMetadata = BAND_DEFAULTS,
) -> RasterBandMetadata:
    if isinstance(v, RasterBandMetadata):
        return v
    # in STAC it's "unit" not "units", so check both
    units = v.get("units", v.get("unit", fallback.units))

    # non-STAC addition
    dims = v.get("dims", v.get("dims", fallback.dims))

    return RasterBandMetadata(
        v.get("data_type", fallback.data_type),
        v.get("nodata", norm_nodata(fallback.nodata)),
        units,
        tuple(dims),
    )


def norm_key(k: BandIdentifier) -> BandKey:
    """
    ("band", i) -> ("band", i)
    "band" -> ("band", 1)
    "band.3" -> ("band", 3)
    "band.tiff" -> ("band.tiff", 1)
    """
    if isinstance(k, str):
        parts = k.rsplit(".", 1)
        if len(parts) == 2:
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return (k, 1)
        return (k, 1)
    return k


def _ydim(dims: Tuple[str, ...]) -> int:
    if dims:
        return dims.index("y")
    return 0


def _extra_dims(dims: tuple[str, ...]) -> tuple[str, ...]:
    if not dims:
        return ()
    ydim = _ydim(dims)
    return dims[:ydim] + dims[ydim + 2 :]


def _jsonify_float(nodata: float) -> float | str:
    assert isinstance(nodata, float)
    if np.isfinite(nodata):
        return nodata
    return str(nodata)


def _maybe_json(
    obj,
    *,
    on_error=Unset(),
    allow_nan: bool = False,
    roundtrip: bool = False,
):
    """
    Try to convert object to json string, return on_error on failure
    """
    # pylint: disable=unnecessary-lambda-assignment
    if isinstance(obj, (int, str, bool, type(None))):
        return obj

    if isinstance(obj, float):
        return _jsonify_float(obj)

    if isinstance(on_error, Unset):
        on_error = lambda _: "** NOT JSON SERIALIZABLE **"

    if not callable(on_error):
        error_value = on_error
        on_error = lambda _: error_value

    try:
        json_txt = json.dumps(obj, allow_nan=allow_nan)
    except (ValueError, TypeError):
        return on_error(obj)

    if roundtrip:
        try:
            obj = json.loads(json_txt)
        except ValueError:
            return on_error(obj)

    return obj


def _merge_unique(a: Sequence[T], b: Sequence[T]) -> list[T]:
    """
    Merge two sequences, removing duplicates.

    :return: ``a`` + ``b``(without elements already in ``a``)
    """
    return [*a, *[v for v in b if v not in a]]


def _merge_aliases(
    a: dict[str, list[BandKey]], b: dict[str, list[BandKey]]
) -> dict[str, list[BandKey]]:
    """
    Merge two alias dictionaries.
    """

    out = {**a}
    for k, bb in b.items():
        if k in out:
            out[k] = _merge_unique(out[k], bb)
        else:
            out[k] = bb
    return out
