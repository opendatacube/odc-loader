"""
Tools for constructing xarray objects from parsed metadata.
"""

from ._builder import chunked_load, resolve_chunk_shape
from ._driver import reader_driver, register_driver, unregister_driver
from ._reader import (
    resolve_dst_dtype,
    resolve_dst_nodata,
    resolve_load_cfg,
    resolve_src_nodata,
)
from ._rio import RioDriver, RioReader, configure_rio, configure_s3_access
from .types import (
    BandIdentifier,
    BandKey,
    BandQuery,
    FixedCoord,
    MultiBandSource,
    RasterBandMetadata,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterSource,
    ReaderDriver,
)

__all__ = (
    "BandIdentifier",
    "BandKey",
    "BandQuery",
    "RasterBandMetadata",
    "RasterLoadParams",
    "RasterSource",
    "FixedCoord",
    "MultiBandSource",
    "RasterGroupMetadata",
    "ReaderDriver",
    "chunked_load",
    "register_driver",
    "unregister_driver",
    "reader_driver",
    "resolve_load_cfg",
    "resolve_src_nodata",
    "resolve_dst_nodata",
    "resolve_dst_dtype",
    "resolve_chunk_shape",
    "configure_rio",
    "RioDriver",
    "RioReader",
    "configure_s3_access",
)


def __getattr__(name: str) -> str:
    # pylint: disable=import-outside-toplevel
    from importlib.metadata import version

    if name == "__version__":
        return version("odc_loader")

    raise AttributeError(f"module {__name__} has no attribute {name}")
