"""Default fuser function and fuser utils"""
import numpy as np

from ._reader import nodata_mask
from .types import FuserFunc


def nodata_fuser(dst: np.ndarray, src: np.ndarray, nodata: int | float) -> None:
    missing = nodata_mask(dst, nodata)
    np.copyto(dst, src, where=missing)


def fuser_for_nodata(nodata: int | float) -> FuserFunc:
    def out(dst: np.ndarray, src: np.ndarray) -> None:
        nodata_fuser(dst, src, nodata)
    return out
