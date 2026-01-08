"""Default fuser function and fuser utils"""
import numpy as np
from importlib import import_module
from typing import cast

from ._reader import nodata_mask
from .types import FuserFunc


def nodata_fuser(dst: np.ndarray, src: np.ndarray, nodata: int | float) -> None:
    """Default fuser - only fill where src is nodata"""
    missing = nodata_mask(dst, nodata)
    np.copyto(dst, src, where=missing)


def fuser_for_nodata(nodata: int | float) -> FuserFunc:
    """Create a nodata fuser function for a particular nodata value"""
    def out(dst: np.ndarray, src: np.ndarray) -> None:
        nodata_fuser(dst, src, nodata)
    return out

def resolve_fuser(fqn: str) -> FuserFunc:
    """Resolve a fuser function from fully qualified function name"""
    mod_name, func_name = fqn.rsplit(".", 1)
    try:
        mod = import_module(mod_name)
        func = getattr(mod, func_name)
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        raise ValueError(f"Could not import python object: {fqn}") from None
    if not callable(func):
        raise ValueError(f"{fqn} is not a function")

    # Assume imported function has correct signature
    return cast(FuserFunc, func)
