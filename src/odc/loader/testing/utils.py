import numpy as np


def dummy_fuser(dst: np.ndarray, src: np.ndarray) -> None:
    np.copyto(dst, src)
