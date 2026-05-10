"""
Various Dask helpers.
"""

from collections.abc import Callable, Generator, Hashable, Iterable, MutableMapping
from typing import Any, TypeVar

from dask.base import tokenize

T = TypeVar("T")


def tokenize_stream(
    xx: Iterable[T],
    key: Callable[[str], Hashable] | None = None,
    dsk: MutableMapping[Hashable, Any] | None = None,
) -> Generator[tuple[Hashable, T]]:
    if key:
        kx = ((key(tokenize(x)), x) for x in xx)
    else:
        kx = ((tokenize(x), x) for x in xx)

    if dsk is None:
        yield from kx
    else:
        for k, x in kx:
            dsk[k] = x
            yield k, x
