from collections.abc import Iterable
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def nth(iterable: Iterable[T], n: int, default: T | None = None) -> T | None:
    """Returns the nth item or a default value"""
    return next(islice(iterable, n, None), default)
