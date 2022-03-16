from typing import Literal, Union
import numpy as np
from utils.types import number


def tuples_operation(
    t1: tuple[number, ...],
    t2: Union[tuple, number],
    operation: Literal["add", "sub", "mul", "div"],
) -> tuple[number, ...]:
    _len = len(t1)
    if type(t2) is tuple:
        assert _len == len(t2)
    else:
        value = t2
        t2 = np.arange(_len)
        t2 = np.full_like(t2, value)

    if operation == "add":
        return tuple(map(lambda p1, p2: p1 + p2, t1, t2))
    if operation == "sub":
        return tuple(map(lambda p1, p2: p1 - p2, t1, t2))
    if operation == "mul":
        return tuple(map(lambda p1, p2: p1 * p2, t1, t2))
    if operation == "div":
        return tuple(map(lambda p1, p2: p1 / p2, t1, t2))
