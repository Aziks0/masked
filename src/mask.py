import math
import numpy as np

from utils.number import tuples_operation
from utils.types import number


def extend_line(
    p1: tuple[number, number], p2: tuple[number, number], distance: number
) -> tuple[tuple[number, number], tuple[number, number]]:
    v_x, v_y = tuples_operation(p2, p1, "sub")
    v_norm = math.hypot(v_x, v_y)
    v_prime = tuples_operation((v_x, v_y), v_norm, "div")
    add_distance = tuples_operation(v_prime, distance, "mul")
    p1 = tuples_operation(p1, add_distance, "sub")
    p2 = tuples_operation(p2, add_distance, "add")
    return p1, p2


def get_mask_coordinates(keypoints):
    try:
        re = keypoints["right_eye"]
        le = keypoints["left_eye"]
    except KeyError:
        return None

    re_x, re_y = re
    le_x, le_y = le
    eyes_distance = math.hypot(le_x - re_x, le_y - re_y)

    # Mask height

    shrinked_width = eyes_distance * 0.2
    if shrinked_width <= 0:
        return None
    shinked_re, _ = extend_line(re, le, 0 - shrinked_width)

    # Shift vector (right_eye, left_eye) to origin
    le_origin_x, le_origin_y = tuples_operation(le, shinked_re, "sub")

    # Rotate the new vector by 90°
    le_origin_90 = (le_origin_y, 0 - le_origin_x)
    le_origin_90_x, le_origin_90_y = le_origin_90

    # Rotate the new vector by 180°
    le_origin_180 = (0 - le_origin_90_x, 0 - le_origin_90_y)

    # Mask width

    extended_width = eyes_distance * 0.6
    re, le = extend_line(re, le, extended_width)

    # Get the mask rectangle
    # A  B
    # D  C
    A = tuples_operation(le_origin_180, re, "add")
    B = tuples_operation(le_origin_180, le, "add")
    C = tuples_operation(le_origin_90, le, "add")
    D = tuples_operation(le_origin_90, re, "add")

    return np.array(
        [
            [
                tuple(map(round, A)),
                tuple(map(round, B)),
                tuple(map(round, C)),
                tuple(map(round, D)),
            ]
        ]
    )
