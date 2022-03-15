import numpy as np

from detectron2.structures import Keypoints


def convert_keypoints(keypoints):
    if isinstance(keypoints, Keypoints):
        keypoints = keypoints.tensor
    return np.asarray(keypoints)


def get_visible_keypoints(keypoint_names, keypoints):
    _keypoints = []
    for kps in keypoints:
        visible = {}
        for i, keypoint in enumerate(kps):
            x, y, _ = keypoint
            keypoint_name = keypoint_names[i]
            visible[keypoint_name] = tuple(map(int, (x, y)))
        _keypoints.append(visible)
    return _keypoints


def sort_keypoints_by_scores(keypoints, scores):
    _keypoints = []
    scores_argsort = np.argsort(scores, kind="heapsort")
    scores_argsort = np.flipud(scores_argsort)  # Descending order
    for argsort in scores_argsort:
        _keypoints.append(keypoints[argsort])
    return _keypoints


def get_duplicate_keypoints_indexes(
    keypoints, le_x: int, le_y: int, re_x: int, re_y: int
):
    delete_index = []
    for i, kps in enumerate(keypoints):
        try:
            cur_le_x, cur_le_y = kps["left_eye"]
            cur_re_x, cur_re_y = kps["right_eye"]
        except KeyError:
            continue
        # We consider the left eye to be a duplicate if its positioned in a
        # 10x10 box with the current left eye as its center
        if (
            cur_le_x - 10 <= le_x
            and le_x <= cur_le_x + 10
            and cur_le_x - 10 <= le_y
            and le_y <= cur_le_y + 10
        ):
            delete_index.append(i)
            continue
        # We consider the right eye to be a duplicate if its positioned in a
        # 10x10 box with the current right eye as its center
        if (
            cur_re_x - 10 <= re_x
            and re_x <= cur_re_x + 10
            and cur_re_x - 10 <= re_y
            and re_y <= cur_re_y + 10
        ):
            delete_index.append(i)
    return delete_index


def remove_duplicate_keypoints(keypoints):
    _keypoints = []
    while len(keypoints) > 0:
        # Pop arrays
        cur_keypoints = keypoints[0]
        keypoints = keypoints[1:]

        try:
            cur_le_x, cur_le_y = cur_keypoints["left_eye"]
            cur_re_x, cur_re_y = cur_keypoints["right_eye"]
        except KeyError:
            continue

        _keypoints.append(cur_keypoints)
        if len(keypoints) == 0:
            break

        delete_index = get_duplicate_keypoints_indexes(
            keypoints, cur_le_x, cur_le_y, cur_re_x, cur_re_y
        )
        if len(delete_index) != 0:
            keypoints = np.delete(keypoints, delete_index)
    return _keypoints
