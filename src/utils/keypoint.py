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
