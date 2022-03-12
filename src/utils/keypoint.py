import numpy as np

from detectron2.structures import Keypoints


def convert_keypoints(keypoints):
    if isinstance(keypoints, Keypoints):
        keypoints = keypoints.tensor
    keypoints = np.asarray(keypoints)
    return keypoints


def get_visible_keypoints(keypoint_names, keypoints):
    visible = {}
    for i, keypoint in enumerate(keypoints):
        x, y, _ = keypoint
        keypoint_name = keypoint_names[i]
        visible[keypoint_name] = tuple(map(int, (x, y)))
    return visible
