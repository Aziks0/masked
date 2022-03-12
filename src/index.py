import cv2, tqdm, argparse
from colorama import init

from utils.video import (
    get_frame_generator,
    get_capture_and_writer,
)
from utils.keypoint import get_visible_keypoints, convert_keypoints

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

init()  # Colorama init
setup_logger()  # Detectron logger


def setup_cfg(threshold: float = 0.9):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    return cfg


def frame_anonymize(predictor, frame, metadata):
    predictions = predictor(frame)["instances"].to("cpu")
    if len(predictions) == 0:
        return frame

    keypoints = (
        predictions.pred_keypoints if predictions.has("pred_keypoints") else None
    )
    keypoints = convert_keypoints(keypoints)
    if keypoints is None:
        return frame

    keypoint_names = metadata.get("keypoint_names")
    for keypoints_prediction in keypoints:
        visible = get_visible_keypoints(keypoint_names, keypoints_prediction)
        try:
            le_x, le_y = visible["left_eye"]
            re_x, re_y = visible["right_eye"]
        except KeyError:
            continue
        cv2.line(frame, (le_x, le_y), (re_x, re_y), (0, 0, 255), 2)

    return frame


def video_anonymize(input_file: str, output_file: str, threshold: float):
    cfg = setup_cfg(threshold)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    predictor = DefaultPredictor(cfg)

    cap, out = get_capture_and_writer(input_file, output_file)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_gen = get_frame_generator(cap)
    bar = tqdm.tqdm(total=int(num_frames))
    for frame in frame_gen:
        frame_anon = frame_anonymize(predictor, frame, metadata)
        out.write(frame_anon)
        bar.update()

    bar.close()
    cap.release()
    out.release()


def get_parser():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("--input", "-i", required=True, help="Path to video file")
    parser.add_argument(
        "--output", "-o", default="out.mp4", help="Path to the output video file"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Minimum score for faces to be masked, [0,1]",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    input_file = args.input
    output_file = args.output
    threshold = args.threshold
    video_anonymize(input_file, output_file, threshold)
