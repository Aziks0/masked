#!/usr/bin/env python3

import numpy as np
import cv2, tqdm, argparse, sys
from colorama import init

from video import (
    get_frame_generator,
    copy_audio_from_video,
    get_capture_and_writer,
)
from keypoint import (
    convert_keypoints,
    get_visible_keypoints,
    sort_keypoints_by_scores,
    remove_duplicate_keypoints,
)
from mask import get_mask_coordinates

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.logger import setup_logger

init()  # Colorama init
setup_logger()  # Detectron logger


def setup_cfg(threshold: float = 0.9, use_cpu=False):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cpu" if use_cpu else "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    return cfg


def visualize_predictions(predictor, frame, v: VideoVisualizer):
    predictions = predictor(frame)["instances"].to("cpu")
    if len(predictions) == 0:
        return frame

    visualization = v.draw_instance_predictions(frame, predictions)
    return visualization.get_image()


def frame_anonymize(
    predictor,
    frame,
    metadata,
    no_duplicate: bool,
    num_faces: int,
    mask_scale: tuple[float, float],
    mask_color: tuple[int, int, int],
):
    predictions = predictor(frame)["instances"].to("cpu")
    if len(predictions) == 0:
        return frame

    keypoints = (
        predictions.pred_keypoints if predictions.has("pred_keypoints") else None
    )
    keypoints = convert_keypoints(keypoints)
    if keypoints is None:
        return frame

    scores = predictions.scores
    scores = np.asarray(scores)
    keypoints = sort_keypoints_by_scores(keypoints, scores)

    keypoint_names = metadata.get("keypoint_names")
    keypoints = get_visible_keypoints(keypoint_names, keypoints)

    if no_duplicate and (num_faces is None or (len(keypoints) > num_faces)):
        keypoints = remove_duplicate_keypoints(keypoints)

    if num_faces is not None and len(keypoints) > num_faces:
        keypoints = keypoints[:num_faces]

    for kps in keypoints:
        mask_coordinates = get_mask_coordinates(kps, mask_scale)
        if mask_coordinates is not None:
            cv2.fillPoly(frame, mask_coordinates, mask_color)

    return frame


def video_anonymize(
    input_file: str,
    output_file: str,
    threshold: float,
    mask_scale: tuple[float, float],
    mask_color: tuple[int, int, int],
    no_duplicate: bool,
    num_faces: int,
    use_cpu: bool,
    visualize: bool,
):
    cfg = setup_cfg(threshold, use_cpu)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    predictor = DefaultPredictor(cfg)

    if visualize:
        v = VideoVisualizer(metadata)

    cap, out = get_capture_and_writer(input_file, output_file)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_gen = get_frame_generator(cap)
    bar = tqdm.tqdm(total=int(num_frames))
    for frame in frame_gen:
        frame_anon = (
            visualize_predictions(predictor, frame, v)
            if visualize
            else frame_anonymize(
                predictor,
                frame,
                metadata,
                no_duplicate,
                num_faces,
                mask_scale,
                mask_color,
            )
        )
        out.write(frame_anon)
        bar.update()

    bar.close()
    cap.release()
    out.release()


def get_parser():
    def type_mask_scale(value):
        try:
            value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError("invalid type(s), should be a float.")
        if value <= 0:
            raise argparse.ArgumentTypeError("invalid value(s), should be > 0.")
        return value

    def type_mask_color(value):
        try:
            value = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("invalid type(s), should be an int.")
        if value < 0 or value > 255:
            raise argparse.ArgumentTypeError(
                "invalid value(s), should be >= 0 and <= 255."
            )
        return value

    parser = argparse.ArgumentParser(
        prog="masked", description="Anonymize faces in videos"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to video file.")
    parser.add_argument(
        "--output", "-o", default="out.mp4", help="Path to the output video file."
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.9,
        help="Minimum score for faces to be masked, [0,1].",
    )
    parser.add_argument(
        "--mask-scale",
        "-ms",
        nargs=2,
        type=type_mask_scale,
        default=[1, 1],
        metavar=("WIDTH", "HEIGHT"),
        help="Scale factor for face masks, ]0,+∞].",
    )
    parser.add_argument(
        "--mask-color",
        "-mc",
        nargs=3,
        type=type_mask_color,
        default=[127, 127, 127],
        metavar=("RED", "GREEN", "BLUE"),
        help="Face masks color, in RGB.",
    )
    parser.add_argument(
        "--remove-duplicates",
        "-rd",
        action="store_true",
        help="Try to remove duplicated detections.",
    )
    parser.add_argument(
        "--faces",
        "-f",
        type=int,
        default=None,
        help="Specify the number of faces that will appear in the video."
        + " Only the X most probable detections will be anonymized.",
    )
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Don't keep the audio from the input video.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Draw scores and boxes. This argument is only compatible"
        + " with {input|output|threshold|no-audio} arguments.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    input_file = args.input
    output_file = args.output
    threshold = args.threshold
    mask_scale = tuple(args.mask_scale)
    mask_color = tuple(args.mask_color)[::-1]  # RGB to BGR
    no_duplicate = args.remove_duplicates
    num_faces = args.faces
    use_cpu = args.cpu
    keep_audio = not args.no_audio
    visualize = args.visualize

    video_anonymize(
        input_file,
        output_file,
        threshold,
        mask_scale,
        mask_color,
        no_duplicate,
        num_faces,
        use_cpu,
        visualize,
    )

    if keep_audio:
        success = copy_audio_from_video(input_file, output_file)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
