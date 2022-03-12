import cv2

def get_frame_generator(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def get_capture_and_writer(input: str, output: str):
    cap = cv2.VideoCapture(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return cap, out
