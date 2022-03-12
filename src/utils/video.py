import cv2, ffmpeg, subprocess, os, uuid

from utils.logging import log_error


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


def copy_audio_from_video(input, output):
    print("Copying audio from input to output..")
    try:
        subprocess.check_call(
            ["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        log_error(
            "Audio cannot be kept because FFmpeg has not been found,"
            + "\n you should verify that it is installed and accessible from the PATH"
            + "\n(see https://ffmpeg.org for more information)"
        )
        return False

    # FFmpeg cannot rewrite the output file, so we need to write the output with
    # audio to a new file and then remove the one without audio

    # Get a unique filename to avoid replacing an existing file
    output_dirname = os.path.dirname(output)
    tmp_filename = str(uuid.uuid4()) + ".mp4"
    tmp_output = os.path.join(output_dirname, tmp_filename)

    audio_input = ffmpeg.input(input).audio
    video_output = ffmpeg.input(output).video
    ffmpeg.output(video_output, audio_input, tmp_output, loglevel="quiet").run()

    os.remove(output)
    os.rename(tmp_output, output)
    return True
