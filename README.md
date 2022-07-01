# masked

`masked` is a cli tool made to automatically anonymize eyes in videos.
It uses [detectron2](https://github.com/facebookresearch/detectron2) for keypoints detections, and some weird math to apply a mask.

This project was inspired by one of the poster from the movie _Parasite_, so the attempt was to only cover the eyes _(just like in the poster)_.
If you tweak the default values, you can cover whole faces, but it might be easier for you to use a tool made for that purpose,
like [deface](https://github.com/ORB-HD/deface).

Parasite poster | `masked` demo
--|--
![Parasite Poster](https://user-images.githubusercontent.com/36425380/174674879-dcf50024-3fdb-4f1a-8be9-82bdea214a64.png "Parasite Poster") | <video src="https://user-images.githubusercontent.com/36425380/176862852-4455fc6c-0e3a-49b4-9167-f1aa4cd6b957.mp4" />

## Compatibility

`masked` is compatible with Windows, Linux and macOS.

If you have a nvidia CUDA-enabled GPU, you can _(and you really should)_ install CUDA to benefit from hardware acceleration, making the inference faster
_(see [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
or [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) CUDA installation guide)_.

If you don't have CUDA installed, you can still use `masked`, you just need to use the `--cpu` option.

## Installation

### Notes

`masked` uses [poetry](https://github.com/python-poetry/poetry) as a dependency manager, make sure it is installed before getting started
_(see [poetry installation instructions](https://python-poetry.org/docs/master/#installation)_).

It also uses [ffmpeg](https://github.com/FFmpeg/FFmpeg) in order to keep the audio of the videos you mask
_(see [ffmpeg website](https://ffmpeg.org/download.html) to download a binary)_.
**FFmpeg executable needs to be in your `PATH` for `masked` to use it.**
You can also use the `--no-audio` option to skip the audio copy.

### Steps

You first need to get the source of `masked`, and use poetry to create a virtualenv and install the dependencies:

```bash
git clone https://github.com/Aziks0/masked.git
cd masked
poetry install
```

You then need to install `detectron2` (and `pytorch` + `torchvision` if you build it from source)

> **Warning**
> 
> Remember that you are using `poetry`! Install the following package in the `poetry` virtualenv with `poetry run <PIP_COMMAND>`.

#### Linux

`detectron2` has [pre-built wheels](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only) for Linux,
but you can also [build it from source](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#installation) if you prefer.

#### macOS

You need to build `detectron2` from source, please refer to the [detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

#### Windows

You first need to install [pytorch](https://github.com/pytorch/pytorch) and [torchvison](https://github.com/pytorch/vision).
Please refer to the [pytorch website](https://pytorch.org/get-started/locally/) for the installation guide.

Then, you need to install `detectron2`. It doesn't support Windows -_officially_-, so you need a "custom" source to build and install it.
You can either use [my fork version](https://github.com/Aziks0/detectron2-windows.git)
or use your own fork and fix the source yourself _(see [this commit](https://github.com/Aziks0/detectron2-windows/commit/f3a598190da5775474d68ed24c16ab47e7f8cd89))_.

Exemple installation with `CUDA 11.3` installed:

```bash
poetry run pip install torch torchvison --extra-index-url https://download.pytorch.org/whl/cu113
poetry run pip install git+https://github.com/Aziks0/detectron2-windows.git
```

Please refer to the [detectron2 build guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source) for further information.

#### Usage

Use `poetry run masked -i <VIDEO_PATH>` to run the tool.
Check the available options [here](https://github.com/Aziks0/masked#options) or with the `poetry run masked -h` command.

## Options

```text
usage: masked [-h] --input INPUT [--output OUTPUT] [--threshold THRESHOLD] [--mask-scale WIDTH HEIGHT]
              [--mask-color RED GREEN BLUE] [--remove-duplicates] [--faces FACES] [--no-audio] [--visualize]

Anonymize faces in videos

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to video file.
  --output OUTPUT, -o OUTPUT
                        Path to the output video file.
  --threshold THRESHOLD, -t THRESHOLD
                        Minimum score for faces to be masked, [0,1].
  --mask-scale WIDTH HEIGHT, -ms WIDTH HEIGHT
                        Scale factor for face masks, ]0,+∞].
  --mask-color RED GREEN BLUE, -mc RED GREEN BLUE
                        Face masks color, in RGB.
  --remove-duplicates, -rd
                        Try to remove duplicated detections.
  --faces FACES, -f FACES
                        Specify the number of faces that will appear in the video. Only the X most probable detections
                        will be anonymized.
  --no-audio            Don't keep the audio from the input video.
  --visualize           Draw scores and boxes. This argument is only compatible with {input|output|threshold|no-audio}
                        arguments.
```

## Acknowledgments

- [poetry](https://github.com/python-poetry/poetry) — licensed under [MIT License](https://github.com/python-poetry/poetry/blob/master/LICENSE)
- [detectron2](https://github.com/facebookresearch/detectron2) — licensed under [Apache License 2.0](hhttps://github.com/facebookresearch/detectron2/blob/main/LICENSE)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) — licensed under [Apache License 2.0](https://github.com/kkroening/ffmpeg-python/blob/master/LICENSE)
- [opencv-python](https://github.com/opencv/opencv-python) — licensed under [MIT License](https://github.com/opencv/opencv-python/blob/master/LICENSE.txt)
- [numpy](https://github.com/numpy/numpy) — licensed under [BSD 3-Clause License](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- [colorama](https://github.com/tartley/colorama) — licensed under [BSD 3-Clause License](https://github.com/tartley/colorama/blob/master/LICENSE.txt)
- [tqdm](https://github.com/tqdm/tqdm) — licensed under [multiple licenses](https://github.com/tqdm/tqdm/blob/master/LICENCE)
- Parasite — © Barunson Entertainment & Arts Corp and © CJ ENM.
