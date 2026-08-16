# CatsEye — Sclera and Vessel Tracking Pipeline

A computer-vision pipeline for isolating and stabilising the sclera (white of the eye) and extracting vessel / motion outputs from video. The project combines a small ML-based sclera/mask extractor with classical tracking and stabilisation to produce visual overlays, masks, and stabilised videos of blood vessels for downstream analysis.

## What it does
- Isolates the sclera region and produces a binary mask and overlay video.
- Stabilises the extracted sclera frames using ECC from OpenCV.
- Writes processed tiff outputs to timestamped result folders.

## How it works (high level)
1. A lightweight ML model (`ML_stuff/exports/model_640_False.onnx`) segments the sclera and produces a mask and overlay video.
2. The overlay/mask video is analysed with cross-correlation to compute frame-to-frame displacements.
3. The pipeline renders an overlay video and then stabilises frames using estimated motion.
4. Outputs (videos and CSVs) are saved under an `output/results_YYYYMMDD-HHMMSS/` directory.
5. Then a FR-UNet model (trained on DCA-1 dataset by the original creators of FR-UNet) isolates the vessels
6. each vessel is then analyzed and put into the output folder
## Repository layout

- `main.py` — main processing entry point, including `trim_process_stabilize()`
- `GUI_tk.py` — Tkinter desktop UI for selecting a video, model, and output directory
- `CLI.py` — lightweight CLI wrapper for running the processing pipeline
- `pyproject.toml` — project metadata and Python dependencies
- `CV_steps/` — core processing modules
  - `inclass.py` — `ProcessingConfig` dataclass
  - `registration.py` — registration utilities and rigid alignment helpers
  - `render.py` — rendering and video slicing helpers
  - `rollingball.py` — rolling-ball / CLAHE preprocessing
  - `metrics.py` — registration and pipeline metrics
  - `Helpers/` — crop, I/O, and dataset-related utility functions
  - `Isolate/` — segmentation and vessel isolation code
    - `pipeline.py` — ML isolation workflow
    - `YOLO_infer.py` — YOLO-style inference wrapper
    - `FRUnet_infer.py` — FR-UNet vessel model wrapper
    - `binary_cleaning.py` — binary mask cleanup / postprocessing
- `ML_stuff/` — trained and exported models, including YOLO and ONNX assets
  - `exports/model_640_False.onnx` is used as a default model path in the current pipeline
- `output/` — generated results, usually under a timestamped `results_YYYYMMDD-HHMMSS` directory
- `uploads/` — user video inputs and processing assets
- `VSX_stuff/` — additional model files and related resources

## Quickstart

### Install dependencies

```bash
uv sync
```

Or with plain pip:

```bash
pip install -e .
```

### Run the GUI

```bash
python GUI_tk.py
```

or, in the configured uv environment:

```bash
uv run GUI_tk.py
```



### Run the CLI

```bash
python CLI.py --video uploads/your_video.mp4 --output output
```

### Run the processing pipeline directly

`main.py` contains a direct pipeline call using `ProcessingConfig`:

```python
from CV_steps.inclass import ProcessingConfig
from main import trim_process_stabilize

config = ProcessingConfig(
    video_path=r"uploads\IMG_1745.MOV",
    output_dir=r"output\testing",
    best_frame=10,
    model_path=r"ML_stuff\exports\model_640_False.onnx",
)

trim_process_stabilize(config)
```

### Build exe with Nuitka

First install nuitka via pip

then run the following command

python -m nuitka GUI_tk.py --mode=standalone --enable-plugin=tk-inter --include-data-dir=./ML_stuff=ML_stuff --include-data-dir=./VSX_stuff=VSX_stuff


## Output conventions

The project writes timestamped outputs into directories named like:

```text
output/results_YYYYMMDD-HHMMSS/
```

Typical outputs include:

- `sclera_mask.mp4` — binary sclera mask video
- `sclera_overlay.mp4` — masked/overlay render of the processed video
- `clahe.tiff` — CLAHE-enhanced stacked image output
- `reg_stack.tiff` — registered stack
- `pred_stack.tiff` — FR-UNet prediction stack
- `bin_stack.tiff` — thresholded / binary vessel output

## Development notes

- The project is structured around `ProcessingConfig`, which centralizes video input, output directory, best-frame selection, and model path.
- The default model path in the code is currently an ONNX export under `ML_stuff/exports/` rather than the legacy `.pt` model path.
- Model inference is implemented in `CV_steps/Isolate/YOLO_infer.py` and dispatched via `CV_steps/Isolate/pipeline.py`.
- FR-UNet vessel extraction is handled separately in `CV_steps/Isolate/FRUnet_infer.py`.
- The registration and stabilisation pipeline lives primarily in the `CV_steps/` modules and is called from `main.py`.

## Current TODOs

- [x] Move the project tooling to uv
- [x] Evaluate Nuitka packaging
- [x] Use ONNX model assets in the current pipeline
- [ ] Add a Streamlit or browser-based UI

