# CatsEye — Sclera & Vessel Tracking Pipeline

A lightweight computer-vision pipeline for isolating and stabilising the sclera (white of the eye) and extracting vessel / motion outputs from video. The project combines a small ML-based sclera/mask extractor with classical tracking and optical-flow stabilisation to produce visual overlays, masks, and stabilised videos suitable for downstream analysis.

## What it does
- Isolates the sclera region and produces a binary mask and overlay video.
- Stabilises the extracted sclera frames using optical-flow / homography methods.
- (Optional) Tracks motion for the selected ROI via cross-correlation (XCorr pipeline).
- Writes processed video outputs and summary CSVs to timestamped result folders.

## How it works (high level)
1. A lightweight ML model (`ML_stuff/best.pt`) segments the sclera and produces a mask and overlay video.
2. The overlay/mask video is analysed with cross-correlation or feature detection(does not work for the most part, but am trying) to compute frame-to-frame displacements.
3. The pipeline renders an overlay video and then stabilises frames using estimated motion (want to do translation and rotation but only translation with XCorrelation).
4. Outputs (videos and CSVs) are saved under an `output/results_YYYYMMDD-HHMMSS/` directory.

## Repository layout
- `new_pipeline.py` — Primary combined pipeline and CLI entry point.
- `guh.py` — Small test script for stabilising mask videos (example / debug).
- `CV_steps/` — Modular CV functions used by the pipeline:
	- `sclera_ML.py` — ML-based sclera/mask extraction and overlay rendering.
	- `XCorr.py` — Cross-correlation based motion-tracking pipeline.
	- `stabilize_frame.py` — Video stabilisation utilities.
	- other helpers: `render.py`, `vessel.py`, `stabilize_sclera.py`, etc.
- `ML_stuff/` — ML model and helpers (`best.pt` is the trained ultralytics model used by `sclera_ML`).
- `output/` — Example outputs and previously-run result folders.

## Requirements
- Python 3.8+ recommended.
- Key Python packages: `opencv-python`, `numpy`, `matplotlib`, `ultralytics` (for the YOLO model inference).

Example install (recommended in a virtualenv):

```bash
python -m pip install --upgrade pip
pip install opencv-python numpy matplotlib ultralytics
```

If you plan to use GPU acceleration for model inference, follow `ultralytics` documentation to install the appropriate CUDA-enabled PyTorch build.

## Quickstart / Usage

1. Place your source video in `uploads/` (or provide any path).
2. Run the combined pipeline (creates a timestamped results folder under `output/`):

```bash
python new_pipeline.py --video uploads/your_video.mp4 --output output
```

Notes:
- `new_pipeline.py` default behaviour is to run the ML mask/overlay step and then stabilise the overlay video. The script prints the produced file paths (e.g. `sclera_overlay.mp4`, `sclera_mask.mp4`, `stabilized.mp4`).
- Outputs will be placed in `output/results_YYYYMMDD-HHMMSS/`.

## Debug / Test script
Use `guh.py` to run a quick test that stabilises a mask video saved under `output/jupyter_test/`.

```bash
python guh.py
```

This script reads `output/jupyter_test/sclera_mask.mp4` (or adjust the paths inside the file) and writes a `mask_stabilized_test.mp4` file for inspection.

## Development notes
- The pipeline is modular — individual steps in `CV_steps/` can be imported and run separately for debugging.
- The ML model path is `ML_stuff/best.pt` by default; change paths or `conf/imgsz` parameters in `new_pipeline.py` when needed.
- Cross-correlation based tracking is implemented in `CV_steps/XCorr.py` and can be enabled / customised there.

## Outputs
- `sclera_overlay.mp4` — original frames with overlay/ROI visuals from the ML step.
- `sclera_mask.mp4` — binary mask video produced by the ML model.
- `stabilized.mp4` — result of stabilisation applied to the overlay/mask video.
- `tracking_results.csv` — (when enabled) per-frame tracking data produced by the XCorr tracking step.

## Next steps / tips
- If you want, I can add a `requirements.txt`, example command-line flags documentation (`--help` output capture), or a small demo script that runs the pipeline on a provided sample video and commits results.


