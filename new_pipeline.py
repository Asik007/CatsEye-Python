import argparse

import os
import time


# from line_profiler import profile

import matplotlib.pyplot as plt

import numpy as np


from CV_steps.registration import xCorr_pipeline, xCorr_pipeline_debug
from CV_steps.stabilize_frame import stabilize_video
from CV_steps.sclera_IP import sclera_pipeline
import CV_steps.sclera_ML as sclera_ML


# ──────────────────────────────────────────────────────────────────────────────
# Combined Pipeline
# ──────────────────────────────────────────────────────────────────────────────


# @profile
def process_and_stabilize(
    video_path: str,
    output_dir: str,
    smooth_radius: int = 50,
    ) -> dict:
    
    """
    Full pipeline:

      1. Open the video and show the first frame for ROI selection.
         The chosen ROI is used as the cross-correlation template.
      2. Track the ROI across every frame → displacement / motion data.
      3. Render a *motion-tracking video* that overlays the trail,
         moving ROI box, and per-frame displacement on the original footage.
      4. Stabilise the original video with optical-flow stabilisation.
      5. Save tracking data to CSV.

    Returns a dict with paths to all outputs and summary counts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # isolated_video = os.path.join(output_dir, "sclera_isolated.mp4")
    overlay_path = os.path.join(output_dir, "sclera_overlay.mp4")
    mask_path = os.path.join(output_dir, "sclera_mask.mp4")
    sclera_ML.process_video( video_path=video_path, output_mask_path=mask_path, output_overlay_path=overlay_path,model_path="ML_stuff/best.pt", conf=0.25, imgsz=512)

    # XCorr tracking + video render
    # Each on of these steps opens the video independently, but each step must come after the previous one finishes to ensure the video file is not being accessed by multiple processes at once.

    # tracking_video = os.path.join(output_dir, "motion_tracking.mp4")
    # xCorr_pipeline(video_path, output_dir, smooth_radius=smooth_radius)
    # print(f"\n  Motion-tracking video → {tracking_video}")
    # print(f"\n  csv tracking data → {output_dir + '/tracking_results.csv'}")
    
    # ── 6. Extract Sclera ────────────────────────────────────────────────────────────────

    # sclera_pipeline(video_path, overlay_path, mask_path)
    print("\n  Sclera overlay video → " + overlay_path)
    print("\n  Sclera mask video → " + mask_path)



    # ── 4. Stabilisation ──────────────────────────────────────────────────────
    stabilized_video = os.path.join(output_dir, "stabilized.mp4")
    print(f"\n► Stabilising video (smooth_radius={smooth_radius})…\n  → {stabilized_video}")
    xCorr_pipeline(overlay_path, stabilized_video)

    return {
        # "tracking_video":   tracking_video,
        "stabilized_video": stabilized_video,
        # "csv_path":         csv_path,
        # "frames_tracked":   len(tracked_points),
        "sclera_overlay":   overlay_path,
        "sclera_mask":      mask_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

# just for testing how long it takes


def _run_cli() -> None:
    cwd = os.getcwd()

    parser = argparse.ArgumentParser(description="Stabilise a video file.")
    parser.add_argument("--video",    default=os.path.join(cwd, "uploads", "output_001.mp4"), help="Path to source video.")
    parser.add_argument("--output",   default=os.path.join(cwd, "output"),                    help="Base output directory.")
    parser.add_argument("--debug",   default=False, type=bool,                                help="Smoothing radius for stabilisation (in pixels).")
    args = parser.parse_args()

    output_dir = os.path.join(args.output, "results_" + time.strftime("%Y%m%d-%H%M%S"))

    print(f"\n► Processing video: {args.video}")
    print(f" Debug Pipeline: {args.debug}")
    start = time.perf_counter()
    result = process_and_stabilize(args.video, output_dir)
    elapsed = time.perf_counter() - start

    print("✓ All done.")
    print(f"  Stabilised video      : {result['stabilized_video']}")
    print(f"  Total processing time : {elapsed:.2f} seconds")

if __name__ == "__main__":
    _run_cli()

# command to get the same output as the CLI but without the CLI interface, just for testing how long it takes
# if __name__ == "__main__":
#     video = os.path.join(os.getcwd(), "uploads", "output_001.mp4")
#     output = os.path.join(os.getcwd(), "output")
#     radius = 50
#     output_dir = os.path.join(output, "results_" + time.strftime("%Y%m%d-%H%M%S"))

#     start = time.perf_counter()
#     result = process_and_stabilize(video, output_dir, smooth_radius=radius)
#     elapsed = time.perf_counter() - start

#     print("\n✓ All done.")
#     print(f"  Stabilised video      : {result['stabilized_video']}")
#     print(f"  Total processing time : {elapsed:.2f} seconds")