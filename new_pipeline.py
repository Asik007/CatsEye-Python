import cv2
import os
import sys

from line_profiler import profile

import matplotlib.pyplot as plt

import numpy as np

from CV_steps.XCorr import xCorr_pipeline
from CV_steps.stabilize import stabilize_video
from CV_steps.sclera import sclera_pipeline
import CV_steps.sclera_ML as sclera_ML



# ──────────────────────────────────────────────────────────────────────────────
# ROI Selection & Cross-Correlation Tracking
# ──────────────────────────────────────────────────────────────────────────────









# ──────────────────────────────────────────────────────────────────────────────
# Combined Pipeline
# ──────────────────────────────────────────────────────────────────────────────


@profile
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





    # Show histogram of the saturation and brightness channels average over the entire video
    # hsv_data = np.load("hsv_data.npy")
    # s = hsv_data[:, :, 1]
    # v = hsv_data[:, :, 2]
    # plt.hist(s.ravel(), bins=256, range=(0, 1), color='blue', alpha=0.5, label='Saturation')
    # plt.hist(v.ravel(), bins=256, range=(0, 1), color='orange', alpha=0.5, label='Brightness')
    # plt.title("Saturation and Brightness Histograms")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.savefig("histograms.png")
    # plt.close()

    return {
        # "tracking_video":   tracking_video,
        "stabilized_video": stabilized_video,
        # "csv_path":         csv_path,
        # "frames_tracked":   len(tracked_points),
        "sclera_overlay":   overlay_path,
        "sclera_mask":      mask_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Playback helper
# ──────────────────────────────────────────────────────────────────────────────

def _resize_for_playback(frame: np.ndarray, shrink_percent: float = 0.0) -> np.ndarray:
    """Return a resized frame shrunk by `shrink_percent` for display only."""
    if shrink_percent <= 0:
        return frame
    if shrink_percent >= 100:
        raise ValueError("shrink_percent must be between 0 and 99.99.")

    scale = 1.0 - (shrink_percent / 100.0)
    h, w = frame.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def playback_video(
    path: str,
    window_title: str = "Playback  |  Q to close",
    shrink_percent: float = 0.0,
) -> None:
    cap   = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000 / fps))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_to_show = _resize_for_playback(frame, shrink_percent=shrink_percent)
        cv2.imshow(window_title, frame_to_show)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def playback_image(
    image_path: str,
    window_title: str = "Image  |  Q to close",
    shrink_percent: float = 0.0,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Cannot open image: {image_path}")

    image_to_show = _resize_for_playback(image, shrink_percent=shrink_percent)
    cv2.imshow(window_title, image_to_show)
    while True:
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

# just for testing how long it takes
import time

def _run_cli() -> None:
    cwd = os.getcwd()
    print(f"Working directory: {cwd}\n")

    def _clean(p: str) -> str:
        p = os.path.expanduser(p.strip().strip('"').strip("'"))
        return os.path.abspath(p if os.path.isabs(p) else os.path.join(cwd, p))

    def _yn(prompt: str, default: bool = True) -> bool:
        suffix = " [Y/n]: " if default else " [y/N]: "
        raw = input(prompt + suffix).strip().lower()
        return default if not raw else raw in {"y", "yes"}

    # Source video
    while True:
        default_video = os.path.join(cwd, "uploads\\IMG_1702.mov")
        raw = input(f"Source video path [{default_video}]: ").strip()
        video_path = _clean(raw) if raw else default_video
        if os.path.isfile(video_path):
            break
        print("  File not found — try again.")

    # Output folder
    default_out = os.path.join(cwd, "output")
    raw = input(f"Output folder [{default_out}]: ").strip()
    output_dir = _clean(raw) if raw else default_out

    output_dir = os.path.join(output_dir, "results_" + time.strftime("%Y%m%d-%H%M%S"))

    # Smooth radius
    raw = input("Stabilisation smooth radius [50]: ").strip()
    smooth_radius = int(raw) if raw.isdigit() else 50

    # Playback preference
    show_playback = _yn("Play back result videos when done?", default=False)
    if not show_playback:
        cv2.imshow  = lambda *a, **k: None          # type: ignore[assignment]
        cv2.waitKey = lambda _=0: ord("q")          # type: ignore[assignment]

    # Playback resize preference
    raw = input("Shrink playback windows by percent [0]: ").strip()
    try:
        shrink_percent = float(raw) if raw else 0.0
    except ValueError:
        shrink_percent = 0.0
    shrink_percent = min(max(shrink_percent, 0.0), 99.99)

    # Run
    start = time.perf_counter()
    result = process_and_stabilize(video_path, output_dir, smooth_radius=smooth_radius)

    print("\n✓ All done.")
    # print(f"  Motion-tracking video : {result['tracking_video']}")
    print(f"  Stabilised video      : {result['stabilized_video']}")
    # print(f"  Tracking CSV          : {result['csv_path']}")
    # print(f"  Frames tracked        : {result['frames_tracked']}")
    end = time.perf_counter()
    print(f"  Total processing time : {end - start:.2f} seconds")
    #save the stabilised, and motiontracking videos and csv to the output folder
    # os.makedirs(output_dir, exist_ok=True)

    if show_playback:
        print("\n► Playing motion-tracking video  (Q to skip)…")
        playback_video(
            result["tracking_video"],
            "Motion Tracking  |  Q to close",
            shrink_percent=shrink_percent,
        )
        print("► Playing stabilised video  (Q to skip)…")
        playback_video(
            result["stabilized_video"],
            "Stabilised  |  Q to close",
            shrink_percent=shrink_percent,
        )


if __name__ == "__main__":
    if sys.stdin.isatty() and sys.stdout.isatty():
        _run_cli()
    else:
        print("Run this script from a terminal to use the interactive CLI.")