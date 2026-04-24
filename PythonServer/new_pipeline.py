import cv2
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import uniform_filter1d


# ──────────────────────────────────────────────────────────────────────────────
# ROI Selection & Cross-Correlation Tracking
# ──────────────────────────────────────────────────────────────────────────────

def select_roi(first_img: np.ndarray):
    """
    Show the first frame and let the user draw an ROI rectangle.
    Returns (x, y, w, h), template crop, and center point.
    """
    window_name = "Select ROI  –  Enter / Space to confirm"
    roi = cv2.selectROI(window_name, first_img, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise ValueError("No ROI selected.")
    template = first_img[y : y + h, x : x + w].copy()
    center   = (x + w // 2, y + h // 2)
    return (x, y, w, h), template, center


def track_with_cross_correlation(
    video_path: str,
    roi: tuple,
    template: np.ndarray,
    origin_center: tuple,
) -> list[dict]:
    """
    Match `template` in every frame via TM_CCOEFF_NORMED.

    Returns a list of dicts:
        { frame, center, displacement, match_score }
    """
    _x, _y, roi_w, roi_h = roi
    cap      = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracked  = []

    for idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        result   = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)

        cx = max_loc[0] + roi_w // 2
        cy = max_loc[1] + roi_h // 2
        tracked.append({
            "frame":        idx,
            "center":       (cx, cy),
            "displacement": (cx - origin_center[0], cy - origin_center[1]),
            "match_score":  float(score),
        })

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [tracking] {idx + 1}/{n_frames} frames")

    cap.release()
    return tracked


def render_tracking_video(
    video_path: str,
    tracked_points: list[dict],
    output_path: str,
    roi: tuple,
) -> None:
    """
    Re-reads the original video and draws on every frame:
      • Green polyline trail of the tracked center
      • Colored ROI box at the current matched position
      • Current-center dot
      • Displacement (dx / dy) text in the top-left corner
    """
    _x, _y, roi_w, roi_h = roi
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trail: list[tuple[int, int]] = []

    for idx in range(n):
        ret, frame = cap.read()
        if not ret:
            break

        if idx < len(tracked_points):
            pt     = tracked_points[idx]
            center = tuple(pt["center"])
            dx, dy = pt["displacement"]
            trail.append(center)

            # --- draw trail ---
            if len(trail) > 1:
                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

            # --- draw moving ROI box ---
            tl = (center[0] - roi_w // 2, center[1] - roi_h // 2)
            br = (center[0] + roi_w // 2, center[1] + roi_h // 2)
            cv2.rectangle(frame, tl, br, color=(0, 200, 255), thickness=2)

            # --- center dot ---
            cv2.circle(frame, center, radius=6, color=(0, 255, 0), thickness=-1)

            # --- displacement label ---
            # label = f"dx={dx:+d}  dy={dy:+d}"
            # cv2.putText(
            #     frame, label, (10, 34),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
            # )

        writer.write(frame)

        if (idx + 1) % 50 == 0:
            print(f"  [render]   {idx + 1}/{n} frames written")

    cap.release()
    writer.release()


def save_tracking_csv(tracked_points: list[dict], output_path: str) -> None:
    """Save per-frame tracking data plus mean / std summary rows."""
    centers = np.array([p["center"] for p in tracked_points])
    mean    = centers.mean(axis=0)
    std     = centers.std(axis=0)

    fieldnames = ["frame", "center_x", "center_y", "disp_x", "disp_y", "match_score"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in tracked_points:
            w.writerow({
                "frame":       p["frame"],
                "center_x":   p["center"][0],
                "center_y":   p["center"][1],
                "disp_x":     p["displacement"][0],
                "disp_y":     p["displacement"][1],
                "match_score": p["match_score"],
            })
        w.writerow({"frame": "mean", "center_x": mean[0], "center_y": mean[1],
                    "disp_x": "", "disp_y": "", "match_score": ""})
        w.writerow({"frame": "std",  "center_x": std[0],  "center_y": std[1],
                    "disp_x": "", "disp_y": "", "match_score": ""})


# ──────────────────────────────────────────────────────────────────────────────
# Video Stabilisation  (optical-flow based — original, unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def stabilize_video(input_path: str, output_path: str, smooth_radius: int = 50) -> None:
    cap = cv2.VideoCapture(input_path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Pass 1: buffer frames + compute inter-frame transforms ---
    frames     = np.empty((n_frames, h, w, 3), dtype=np.uint8)
    transforms = np.empty((n_frames - 1, 3),   dtype=np.float64)  # dx, dy, angle

    ret, frames[0] = cap.read()
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, n_frames):
        ret, frames[i] = cap.read()
        if not ret:
            n_frames = i
            break
        curr_gray          = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        transforms[i - 1]  = _estimate_transform(prev_gray, curr_gray)
        prev_gray          = curr_gray

    cap.release()

    # --- Trajectory smoothing (vectorised) ---
    trajectory  = np.cumsum(transforms, axis=0)
    smoothed    = uniform_filter1d(trajectory, size=smooth_radius * 2 + 1, axis=0, mode="nearest")
    corrections = smoothed - trajectory

    # --- Pass 2: warp + write (CPU warps pipelined with disk writes) ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cos_a = np.cos(corrections[:, 2])
    sin_a = np.sin(corrections[:, 2])
    dx    = corrections[:, 0]
    dy    = corrections[:, 1]

    Ms = np.stack([                                      # (N-1, 2, 3)
        np.stack([ cos_a, -sin_a, dx], axis=1),
        np.stack([ sin_a,  cos_a, dy], axis=1),
    ], axis=1)

    writer.write(frames[0])

    write_queue = []
    write_lock  = threading.Semaphore(0)

    def _writer_thread():
        for _ in range(n_frames - 1):
            write_lock.acquire()
            writer.write(write_queue.pop(0))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_writer_thread)
        for i in range(n_frames - 1):
            stabilized = cv2.warpAffine(
                frames[i + 1], Ms[i], (w, h), borderMode=cv2.BORDER_REFLECT
            )
            write_queue.append(stabilized)
            write_lock.release()
        future.result()

    writer.release()


def _estimate_transform(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    prev_pts             = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                                   qualityLevel=0.05, minDistance=30)
    curr_pts, status, _  = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    mask                 = status.ravel() == 1
    M, _                 = cv2.estimateAffinePartial2D(prev_pts[mask], curr_pts[mask])
    return M[0, 2], M[1, 2], 0.0                        # dx, dy, no rotation


# ──────────────────────────────────────────────────────────────────────────────
# Combined Pipeline
# ──────────────────────────────────────────────────────────────────────────────

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

    # ── 1. First frame → ROI selection ────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("Could not read first frame from video.")

    print("► Select an ROI on the first frame, then press Enter / Space to confirm.")
    roi, template, origin_center = select_roi(first_frame)
    print(f"  ROI  : x={roi[0]}  y={roi[1]}  w={roi[2]}  h={roi[3]}")
    print(f"  Center : {origin_center}")

    # ── 2. Cross-correlation tracking ─────────────────────────────────────────
    print("\n► Tracking ROI across all frames (cross-correlation)…")
    tracked_points = track_with_cross_correlation(video_path, roi, template, origin_center)
    print(f"  Tracked {len(tracked_points)} frames.")

    # ── 3. Motion-tracking video ───────────────────────────────────────────────
    tracking_video = os.path.join(output_dir, "motion_tracking.mp4")
    print(f"\n► Rendering motion-tracking video…\n  → {tracking_video}")
    render_tracking_video(video_path, tracked_points, tracking_video, roi)

    # ── 4. Stabilisation ──────────────────────────────────────────────────────
    stabilized_video = os.path.join(output_dir, "stabilized.mp4")
    print(f"\n► Stabilising video (smooth_radius={smooth_radius})…\n  → {stabilized_video}")
    stabilize_video(video_path, stabilized_video, smooth_radius=smooth_radius)

    # ── 5. CSV ────────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "tracking_results.csv")
    save_tracking_csv(tracked_points, csv_path)
    print(f"\n  Tracking CSV → {csv_path}")

    return {
        "tracking_video":   tracking_video,
        "stabilized_video": stabilized_video,
        "csv_path":         csv_path,
        "frames_tracked":   len(tracked_points),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Playback helper
# ──────────────────────────────────────────────────────────────────────────────

def playback_video(path: str, window_title: str = "Playback  |  Q to close") -> None:
    cap   = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000 / fps))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(window_title, frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    cap.release()
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
        video_path = _clean(input("Source video path: "))
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
    show_playback = _yn("Play back result videos when done?", default=True)
    if not show_playback:
        cv2.imshow  = lambda *a, **k: None          # type: ignore[assignment]
        cv2.waitKey = lambda _=0: ord("q")          # type: ignore[assignment]

    # Run
    start = time.perf_counter()
    result = process_and_stabilize(video_path, output_dir, smooth_radius=smooth_radius)

    print("\n✓ All done.")
    print(f"  Motion-tracking video : {result['tracking_video']}")
    print(f"  Stabilised video      : {result['stabilized_video']}")
    print(f"  Tracking CSV          : {result['csv_path']}")
    print(f"  Frames tracked        : {result['frames_tracked']}")
    end = time.perf_counter()
    print(f"  Total processing time : {end - start:.2f} seconds")
    #save the stabilised, and motiontracking videos and csv to the output folder
    # os.makedirs(output_dir, exist_ok=True)

    if show_playback:
        print("\n► Playing motion-tracking video  (Q to skip)…")
        playback_video(result["tracking_video"], "Motion Tracking  |  Q to close")
        print("► Playing stabilised video  (Q to skip)…")
        playback_video(result["stabilized_video"], "Stabilised  |  Q to close")


if __name__ == "__main__":
    if sys.stdin.isatty() and sys.stdout.isatty():
        _run_cli()
    else:
        print("Run this script from a terminal to use the interactive CLI.")