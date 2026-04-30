import argparse
import cv2
import numpy as np
import os
import csv


def _select_roi(first_img: np.ndarray):
    """
    Show the first frame and let the user draw an ROI rectangle.
    Returns (x, y, w, h), template crop, and center point.
    """
    window_name = "Select ROI  –  Enter / Space to confirm"
    shrink = 0.5
    display_img = cv2.resize(first_img, (0, 0), fx=shrink, fy=shrink)
    roi = cv2.selectROI(window_name, display_img, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(window_name)
    print(f"Selected ROI (on displayed image): {roi}")
    # x, y, w, h = map(int, roi)
    # scale back up to original image coordinates
    x,y,w,h = (np.array(roi) / shrink).astype(int)
    # x, y, w, h = scaled_roi

    if w <= 0 or h <= 0:
        raise ValueError("No ROI selected.")
    template = first_img[y : y + h, x : x + w].copy()
    center   = (x + w // 2, y + h // 2)
    return (x, y, w, h), template, center

def _render_tracking_video(
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

def _render_stabilized_video(
    video_path: str,
    tracked_points: list[dict],
    output_path: str,
) -> None:

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for idx in range(n):
        ret, frame = cap.read()
        if not ret:
            break

        del_x, del_y = tracked_points[idx]["displacement"]
        #  shift the frame in the opposite direction of the displacement to stabilize
        M = np.float32([[1, 0, -del_x], [0, 1, -del_y]])
        frame = cv2.warpAffine(frame, M, (w, h))
        
        writer.write(frame)

        if (idx + 1) % 50 == 0:
            print(f"  [render]   {idx + 1}/{n} frames written")

    cap.release()
    writer.release()

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

def _save_tracking_csv(tracked_points: list[dict], output_path: str) -> None:
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

def xCorr_pipeline(
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

    Returns a dict with paths to all outputs and summary counts.
    """

    # ensure output directory exists
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
    roi, template, origin_center = _select_roi(first_frame)
    print(f"  ROI  : x={roi[0]}  y={roi[1]}  w={roi[2]}  h={roi[3]}")
    print(f"  Center : {origin_center}")
    
    # ── 2. Cross-correlation tracking ─────────────────────────────────────────
    print("\n► Tracking ROI across all frames (cross-correlation)…")
    tracked_points = track_with_cross_correlation(video_path, roi, template, origin_center)
    print(f"  Tracked {len(tracked_points)} frames.")
    
    # ── 3. Motion-tracking video ───────────────────────────────────────────────
    tracking_video = os.path.join(output_dir, "motion_tracking.mp4")
    print(f"\n► Rendering motion-tracking video…\n  → {tracking_video}")
    _render_tracking_video(video_path, tracked_points, tracking_video, roi)
    
    # stabilize the video based on the tracked points
    stabilized_video = os.path.join(output_dir, "sclera_stabilized_XC.mp4")
    print(f"\n► Rendering stabilized video…\n  → {stabilized_video}")
    _render_stabilized_video(video_path, tracked_points, stabilized_video)

    # CSV
    # csv_path = os.path.join(output_dir, "tracking_results.csv")
    # _save_tracking_csv(tracked_points, csv_path)
    # print(f"\n  Tracking CSV → {csv_path}")
    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track an ROI via cross-correlation and render outputs."
    )
    parser.add_argument(
        "--video",
        default="output/testing_sclera/sclera_overlay_ML.mp4",
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/testing_sclera/",
        help="Directory to save outputs (motion video + CSV).",
    )
    # parser.add_argument(
    #     "--smooth-radius",
    #     type=int,
    #     default=50,
    #     help="Smoothing radius (currently unused).",
    # )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    xCorr_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        # smooth_radius=args.smooth_radius,
    )


