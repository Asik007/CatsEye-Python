import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

DEBUG = True


def _calc_transform(args) -> np.ndarray:
    """Calculate affine transform between two consecutive grayscale frames."""
    prev_gray, cur_gray = args
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    mask = cv2.threshold(prev_gray, 1, 255, cv2.THRESH_BINARY)[1]
    # somehow shrink the mask

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) # calculate distance from edge for each pixel
    mask = (dist > 300).astype(np.uint8) * 255  # keep only pixels >50px from any edge

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.1, minDistance=50, blockSize=3, mask=mask)

    if prev_pts is None or len(prev_pts) == 0:
        return np.array([0.0, 0.0, 0.0])

    # add a debug to show the detected features
    if DEBUG:
        debug_img = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
        for pt in prev_pts:
            x, y = pt.ravel()
            cv2.circle(debug_img, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.imshow("Features", debug_img)
        cv2.waitKey(1)

    cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, prev_pts, None, **lk_params
    )

    idx = status.ravel() == 1
    prev_good, cur_good = prev_pts[idx], cur_pts[idx]

    if len(prev_good) < 4:
        return np.array([0.0, 0.0, 0.0])

    M, _ = cv2.estimateAffinePartial2D(prev_good, cur_good)
    if M is None:
        return np.array([0.0, 0.0, 0.0])

    return np.array([M[0, 2], M[1, 2], np.arctan2(M[1, 0], M[0, 0])])


def _apply_transform(args) -> np.ndarray:
    """Apply a corrective affine warp to a single frame."""
    frame, correction, size = args
    dx, dy, da = correction
    w, h = size

    cos_a, sin_a = np.cos(da), np.sin(da)
    M = np.array([
        [cos_a, -sin_a, dx],
        [sin_a,  cos_a, dy],
    ], dtype=np.float32)

    border = cv2.BORDER_CONSTANT
    stabilized = cv2.warpAffine(
        frame, M, (w, h),
        borderMode=border,
        borderValue=(0, 0, 0),
    )

    return stabilized


def _moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    padded = np.pad(curve, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def stabilize_video(
    input_path: str,
    output_path: str,
    smoothing_radius: int = 50,
) -> None:
    """
    Stabilize a shaky video and write the result to disk.

    Args:
        input_path:       Path to the source video file.
        output_path:      Path where the stabilized video will be saved.
        smoothing_radius: Radius (in frames) of the rolling-average smoother.
        border_mode:      How to fill borders — 'black', 'reflect', or 'crop'.
    """
    # ── 1. Read all frames once ───────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[stabilize] {n_frames} frames  |  {w}×{h}  |  {fps:.2f} fps")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        raise RuntimeError("Video has fewer than 2 readable frames.")
    n_frames = len(frames)


    # ── 2. Calculate transforms (sequential — each frame depends on previous) ─
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    print("[stabilize] Calculating frame transforms …")
    transforms = np.zeros((n_frames - 1, 3), dtype=np.float64)
    for i, result in enumerate(map(
        _calc_transform,
        zip(gray_frames[:-1], gray_frames[1:])
    )):
        transforms[i] = result
        if (i + 1) % 100 == 0 or i == n_frames - 2:
            print(f"[stabilize] Analysed {i + 1}/{n_frames - 1} frame pairs …")

    # ── 3. Smooth trajectory ──────────────────────────────────────────────────

    trajectory  = np.cumsum(transforms, axis=0)
    corrections = np.vstack([-trajectory, [[0.0, 0.0, 0.0]]])  # last frame no-op

    # ── 5. Apply transforms in parallel ──────────────────────────────────────
    print("[stabilize] Applying corrections …")
    workers   = os.cpu_count()
    chunksize = max(1, n_frames // (workers * 4))

    args = [
        (frame, corrections[i], (w, h))
        for i, frame in enumerate(frames)
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        stabilized_frames = list(executor.map(_apply_transform, args,
                                              chunksize=chunksize))

    # ── 6. Write output ───────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for i, frame in enumerate(stabilized_frames):
        out.write(frame)
        if (i + 1) % 100 == 0 or i == n_frames - 1:
            print(f"[stabilize] Written {i + 1}/{n_frames} frames …")
    out.release()

    print(f"[stabilize] Done → {output_path}")

# ------------------------------------------------------------------ #
# CLI entry-point                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    testing_out = os.path.join("output", "jupyter_test")
    # os.makedirs(testing_out, exist_ok=True)
    # print(f"Testing output directory: {testing_out}")

    # outlined_path = os.path.join(testing_out, "sclera_outline.mp4")
    mask_path = os.path.join(testing_out, "sclera_mask.mp4")
    print(f"Input video for stabilization: {mask_path}")
    # import argparse

    # parser = argparse.ArgumentParser(description="Stabilize a shaky video with OpenCV.")
    # parser.add_argument("input_path",  help="Path to the input video")
    # parser.add_argument("output_path", help="Path for the stabilized output video")
    # parser.add_argument(
    #     "--smoothing-radius", type=int, default=50,
    #     help="Rolling-average window radius in frames (default: 50)",
    # )
    # parser.add_argument(
    #     "--border-mode", choices=["crop", "black", "reflect"], default="crop",
    #     help="Edge-fill strategy after warping (default: crop)",
    # )
    # args = parser.parse_args()

    stabilize_video(
        input_path      = mask_path,
        output_path     = os.path.join(testing_out, "stabilized_output.mp4"),
        smoothing_radius= 10,
    )