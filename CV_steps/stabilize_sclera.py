from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np



# ── Feature detection helpers ──────────────────────────────────────────────────

def _build_interior_mask(gray_frame: np.ndarray, min_edge_dist: int = 50) -> np.ndarray:
    """
    Return a uint8 mask that keeps only pixels whose nearest background pixel
    is at least `min_edge_dist` away.  This prevents features being detected
    right on the sclera boundary where they are least stable.
    """
    # default which works OK
    # binary = cv2.threshold(gray_frame, 1, 255, cv2.THRESH_BINARY)[1]
    # dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # return (dist > min_edge_dist).astype(np.uint8) * 255

    # try getting only the center of the mask
    bg_clr = cv2.mean(gray_frame)[0]
    binary = cv2.threshold(gray_frame, bg_clr + 1, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("binary", binary)
    # cv2.waitKey(1)

    # NOrmalized masked cross correlation would probably be good (scikit image)
    
    # Phase cross corrlation

    # li thresholding

    # ORB features
    return binary
    

def _detect_features(
    gray_frame: np.ndarray,
    mask: np.ndarray,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: int = 30,
    block_size: int = 3,
) -> Optional[np.ndarray]:
    """
    Detect Shi-Tomasi corner features inside `mask`.
    Returns a float32 array of shape (N, 1, 2), or None if none are found.
    """
    features = cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        mask=mask,
    )
    if features is None or len(features) == 0:
        return None
    return features.astype(np.float32)


# ── Per-frame stabilization ────────────────────────────────────────────────────

def stabilize_frame(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    curr_frame: np.ndarray,
    min_edge_dist: int = 50,
    ransac_thresh: float = 5.0,
) -> tuple[np.ndarray, bool]:
    """
    Estimate the homography that maps `curr_frame` back onto `prev_gray`
    using Lucas-Kanade optical flow + RANSAC.

    Returns
    -------
    stabilized : np.ndarray
        Warped version of `curr_frame`, or `curr_frame` unchanged if the
        transform could not be computed.
    success : bool
        True when a valid homography was found.
    """
    h, w = curr_frame.shape[:2]

    mask     = _build_interior_mask(prev_gray, min_edge_dist)
    features = _detect_features(prev_gray, mask)
    cv2.imwrite("debug_frame.png", curr_frame)  # save the mask for debugging
    # during debug draw the features on the mask and save it
    if DEBUG:
        debug_img = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
        for pt in features:
            x, y = pt.ravel()
            cv2.circle(debug_img, (int(x), int(y)), 3, (0, 255, 0), -1)
        # shrink debug_img for display
        debug_img = cv2.resize(debug_img, (w // 2, h // 2))
        cv2.imshow("Features", debug_img)
        cv2.waitKey(1)


    if features is None:
        return curr_frame, False

    old_features = features.copy()
    new_features, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, old_features, None
    )

    good_old = old_features[status == 1]
    good_new = new_features[status == 1]

    if len(good_new) < 4:
        return curr_frame, False

    matrix, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, ransac_thresh)
    if matrix is None:
        return curr_frame, False

    stabilized = cv2.warpPerspective(curr_frame, matrix, (w, h))
    return stabilized, True


# ── Video pipeline ─────────────────────────────────────────────────────────────

def stabilize_video(
    video_path: str | Path,
    output_path: str | Path,
    min_edge_dist: int = 50,
    ransac_thresh: float = 5.0,
) -> None:
    """
    Stabilize every frame of `video_path` using sclera-interior optical flow
    and write the result to `output_path`.

    The input is expected to be a sclera-mask or sclera-overlay video where
    the sclera region is bright and the background is black.  Features are
    detected only in the interior of the sclera (away from its boundary) so
    that tracking is not confused by the mask edge.

    Parameters
    ----------
    video_path      : path to the source video
    output_path     : path for the stabilized output video (mp4)
    min_edge_dist   : minimum pixel distance from the sclera boundary for
                      feature detection  (default 50)
    ransac_thresh   : reprojection-error threshold for RANSAC homography
                      estimation  (default 5.0)
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    border = int(max(min_edge_dist * w, min_edge_dist * h))
    if fps <= 0:
        fps = 30.0

    print(f"Processing video : {video_path}")
    print(f"Frames: {n_frames}, Resolution: {w}x{h}, FPS: {fps}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    print(f"Output           : {output_path}")

    # ── seed with first frame ──────────────────────────────────────────────────
    ret, first = cap.read()
    if not ret:
        cap.release()
        writer.release()
        raise IOError("Could not read the first frame.")

    writer.write(first)
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    fail_count = 0

    for i in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: could not read frame {i}, stopping early.")
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        stabilized, success = stabilize_frame(
            prev_gray,
            curr_gray,
            frame,
            min_edge_dist=border,
            ransac_thresh=ransac_thresh,
        )

        if not success:
            fail_count += 1

        writer.write(stabilized)
        prev_gray = curr_gray

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{n_frames} frames processed")

    cap.release()
    writer.release()

    print(f"Stabilization complete!  ({fail_count} frames fell back to original)")
    print(f"Stabilized video saved to {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stabilize a sclera mask / overlay video using interior optical flow "
            "and RANSAC homography estimation."
        )
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input sclera mask or overlay video.",
    )
    parser.add_argument(
        "--output",
        default="output/stabilized/stabilized.mp4",
        help="Path to save the stabilized output video. (default: output/stabilized/stabilized.mp4)",
    )
    parser.add_argument(
        "--min-edge-dist",
        type=float,
        default=.05,
        help="Minimum pixel distance from the sclera boundary for feature detection. (default: 50)",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=5.0,
        help="RANSAC reprojection-error threshold for homography estimation. (default: 5.0)",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        # action="store_true",
        help="Show intermediate feature detection results for debugging.",
    )
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    global DEBUG
    DEBUG = args.debug


    stabilize_video(
        video_path=args.video,
        output_path=args.output,
        min_edge_dist=args.min_edge_dist,
        ransac_thresh=args.ransac_thresh,
    )


# Example commands:
#   python CV_steps/stabilize_sclera.py --video output/testing_sclera/sclera_mask_ML.mp4
#   python CV_steps/stabilize_sclera.py --video output/testing_sclera/sclera_mask_ML.mp4 --output output/testing_sclera/mask_sclera_stabilized.mp4 --min-edge-dist .03 --debug True