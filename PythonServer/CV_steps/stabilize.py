
# ──────────────────────────────────────────────────────────────────────────────
# Video Stabilisation  (optical-flow based — original, unchanged)
# ──────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
import threading
from concurrent.futures import ThreadPoolExecutor



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

