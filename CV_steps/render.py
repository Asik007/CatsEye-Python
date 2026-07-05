import cv2
import numpy as np
from typing import List, Optional, Tuple


def find_seg(
        num_frames: int,
        bad_frames: List[int],
        best_frame_idx: Optional[int] = None,
        max_len: int = 50,
) -> Tuple[int, int]:
    """
    Divide the frame range into segments separated by bad frames,
    pick a segment ≤ max_len, preferring one containing best_frame_idx,
    otherwise the longest valid segment.
    """
    # Create segments between bad frames (bad frames act as walls)
    boundaries = [-1] + sorted(set(bad_frames)) + [num_frames]
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1] - 1
        if start <= end and (end - start + 1) <= max_len:
            segments.append((start, end))

    if not segments:
        raise ValueError("No valid video segments found (all are too long or empty).")

    # Prefer segment containing the requested best frame
    if best_frame_idx is not None:
        for seg in segments:
            if seg[0] <= best_frame_idx <= seg[1]:
                return seg
        # Warn if the best frame exists but lies in a segment that exceeds max_len
        print(
            f"Warning: best_frame_idx {best_frame_idx} is in a segment longer than "
            f"{max_len} frames. Falling back to the longest valid segment."
        )

    # Fallback: longest segment among those that satisfy max_len
    return max(segments, key=lambda s: s[1] - s[0] + 1)


def vid2seg(
        video_path: str,
        bad_frames: List[int],
        frame_sizes: List[Optional[Tuple[int, int, int, int]]],
        output_path: str,
        max_len: int = 50,
        best_frame_idx: Optional[int] = None,
        outlier_std: float = 1.0,
) -> Tuple[int, int]:
    """
    Extract a continuous segment (≤ max_len) from the video, avoiding bad frames
    and frames whose detected bounding box area deviates by more than outlier_std
    standard deviations from the mean area.

    Parameters:
        video_path: path to input video.
        bad_frames: list of indices already known to be bad.
        frame_sizes: for each frame, either None (no detection) or a tuple
                     (tl_x, tl_y, br_x, br_y).
        output_path: where to write the selected segment.
        max_len: maximum allowed length of the extracted segment.
        best_frame_idx: if given, prefer a segment containing this frame.
        outlier_std: number of standard deviations for area‑based outlier rejection.

    Returns:
        (start_frame, end_frame) of the extracted segment.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    bad_frames = set(bad_frames)
    frame_areas = {}  # i -> area, for frames that passed the basic checks

    for i, bbox in enumerate(frame_sizes):
        if i in bad_frames:
            continue
        if bbox is None or bbox.size <= 0:
            bad_frames.add(i)
            continue
        tl_x, tl_y, br_x, br_y = bbox[0]
        area = (br_x - tl_x) * (br_y - tl_y)
        if area <= 0:
            bad_frames.add(i)
            print(f"Warning: Frame {i} has invalid bounding box: {bbox} area: {area}, skipping.")
        else:
            frame_areas[i] = area

    if frame_areas:
        areas = np.array(list(frame_areas.values()))
        mean_area, std_area = areas.mean(), areas.std()
        lower_bound = mean_area - outlier_std * std_area
        upper_bound = mean_area + outlier_std * std_area

        for i, area in frame_areas.items():
            if area < lower_bound or area > upper_bound:
                bad_frames.add(i)

    bad_frames = list(bad_frames)
    areas = list(frame_areas.values())


    # Remove duplicates and sort once
    bad_frames = sorted(set(bad_frames))

    # ---- Find the best segment ----
    start_frame, end_frame = find_seg(
        num_frames=num_frames,
        bad_frames=bad_frames,
        best_frame_idx=best_frame_idx,
        max_len=max_len,
    )

    # ---- Write the segment ----
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Cannot create output video: {output_path}")

    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= frame_idx <= end_frame:
            out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"Segment saved: frames {start_frame}-{end_frame} to {output_path}")
    return start_frame, end_frame