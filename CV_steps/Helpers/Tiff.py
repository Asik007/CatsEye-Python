import os

import cv2
# import numpy as np
from pathlib import Path
from typing import Optional


def export_tiff_stack(
    input_path: str,
    output_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> str:
    """Export selected frames to a single multi-page TIFF stack."""
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    s_frame = start_frame
    e_frame = end_frame

    os.makedirs(Path(output_path), exist_ok=True)
    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
        current = s_frame
        while True:
            if e_frame is not None and current >= e_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            current += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError("No frames were read from the input video")

    if not cv2.imwritemulti(output_path, frames):
        raise RuntimeError("Could not write multi-page TIFF stack")
    print(f"TIFF stack with {len(frames)} frames saved to {output_path}")
    return output_path

