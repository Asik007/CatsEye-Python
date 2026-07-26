"""
Isolate pipeline: Dispatcher for sclera isolation using either classical (IP) or ML-based methods.

This module provides a unified CLI interface to process images and videos with sclera
isolation, supporting both traditional image processing (Sclera_IP) and YOLO-based ML
segmentation (Sclera_ML).

Usage:
    # ML-based video processing
    python CV_steps/Isolate/pipeline.py --mode ml --video uploads/video.mp4

    # Classical IP-based video processing
    python CV_steps/Isolate/pipeline.py --mode ip --video uploads/video.mp4

    # Single image with ML
    python CV_steps/Isolate/pipeline.py --mode ml --image uploads/frame.jpg
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# from .Sclera_IP import process_eye_pipeline
from .Sclera_INFER import load_segmentation_model, process_image
from ..inclass import ProcessingConfig


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────


def process_image_ml(
    image_path: str,
    model_path: str,
    target_class: Optional[str] = "Eye",
    conf: float = 0.25,
    imgsz: int = 640,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a single image using ML-based segmentation (Sclera_ML).

    Args:
        image_path: Path to input image
        model_path: Path to YOLO model
        target_class: Class name to segment (None for all)
        conf: Confidence threshold
        imgsz: Inference image size

    Returns:
        (mask, overlay) tuple
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    model = load_segmentation_model(model_path)
    return process_image(
        image_bgr=image_bgr,
        model=model,
        target_class=target_class,
        conf=conf,
        imgsz=imgsz,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_video_ml(
        config: ProcessingConfig
) -> tuple[list[int], list[np.ndarray]]:
    
    
    """
    Process video using ML-based segmentation (Sclera_ML).

    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        output_mask_path: Output path for mask video
        output_overlay_path: Output path for overlay video
        target_class: Class name to segment (None for all)
        conf: Confidence threshold
        imgsz: Inference image size
    """
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {config.video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bad_frames = []
    mask_sizes = []
    print(f"Processing video: {config.video_path}")
    print(f"Frames: {n_frames}, Resolution: {w}x{h}, FPS: {fps}")


    model = load_segmentation_model(config.model_path)
    print(f"Model loaded from: {config.model_path}")


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if config.sclera_mask_path == "":
        config.sclera_mask_path = config.output_dir + r"\sclera_mask.mp4"

    mask_writer = None
    if config.sclera_mask_path is not None:
        mask_writer = cv2.VideoWriter(str(config.sclera_mask_path), fourcc, fps, (w, h), isColor=False)
        print(f"Mask output: {config.sclera_mask_path}")

    overlay_writer = None
    if config.sclera_overlay_path == "":
        config.sclera_overlay_path = config.output_dir + r"\sclera_overlay.mp4"

    if config.sclera_overlay_path is not None:
        overlay_writer = cv2.VideoWriter(str(config.sclera_overlay_path), fourcc, fps, (w, h))
        print(f"Overlay output: {config.sclera_overlay_path}")


    mask_stack = []

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i}, stopping early.")
            break

        # print(f" Processing frame {i}")
        mask, overlay, boxes = process_image(
            image_bgr=frame,
            model=model,
            target_class="Eye",
            # conf=conf,
            # imgsz=imgsz,
        )
        if mask is None:
            bad_frames.append(i)
            print(f"Warning: Frame {i} produced no mask, skipping.")
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            boxes = np.zeros((0, 4), dtype=np.int32)
            continue
        if mask_writer is not None:
            mask_writer.write(mask)
        if overlay_writer is not None:
            overlay_writer.write(overlay)

        # TODO: Theres a numpy function to flatten the boxes to something normal
        mask_sizes.append(boxes)
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{n_frames} frames processed")

    cap.release()
    if mask_writer is not None:
        mask_writer.release()
        print("release writer")
    if overlay_writer is not None:
        overlay_writer.release()

    print("Video processing complete!")

    return bad_frames, mask_sizes


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────