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

from .Sclera_IP import process_eye_pipeline
from .Sclera_ML import load_segmentation_model, process_image as ml_process_image


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────


def process_image_ip(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a single image using classical image processing (Sclera_IP).

    Args:
        image_path: Path to input image

    Returns:
        (eye_mask, overlay) tuple
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return process_eye_pipeline(image=image)


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
    return ml_process_image(
        image_bgr=image_bgr,
        model=model,
        target_class=target_class,
        conf=conf,
        imgsz=imgsz,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────


def process_video_ip(
    input_path: str,
    overlay_path: str,
    mask_path: str,
    max_workers: int = 4,
    debug: bool = False,
) -> None:
    """
    Process video using classical image processing (Sclera_IP) with parallel workers.

    Args:
        input_path: Path to input video
        overlay_path: Output path for overlay video
        mask_path: Output path for mask video
        max_workers: Number of parallel workers
        debug: Enable debug output
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Processing video: {input_path}")
    print(f"Frames: {n_frames}, Resolution: {w}x{h}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    raw_frames = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i}, stopping early.")
            break
        raw_frames.append((i, frame))
    cap.release()

    def process(item):
        idx, frame = item
        if debug:
            import threading
            print(f"Processing frame {idx + 1}/{n_frames} in thread {threading.current_thread().name}")
        masked_img, outlined = process_eye_pipeline(image=frame)
        return idx, outlined, masked_img

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, item): item[0] for item in raw_frames}
        for future in as_completed(futures):
            idx, outlined, masked_img = future.result()
            results[idx] = (outlined, masked_img)

    if results and results[0][0] is not None:
        Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))
        for i in range(len(raw_frames)):
            outlined, _ = results[i]
            writer.write(outlined)
            print(f"Writing overlay: {i + 1}/{len(raw_frames)}", end="\r")
        writer.release()
        print(f"\nSaved overlay → {overlay_path}")

    Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(mask_path, fourcc, fps, (w, h))
    for i in range(len(raw_frames)):
        _, masked_img = results[i]
        writer.write(masked_img)
    writer.release()
    print(f"Saved mask    → {mask_path}")


def process_video_ml(
    video_path: str,
    model_path: str,
    output_mask_path: Optional[str] = None,
    output_overlay_path: Optional[str] = None,
    target_class: Optional[str] = "Eye",
    conf: float = 0.25,
    imgsz: int = 640,
) -> None:
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Processing video: {video_path}")
    print(f"Frames: {n_frames}, Resolution: {w}x{h}, FPS: {fps}")

    model = load_segmentation_model(model_path)
    print(f"Model loaded from: {model_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    mask_writer = None
    if output_mask_path is not None:
        Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
        mask_writer = cv2.VideoWriter(str(output_mask_path), fourcc, fps, (w, h), isColor=False)
        print(f"Mask output: {output_mask_path}")

    overlay_writer = None
    if output_overlay_path is not None:
        Path(output_overlay_path).parent.mkdir(parents=True, exist_ok=True)
        overlay_writer = cv2.VideoWriter(str(output_overlay_path), fourcc, fps, (w, h))
        print(f"Overlay output: {output_overlay_path}")

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i}, stopping early.")
            break

        mask, overlay = ml_process_image(
            image_bgr=frame,
            model=model,
            target_class=target_class,
            conf=conf,
            imgsz=imgsz,
        )

        if mask_writer is not None:
            mask_writer.write(mask)
        if overlay_writer is not None:
            overlay_writer.write(overlay)

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{n_frames} frames processed")

    cap.release()
    if mask_writer is not None:
        mask_writer.release()
    if overlay_writer is not None:
        overlay_writer.release()

    print("Video processing complete!")


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────


def run_image(args: argparse.Namespace) -> None:
    """Dispatch single-image processing based on --mode."""
    print(f"Processing image: {args.image}")

    target_class = None if args.class_name.lower() == "all" else args.class_name

    if args.mode == "ml":
        mask, overlay = process_image_ml(
            args.image,
            args.model,
            target_class=target_class,
            conf=args.conf,
            imgsz=args.imgsz,
        )
    else:
        mask, overlay = process_image_ip(args.image)

    Path(args.out_mask).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_mask), mask)
    cv2.imwrite(str(args.out_overlay), overlay)
    print(f"Saved mask:    {args.out_mask}")
    print(f"Saved overlay: {args.out_overlay}")


def run_video(args: argparse.Namespace) -> None:
    """Dispatch video processing based on --mode."""
    print(f"Processing video: {args.video}")

    target_class = None if args.class_name.lower() == "all" else args.class_name

    if args.mode == "ml":
        process_video_ml(
            args.video,
            args.model,
            output_mask_path=args.out_mask,
            output_overlay_path=args.out_overlay,
            target_class=target_class,
            conf=args.conf,
            imgsz=args.imgsz,
        )
    else:
        process_video_ip(
            args.video,
            args.out_overlay,
            args.out_mask,
            max_workers=args.workers,
            debug=args.debug,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolate sclera from images/videos using IP or ML methods."
    )
    parser.add_argument(
        "--mode",
        choices=["ip", "ml"],
        default="ml",
        help="Isolation method: 'ip' (classical) or 'ml' (YOLO)",
    )
    parser.add_argument("--image", help="Input image path (single image mode)")
    parser.add_argument("--video", help="Input video path")
    parser.add_argument(
        "--model",
        default="ML_stuff/best.pt",
        help="Path to YOLO model (ML mode only)",
    )
    parser.add_argument(
        "--out-mask",
        default="output/testing/sclera_mask.mp4",
        help="Output path for mask video",
    )
    parser.add_argument(
        "--out-overlay",
        default="output/testing/sclera_overlay.mp4",
        help="Output path for overlay video",
    )
    parser.add_argument(
        "--class-name",
        default="Eye",
        help="Target class name in ML model. Use 'all' for all classes.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold (ML mode)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (ML mode)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Max workers for parallel processing (IP mode)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.image is None and args.video is None:
        print("Error: provide --image or --video")
        return

    try:
        if args.image is not None:
            run_image(args)
        if args.video is not None:
            run_video(args)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()