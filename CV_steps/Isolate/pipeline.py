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

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np
from numpy import dtype, ndarray

from CV_steps.Isolate.YOLO_infer import YOLOModel
# from .Sclera_IP import process_eye_pipeline
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
) -> tuple[
    ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]]:
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
        # conf=conf,
        # imgsz=imgsz,
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


def load_segmentation_model(model_path: str | Path, confidence: float = 0.5) -> YOLOModel:
    return YOLOModel(confidence, str(model_path))


def infer_mask(
        image_bgr: np.ndarray,
        model: YOLOModel,
        target_class: Optional[str | int] = None,
        class_names: Optional[dict[int, str]] = None,
):

    detections = model.Predict(image_bgr)

    h, w = image_bgr.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if not detections:
        return combined_mask, np.empty((0, 4))

    # target_id = _resolve_target_id(target_class, class_names)

    kept_boxes = []
    for detection in detections:
        # if target_id is not None and detection["class"] != target_id:
        #     continue

        binary_mask = detection["mask"].astype(np.uint8) * 255
        combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
        kept_boxes.append(detection["box"])

    boxes = np.array(kept_boxes, dtype=float) if kept_boxes else np.empty((0, 4))
    return combined_mask, boxes


def _resolve_target_id(
        target_class: Optional[str | int],
        class_names: Optional[dict[int, str]],
) -> Optional[int]:
    """Resolve target_class (name or index) to a class index."""
    if target_class is None:
        return None
    if isinstance(target_class, int):
        return target_class

    if class_names is None:
        raise ValueError(
            "target_class was given as a name but no class_names mapping was "
            "provided. YOLOModel does not store class names itself -- pass a "
            "{index: name} dict via class_names, or pass target_class as an int."
        )

    for class_id, class_name in class_names.items():
        if str(class_name).lower() == str(target_class).lower():
            return int(class_id)

    raise ValueError(f"Unknown class name: {target_class!r}")


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def process_image(
    image_bgr: np.ndarray,
    model: YOLOModel,
    target_class: Optional[str | int] = None,
    class_names: Optional[dict[int, str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image_bgr is None:
        raise FileNotFoundError("Could not read image")

    if model is None:
        raise ValueError("A valid YOLOModel must be provided")

    mask, boxes = infer_mask(
        image_bgr, model,
        # target_class=target_class,
        class_names=class_names,
    )

    overlay = apply_mask(image_bgr, mask)

    # debug_show_overlay(overlay, boxes, window_name="Processed Frame")
    # I want a function that I can comment out that shows each frame in the overlay with the bounding boxes laid on top

    return mask, overlay, boxes


def debug_show_overlay(
    overlay: np.ndarray,
    boxes: np.ndarray,
    window_name: str = "Debug Overlay",
    color: tuple[int, int, int] = (0, 255, 0),   # green
    thickness: int = 2,
    wait_key: bool = True,
) -> bool:
    """
    Draw bounding boxes on the overlay and show it in a window.
    Pauses until a key is pressed.

    Args:
        overlay: Masked image (BGR).
        boxes: (N, 4) array of bounding boxes [x1, y1, x2, y2].
        window_name: Title of the display window.
        color: BGR color for the boxes.
        thickness: Line thickness.
        wait_key: If True, waits for a key press; if False, shows briefly.

    Returns:
        False if user pressed 'q' or ESC (to stop batch), True otherwise.
    """
    # Work on a copy so we don't modify the original overlay
    display = overlay.copy()
    print(f"Boxes: {boxes}")

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow(window_name, display)

    if wait_key:
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)
        if key == ord('q') or key == 27:
            return False
    else:
        cv2.destroyWindow(window_name)

    return True


# if __name__ == "__main__":
#     in_img = cv2.imread(r"C:\Users\dragon\Code\CatsEye-Python\uploads\frames\frame0015.jpg")
#     YOLOM = load_segmentation_model(r"C:\Users\dragon\Code\CatsEye-Python\ML_stuff\exports\model_640_False.onnx")
#     mask, overlay, boxes = process_image(in_img, YOLOM)
#     cv2.imshow("Mask", mask)
#     cv2.imshow("Overlay", overlay)
#     cv2.waitKey(0)