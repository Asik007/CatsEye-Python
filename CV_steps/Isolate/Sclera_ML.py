from __future__ import annotations

import argparse
from pathlib import Path
# from line_profiler import profile
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


def load_segmentation_model(model_path: str | Path) -> YOLO:
    return YOLO(str(model_path))


def infer_mask(
        image_bgr: np.ndarray,
        model: YOLO,
        target_class: Optional[str] = "Eye",
        conf: float = 0.25,
        imgsz: int = 160,
):
    """
    Returns a binary mask (uint8, 0/255) for the target class.
    If target_class is None, combines all predicted instance masks.
    """
    # Run inference
    results = model.predict(source=image_bgr, conf=conf, imgsz=imgsz, verbose=False)
    if not results or results[0].masks is None:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8), np.empty((0, 4))

    result = results[0]
    h, w = image_bgr.shape[:2]

    # Get class IDs if available
    class_ids = None
    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

    # Find target class ID
    target_id = _get_class_id(model, target_class)

    # Process masks
    masks = result.masks.data.cpu().numpy()   # (n, h, w) numpy array
    boxes = result.boxes.xyxy.cpu().numpy()    # (n, 4) bounding boxes
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for i, mask in enumerate(masks):
        # Skip if filtering by class and class info is missing
        if target_class is not None and (target_id is None or class_ids is None):
            continue
        # Skip if this mask's class doesn't match the target
        if target_class is not None and class_ids[i] != target_id:
            continue

        # Resize to original image dimensions and binarize
        mask_resized = cv2.resize(mask.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

        # Combine with existing mask (logical OR)
        combined_mask = cv2.bitwise_or(combined_mask, binary_mask)

    return combined_mask, boxes


def _get_class_id(model: YOLO, target_class: Optional[str]) -> Optional[int]:
    """Find the class ID matching target_class name."""
    if target_class is None:
        return None

    names = getattr(model, "names", {})
    for class_id, class_name in names.items():
        if str(class_name).lower() == target_class.lower():
            return int(class_id)
    return None


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def process_image(
    image_bgr: np.ndarray,
    target_class: Optional[str] = "sclera",
    conf: float = 0.25,
    imgsz: int = 640,
    model: Optional[YOLO] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image_bgr is None:
        raise FileNotFoundError("Could not read image")

    if model is None:
        raise ValueError("A valid YOLO model must be provided")

    # Unpack the two return values of infer_mask
    mask, boxes = infer_mask(
        image_bgr, model,
        target_class=target_class,
        conf=conf,
        imgsz=imgsz
    )

    overlay = apply_mask(image_bgr, mask)

    return mask, overlay, boxes