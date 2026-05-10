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
) -> np.ndarray:
    """
    Returns a binary mask (uint8, 0/255) for the target class.
    If target_class is None, combines all predicted instance masks.
    """
    results = model.predict(source=image_bgr, conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    result = results[0]
    # print(f"Inference results: {result.con}")
    h, w = image_bgr.shape[:2]
    out_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return out_mask

    class_ids = None

    # get the ids of the predicted classes if they exist
    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

    names = getattr(model, "names", {})
    target_id = None
    if target_class is not None:
        for k, v in names.items():
            if str(v).lower() == target_class.lower():
                target_id = int(k)
                break

    mask_data = result.masks.data.detach().cpu().numpy()  # (n, mask_h, mask_w)
    print(f"Mask data shape: {result.boxes.conf}, Class IDs: {class_ids}")

    for i, mask in enumerate(mask_data):
        if target_class is not None and target_id is not None and class_ids is not None:
            if i >= len(class_ids) or class_ids[i] != target_id:
                continue

        mask_resized = cv2.resize(
            mask.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        out_mask = np.maximum(out_mask, (mask_resized > 0.5).astype(np.uint8) * 255)

    return out_mask


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def process_image(
    image_bgr: np.ndarray,
    # model: str | Path,
    # output_mask_path: Optional[str | Path] = None,
    # output_overlay_path: Optional[str | Path] = None,
    target_class: Optional[str] = "sclera",
    conf: float = 0.25,
    imgsz: int = 640,
    model: Optional[YOLO] = None,
) -> tuple[np.ndarray, np.ndarray]:
    # image_path = Path(image_path)
    # model_path = Path(model_path)

    # image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError("Could not read image")

    mask = infer_mask(image_bgr, model, target_class=target_class, conf=conf, imgsz=imgsz)

    overlay = apply_mask(image_bgr, mask)

    return mask, overlay

