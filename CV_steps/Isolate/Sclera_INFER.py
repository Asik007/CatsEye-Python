from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from CV_steps.Isolate.YOLO_infer import YOLOModel


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

    target_id = _resolve_target_id(target_class, class_names)

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