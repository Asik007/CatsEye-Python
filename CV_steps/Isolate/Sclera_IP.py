import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func

DEBUG = False

# @profile


# @profile
def process_eye_pipeline(image: np.ndarray, DEBUG: bool = False):
    # ── 1. Resize ─────────────────────────────────────────────────────────────
    aspect_ratio = image.shape[1] / image.shape[0]
    new_w = min(500, int(image.shape[1] * 0.25))
    new_h = int(new_w / aspect_ratio)
    low_res = cv2.resize(image, (new_w, new_h))

    # ── 2. Threshold in HSV ───────────────────────────────────────────────────
    hsv = cv2.cvtColor(low_res, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, (0, 1, 180), (180, 35, 255))

    opened_hsv = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    close_hsv = cv2.morphologyEx(
        opened_hsv, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)
    )

    # ── 3. Find largest contour ───────────────────────────────────────────────
    contours, _ = cv2.findContours(
        close_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print("No contours found!")
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    # peri = cv2.arcLength(largest_contour, True)
    # simplified_contour = cv2.approxPolyDP(largest_contour, epsilon=0.015 * peri, closed=True)

    # ── 4. Scale contour coords back to original image space ─────────────────
    # No magic offset — pure scale from low-res → original
    scale_x = image.shape[1] / new_w
    scale_y = image.shape[0] / new_h

    M = np.array(
        [
            [scale_x, 0, 0],
            [0, scale_y, 25],
            [0, 0, 1],
        ]
    )

    homogeneous = np.hstack(
        [largest_contour.reshape(-1, 2), np.ones((largest_contour.shape[0], 1))]
    )
    transformed = (M @ homogeneous.T).T[:, :2].astype(np.int32)

    mask = np.zeros_like(image, dtype=np.uint8)

    mask = cv2.drawContours(
        mask, [transformed], -1, (255, 255, 255), thickness=cv2.FILLED
    )

    if mask is None:
        print("tf my data at?")

    # pos_mask = image * mask.astype(bool) #works but is our slow step
    pos_mask = cv2.bitwise_and(image, mask)

    # ── 6. Debug overlay ──────────────────────────────────────────────────────
    outlined = None
    DEBUG = False
    if DEBUG:
        outlined = image.copy()
        cv2.drawContours(outlined, [transformed], -1, (0, 255, 0), thickness=2)

    return pos_mask, outlined  # was returning mask instead of pos_mask
    # return largest_contour, overlay


# for testing
# EVERYTHING BELOW THIS POINT SHOULD BE LOCATED IN THE PIPELINE.PY SCRIPT WHICH WOULD HAVE A CLI TO TEST ITb

