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
def sclera_pipeline(
    input_path: str, overlay_path: str, mask_path: str, max_workers: int = 4
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # ✅ Read BEFORE release
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Read all frames first (lightweight — raw bytes only)
    raw_frames = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i}, stopping early.")
            break
        raw_frames.append((i, frame))
    cap.release()

    # Process frames in parallel
    results = {}  # {index: (overlay, mask)}

    def process(item):
        idx, frame = item

        if DEBUG:
            print(
                f"Processed frame {idx + 1}/{n_frames} in thread {threading.current_thread().name}"
            )
        masked_img, outlined = process_eye_pipeline(image=frame)
        return idx, outlined, masked_img

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, item): item[0] for item in raw_frames}
        for future in as_completed(futures):
            idx, outlined, masked_img = future.result()
            # mask only returned for debugging, not currently being written to video
            results[idx] = (outlined, masked_img)

    # only run if there is data to write
    if results and results[0][0] is not None:
        outlined_writer = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))
        # Write in order
        for i in range(len(raw_frames)):
            outlined, _ = results[i]
            # Ensure overlay is 3-channel BGR before writing
            outlined_writer.write(outlined)
            print(f"Processed frame {i + 1}/{len(raw_frames)}", end="\r")
        outlined_writer.release()

    mask_writer = cv2.VideoWriter(mask_path, fourcc, fps, (w, h))
    for i in range(len(raw_frames)):
        _, masked_img = results[i]
        mask_writer.write(masked_img)
    mask_writer.release()
    print(f"Saved overlay → {overlay_path}")
    print(f"Saved mask    → {mask_path}")


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


def show_results(sclera_mask: np.ndarray, overlay: np.ndarray) -> None:
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(sclera_mask)
    plt.title("Sclera mask")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Combined eye mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# for testing


def process_image(
    path: str, scale: float = 0.7, v_thresh: float = 0.1, s_thresh: float = 0.1
):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    eye_mask, overlay = process_eye_pipeline(image=image)
    # show_results(eye_mask, overlay) # show results in a nice format
    return eye_mask, overlay


# from stabilize import stabilize_video
# kernprof -l ./CV_steps/sclera.py

if __name__ == "__main__":
    print("why am I running?")
    from CV_steps.stabilize_frame import stabilize_video

    image_path = (
        Path(__file__).resolve().parent.parent / "uploads" / "frames" / "frame_0011.png"
    )
    process_image(str(image_path))

    start = cv2.getTickCount()
    vid_path = Path(__file__).resolve().parent.parent / "uploads" / "IMG_1759.MOV"
    output_dir = Path(__file__).resolve().parent.parent / "output" / "testing_sclera"
    overlay_path = os.path.join(output_dir, "sclera_overlay_IP.mp4")
    mask_path = os.path.join(output_dir, "sclera_mask_IP.mp4")
    os.makedirs(output_dir, exist_ok=True)
    sclera_pipeline(str(vid_path), overlay_path, mask_path, max_workers=8)
    stabilize_video(
        mask_path, os.path.join(output_dir, "sclera_overlay_stabilized.mp4")
    )
    print(
        f"Saved stabilized overlay → {os.path.join(output_dir, 'sclera_overlay_stabilized.mp4')}"
    )
    end = cv2.getTickCount()
    elapsed = (end - start) / cv2.getTickFrequency()
    print(f"Total processing time: {elapsed:.2f} seconds")
