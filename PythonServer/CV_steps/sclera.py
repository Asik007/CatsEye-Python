from email.mime import image

import cv2
import os

from matplotlib.offsetbox import DEBUG
import numpy as np
import numpy.ma as ma
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
# from skimage.color import rgb2hsv
from skimage.morphology import disk, opening, closing
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops
# from skimage.transform import resize

import matplotlib.pyplot as plt

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from stabilize import stabilize_video

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func



@profile
def sclera_pipeline(input_path: str, overlay_path: str, mask_path: str, max_workers: int = 4):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS)   # ✅ Read BEFORE release
    if fps <= 0:
        fps = 30.0

    fourcc         = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))
    mask_writer    = cv2.VideoWriter(mask_path, fourcc, fps, (w, h))

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
        print(f"Processed frame {idx + 1}/{n_frames} in thread {threading.current_thread().name}")
        eye_mask, overlay = process_eye_pipeline(image=frame)
        return idx, overlay, eye_mask

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, item): item[0] for item in raw_frames}
        for future in as_completed(futures):
            idx, overlay, mask = future.result()
            # mask only returned for debugging, not currently being written to video
            results[idx] = (overlay, mask)

    # Write in order
    for i in range(len(raw_frames)):
        overlay, _ = results[i]
        # Ensure overlay is 3-channel BGR before writing
        if overlay is None:
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
        elif overlay.ndim == 2:  # Single-channel (grayscale)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        overlay_writer.write(overlay)
        print(f"Processed frame {i + 1}/{len(raw_frames)}", end="\r")
    overlay_writer.release()
    
    for i in range(len(raw_frames)):
        _, mask = results[i]
        mask_writer.write(mask)
    mask_writer.release()
    print(f"Saved overlay → {overlay_path}")
    print(f"Saved mask    → {mask_path}")




# _DISK_3 = disk(3)
# _DISK_5 = disk(5)
# _DISK_7 = disk(7)

import cv2
import numpy as np

SCALE = 0.05  # 5% of original size for faster processing
@profile
def process_eye_pipeline(image: np.ndarray, DEBUG: bool = False):
    # ── 1. Resize ─────────────────────────────────────────────────────────────
    aspect_ratio = image.shape[1] / image.shape[0]
    new_w = min(500, int(image.shape[1] * 0.25))
    new_h = int(new_w / aspect_ratio)
    low_res = cv2.resize(image, (new_w, new_h))

    # ── 2. Threshold in HSV ───────────────────────────────────────────────────
    hsv    = cv2.cvtColor(low_res, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, (0, 1, 180), (180, 35, 255))

    opened_hsv = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    close_hsv = cv2.morphologyEx(opened_hsv, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
    
    # ── 3. Find largest contour ───────────────────────────────────────────────
    contours, _ = cv2.findContours(close_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)

    # ── 4. Scale contour coords back to original image space ─────────────────
    # No magic offset — pure scale from low-res → original
    scale_x = image.shape[1] / new_w
    scale_y = image.shape[0] / new_h

    M = np.array([
        [scale_x, 0,       0],
        [0,       scale_y, 25],
        [0,       0,       1],
    ])

    homogeneous    = np.hstack([
        largest_contour.reshape(-1, 2),
        np.ones((largest_contour.shape[0], 1))
    ])
    transformed    = (M @ homogeneous.T).T[:, :2].astype(np.int32)
 
    

    mask = np.zeros_like(image, dtype=np.uint8)
    mask = cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    
    if mask is None:
        print("tf my data at?")

    contour_img = image * mask.astype(bool)
    
    
    # ── 6. Debug overlay ──────────────────────────────────────────────────────
    overlay = None
    DEBUG = True
    if DEBUG:
        overlay = image.copy()
        cv2.drawContours(overlay, [transformed], -1, (0, 255, 0), thickness=2)

    
    return contour_img, overlay  # was returning mask instead of contour_img
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


#for testing

def process_image(path: str, scale: float = 0.7,v_thresh: float = 0.1,s_thresh: float = 0.1):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    eye_mask, overlay = process_eye_pipeline(image=image)
    # show_results(eye_mask, overlay) # show results in a nice format
    return eye_mask, overlay



# from stabilize import stabilize_video
# kernprof -l ./CV_steps/sclera.py

if __name__ == "__main__":
    image_path = Path(__file__).resolve().parent.parent / "uploads" / "frames" / "frame_0011.png"
    process_image(str(image_path))

    start = cv2.getTickCount()
    vid_path = Path(__file__).resolve().parent.parent / "uploads" / "IMG_1759.MOV"
    output_dir = Path(__file__).resolve().parent.parent / "output" / "testing_sclera"
    overlay_path = os.path.join(output_dir, "sclera_overlay2.mp4")
    mask_path = os.path.join(output_dir, "sclera_mask2.mp4")
    os.makedirs(output_dir, exist_ok=True)
    sclera_pipeline(str(vid_path), overlay_path, mask_path, max_workers=8)
    # stabilize_video(overlay_path, os.path.join(output_dir, "sclera_overlay_stabilized.mp4"))
    print(f"Saved stabilized overlay → {os.path.join(output_dir, 'sclera_overlay_stabilized.mp4')}")
    end = cv2.getTickCount()
    elapsed = (end - start) / cv2.getTickFrequency()
    print(f"Total processing time: {elapsed:.2f} seconds")

