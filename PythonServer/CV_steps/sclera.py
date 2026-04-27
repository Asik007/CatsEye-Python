import cv2
from matplotlib import image
import os

import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from skimage.color import rgb2hsv
from skimage.morphology import disk, opening, closing
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops
from skimage.transform import resize

import matplotlib.pyplot as plt

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

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
        eye_mask, overlay = process_eye_pipeline(image=frame)
        return idx, overlay, eye_mask

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, item): item[0] for item in raw_frames}
        for future in as_completed(futures):
            idx, overlay, mask = future.result()
            results[idx] = (overlay, mask)

    # Write in order
    for i in range(len(raw_frames)):
        overlay, _ = results[i]
        overlay_writer.write(overlay)
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
def process_eye_pipeline(image: np.ndarray):
    # This blurring takes so damn long — need to optimize
    # blurred = cv2.medianBlur(image, 67)  # must be odd
    
    # blur is so much faster but more jagged
    blurred = cv2.blur(image, (15, 15))  # simple box blur — faster than Gaussian and median, but less effective at noise reduction

    # TODO: Redo with single resize + HSV thresholding — should be faster and better than multiple blurs

    cap_32_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    thresh_hsv = cv2.inRange(cap_32_hsv, (0, 1, 180), (180, 35, 255))
    contours, _ = cv2.findContours(thresh_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)


    overlay = cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    mask = np.zeros_like(image)
    mask_out = image.copy()
    # isolate only the masked region (white) and make the rest black

    mask = cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED).astype(bool)
    mask_out[~mask] = 0  # set non-sclera pixels to black


    return mask_out, overlay

    # return largest_contour, overlay

def draw_contour(frame, contour):
    # Only draw once
    return cv2.drawContours(frame.copy(), [contour], -1, (0, 255, 0), 2)

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
    show_results(eye_mask, overlay) # show results in a nice format
    return eye_mask, overlay



from CV_steps.stabilize import stabilize_video

if __name__ == "__main__":
    image_path = Path(__file__).resolve().parent / "uploads" / "frames" / "frame_0011.png"
    process_image(str(image_path))

    vid_path = Path(__file__).resolve().parent / "uploads" / "IMG_1734.mov"
    output_dir = Path(__file__).resolve().parent / "output" / "testing_sclera"
    overlay_path = os.path.join(output_dir, "sclera_overlay2.mp4")
    mask_path = os.path.join(output_dir, "sclera_mask2.mp4")
    os.makedirs(output_dir, exist_ok=True)
    sclera_pipeline(str(vid_path), overlay_path, mask_path, max_workers=8)
    stabilize_video(mask_path, os.path.join(output_dir, "sclera_overlay_stabilized.mp4"))
