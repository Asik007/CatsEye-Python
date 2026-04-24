import cv2
from matplotlib import image
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
        mask = eye_mask.astype(np.uint8) * 255
        return idx, overlay, mask

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




_DISK_3 = disk(3)
_DISK_5 = disk(5)
_DISK_7 = disk(7)

def process_eye_pipeline(
    image: np.ndarray,
    scale: float = 0.7, # Resize for speed (0.7 = 70% of original size)
    v_thresh: float = 0.1, # Brightness threshold (0-1)
    s_thresh: float = 0.1, # Saturation threshold (0-1)
) -> tuple[np.ndarray, np.ndarray]:
    # outputs are: resized RGB image, raw sclera mask, refined eye mask + overlay
    
    # --- 1) Load image (BGR -> RGB) ---
    if image is None:
        raise FileNotFoundError("Could not read image")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 2) Resize for speed ---
    new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    image_small = cv2.resize(image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)

    # --- 3) Convert to HSV channels ---
    hsv = rgb2hsv(image_small.astype(np.float32) / 255.0)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # --- 4) Build initial sclera mask (bright + low saturation) ---
    sclera_mask = (v > v_thresh) & (s < s_thresh)
    sclera_mask = opening(sclera_mask, disk(3))
    sclera_mask = closing(sclera_mask, disk(7))
    sclera_mask = remove_small_holes(sclera_mask, max_size=64)
    sclera_mask = remove_small_objects(sclera_mask, max_size=64)

    # --- 5) Keep only largest connected component ---
    labeled = label(sclera_mask)
    if labeled.max() > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0  # exclude background
        eye_mask = labeled == counts.argmax()
    else:
        eye_mask = sclera_mask.astype(bool)

    # --- 6) Final eye-mask refinement ---
    eye_mask = closing(eye_mask, disk(5))
    eye_mask = remove_small_holes(eye_mask, max_size=64)

    # --- 7) Create overlay (mask highlighted in red) ---
    
    overlay = image.copy()

    # ✅ Rotate FIRST (when mask is still at its own small resolution, cheap op)
    # eye_mask_rotated = np.rot90(eye_mask)

    # THEN resize to match overlay/image dimensions
    eye_mask_full = cv2.resize(
        eye_mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),   # (width, height)
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    print((~eye_mask_full).dtype, (~eye_mask_full).shape, np.sum(~eye_mask_full))

    overlay[eye_mask_full] = [255, 0, 0]  # Red overlay for eye region


    # Returns: resized RGB image, raw sclera mask, refined eye mask + overlay

    #TODO: SOMETHING IS WRONG HERE AND IT RETURNS AN INVERTED COLOR IMAGE

    eye_mask_region1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inv_mask = ~eye_mask_full
    eye_mask_region1[inv_mask] = [255,255,255]
    eye_mask_region2 = cv2.cvtColor(eye_mask_region1, cv2.COLOR_BGR2RGB)


    return eye_mask_region2, overlay


def show_results(image: np.ndarray, sclera_mask: np.ndarray, overlay: np.ndarray) -> None:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(sclera_mask, cmap="gray")
    plt.title("Sclera mask")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(overlay)
    plt.title("Combined eye mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


#for testing

def process_image(path: str, scale: float = 0.7,v_thresh: float = 0.1,s_thresh: float = 0.1):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    im, eye_mask, overlay = process_eye_pipeline(image=image, scale=scale, v_thresh=v_thresh, s_thresh=s_thresh)
    # show_results(im, eye_mask, overlay) # show results in a nice format
    return im, eye_mask, overlay




# if __name__ == "__main__":
#     image_path = Path(__file__).resolve().parent / "uploads" / "frames" / "frame_0011.png"
#     process_image(str(image_path))