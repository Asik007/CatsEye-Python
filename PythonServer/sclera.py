import cv2
from matplotlib import image
import numpy as np
from pathlib import Path
from skimage.color import rgb2hsv
from skimage.morphology import disk, opening, closing
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops
from skimage.transform import resize

import matplotlib.pyplot as plt


def process_eye_pipeline(
    path: str,
    scale: float = 0.7, #how much it scales the image down for processing
    v_thresh: float = 0.1, #value threshold
    s_thresh: float = 0.1): #saturation threshold

    # --- 1) Load image (BGR -> RGB) ---
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 2) Resize for speed ---
    new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    image_small = resize(image, new_shape, anti_aliasing=True)

    # --- 3) Convert to HSV channels ---
    hsv = rgb2hsv(image_small.astype(np.float64))
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # --- 4) Build initial sclera mask (bright + low saturation) ---
    sclera_mask = (v > v_thresh) & (s < s_thresh)
    sclera_mask = opening(sclera_mask, disk(3))
    sclera_mask = closing(sclera_mask, disk(7))
    sclera_mask = remove_small_holes(sclera_mask, area_threshold=64)
    sclera_mask = remove_small_objects(sclera_mask, min_size=64)

    # --- 5) Keep only largest connected component ---
    labeled = label(sclera_mask)
    if labeled.max() > 0:
        regions = regionprops(labeled)
        largest_label = max(regions, key=lambda r: r.area).label
        eye_mask = labeled == largest_label
    else:
        eye_mask = sclera_mask.astype(bool)

    # --- 6) Final eye-mask refinement ---
    eye_mask = closing(eye_mask, disk(5))
    eye_mask = remove_small_holes(eye_mask, area_threshold=64)

    # --- 7) Create overlay (mask highlighted in red) ---
    overlay = image.copy()
    #resize mask to original size and overlay
    eye_mask = resize(eye_mask.astype(bool), image.shape[:2], order=0)

    overlay[eye_mask] = [255, 0, 0]  # Red overlay for eye region

    # Returns: resized RGB image, raw sclera mask, refined eye mask + overlay
    return image, eye_mask, overlay


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


def process_image(path: str, scale: float = 0.7,v_thresh: float = 0.1,s_thresh: float = 0.1):
    im, eye_mask, overlay = process_eye_pipeline(path=path,scale=scale,v_thresh=v_thresh,s_thresh=s_thresh)
    show_results(im, eye_mask, overlay)
    return im, eye_mask, overlay




if __name__ == "__main__":
    image_path = Path(__file__).resolve().parent / "uploads" / "frames" / "frame_0011.png"
    process_image(str(image_path))