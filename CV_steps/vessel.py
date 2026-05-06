import numpy as np
import cv2
from CV_steps.XCorr import gen_mask

# ── Configuration ─────────────────────────────────────────────────────────────
input_frame = r"C:\Users\dragon\Code\CatsEye-Python\output\testing_sclera\frames\frame090.png"
off_frame = r"C:\Users\dragon\Code\CatsEye-Python\output\testing_sclera\frames\frame003.png"
RADIUS = 705


def normalize_and_enhance(
    image_bgr, 
    mask, 
    sigma_x=51, 
    scale=128, 
    clip_limit=10.0, 
    tile_grid_size=(16, 16)
):
    """
    Normalize image based on background model, then enhance with CLAHE.
    
    Args:
        image_bgr: Input BGR image
        mask: Binary mask of region of interest
        sigma_x: Gaussian blur sigma for background estimation
        scale: Scaling factor after normalization
        clip_limit: CLAHE clip limit
        tile_grid_size: CLAHE tile grid size
    
    Returns:
        Masked and enhanced image
    """
    # Estimate background as slow-varying Gaussian blur
    bg_model = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma_x)

    image_float = image_bgr.astype(np.float32) + 1.0
    bg_float = bg_model.astype(np.float32) + 1.0

    image_normalized = np.clip((image_float / bg_float) * scale, 0, 255).astype(np.uint8)

    # CLAHE on LAB luminance channel
    lab_image = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe_obj.apply(l_channel)

    # Threshold based on average color in masked area
    avg_color = np.mean(image_normalized[mask == 255])
    _, thresh = cv2.threshold(l_enhanced, avg_color, 255, cv2.THRESH_BINARY)
    img_bgr = cv2.cvtColor(cv2.merge([thresh, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    mask_img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    return mask_img_bgr


# ── Load and Preprocess Images ────────────────────────────────────────────────
img_in = cv2.imread(input_frame)
img_off = cv2.imread(off_frame)

if img_in is None or img_off is None:
    raise FileNotFoundError(f"Could not load images:\n  {input_frame}\n  {off_frame}")

# Generate masks for both images
mask = gen_mask(img_in)
mask_off = gen_mask(img_off)

# Normalize and enhance
result = normalize_and_enhance(img_in, mask)
result2 = normalize_and_enhance(img_off, mask_off)

result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)

cv2.imwrite("result1.png", result)
cv2.imwrite("result2.png", result2)

