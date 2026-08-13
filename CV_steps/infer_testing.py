import numpy as np
from PIL import Image, ImageSequence
import onnxruntime as ort
import math
import re
import cv2
from CV_steps.Helpers.utils import normalize

# ----------------------------
# Configuration
# ----------------------------
BATCH_SIZE = 10                     # number of patches per inference batch
NUM_FRAMES = 1
# model_path = "../VSX_stuff/model_drive.onnx"
model_path = "../VSX_stuff/model_DCA1.onnx"
# model_path = "../VSX_stuff/model_CHASEDB1.onnx"
# model_path = "../VSX_stuff/model_CHUAC.onnx"
tiff_path = "../output/testing/clahe_stack.tiff"   # your multi-page TIFF

# ----------------------------
# 1. Load ONNX model
# ----------------------------
session = ort.InferenceSession(model_path)

# Map dataset names to model input sizes (H, W)
nn_models = {
    "DRIVE": (512, 512),
    "CHASEDB1": (1024, 1024),
    "DCA1": (256, 256),
    "CHUAC": (2048, 2048),
}

input_meta = session.get_inputs()[0]
match = re.search(r"model_([a-z0-9]+)\.onnx", model_path, re.IGNORECASE)
input_name = match.group(1).upper() if match else "idk"
print(f"Input name: {input_name}")
H_in, W_in = nn_models[input_name]
input_shape = [1, 1, H_in, W_in]   # batch, channels, height, width
_, num_channels, tile_h, tile_w = input_shape
tile_h, tile_w = int(tile_h), int(tile_w)
print(f"Model input: {num_channels} channel(s), tile size = {tile_h}x{tile_w}")

# ----------------------------
# 2. Load all pages from multi-page TIFF
# ----------------------------
pages = []   # list of (H, W) float32 arrays in [0,1]
with Image.open(tiff_path) as img:
    for page in ImageSequence.Iterator(img):
        gray = page.convert('L')            # ensure grayscale
        arr = np.array(gray).astype(np.float32) / 255.0
        pages.append(arr)
print(f"Loaded {len(pages)} pages.")

pages = pages[:NUM_FRAMES]

# ----------------------------
# 3. Generate patches for all pages
# ----------------------------
def generate_patches(image_np, tile_h, tile_w, input_name):
    """
    Generate patches (tiles) for one image.
    Returns a list of dicts: {'tensor': (1,1,tile_h,tile_w),
                              'image_idx': int,
                              'is_tile': bool,
                              'start_y': int (if tile else None),
                              'start_x': int (if tile else None)}
    """
    H, W = image_np.shape
    patches = []
    image_idx = len(pages)  # will be set later

    # Decide tiling or resize
    use_tiling = (H > tile_h * 1.5) or (W > tile_w * 1.5)
    if input_name == "CHUAC":
        use_tiling = False

    if not use_tiling:
        # Resize entire image to model input size
        resized = cv2.resize(image_np, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        tensor = resized[np.newaxis, :, :]   # (1,1,H,W)
        patches.append({
            'tensor': tensor,
            'image_idx': image_idx,
            'is_tile': False,
            'start_y': None,
            'start_x': None,
            'orig_shape': (H, W)
        })
    else:
        overlap_ratio = 0.25
        stride_h = int(tile_h * (1 - overlap_ratio))
        stride_w = int(tile_w * (1 - overlap_ratio))

        n_h = math.ceil((H - tile_h) / stride_h) + 1 if H > tile_h else 1
        n_w = math.ceil((W - tile_w) / stride_w) + 1 if W > tile_w else 1

        starts_h = [min(i * stride_h, H - tile_h) for i in range(n_h)]
        starts_w = [min(j * stride_w, W - tile_w) for j in range(n_w)]

        for y0 in starts_h:
            for x0 in starts_w:
                tile = image_np[y0:y0+tile_h, x0:x0+tile_w]
                tensor = tile[np.newaxis, :, :]
                patches.append({
                    'tensor': tensor,
                    'image_idx': image_idx,
                    'is_tile': True,
                    'start_y': y0,
                    'start_x': x0,
                    'orig_shape': (H, W)
                })
    return patches

# Build list of all patches with correct image index
all_patches = []
for idx, img in enumerate(pages):
    patches = generate_patches(img, tile_h, tile_w, input_name)
    for p in patches:
        p['image_idx'] = idx   # assign correct index
    all_patches.extend(patches)

print(f"Total patches to process: {len(all_patches)} with {len(all_patches) / BATCH_SIZE} batches")

# ----------------------------
# 4. Prepare accumulators per image
# ----------------------------
num_images = len(pages)
accumulators = [None] * num_images
counts = [None] * num_images
final_maps = [None] * num_images   # for non-tiled images (resized)

# For each image, decide whether it was tiled or not
# We can infer from the first patch of that image (all patches for an image have same is_tile)
for idx, img in enumerate(pages):
    # Check if any patch for this image is a tile
    is_tiled = any(p['is_tile'] for p in all_patches if p['image_idx'] == idx)
    if is_tiled:
        accumulators[idx] = np.zeros(img.shape, dtype=np.float32)
        counts[idx] = np.zeros(img.shape, dtype=np.float32)
    else:
        final_maps[idx] = None   # will be filled after inference

# ----------------------------
# 5. Run inference in batches
# ----------------------------
def run_batch(patch_list):
    """Run inference on a list of patches, return outputs as list of 2D maps."""
    batch_tensors = np.stack([p['tensor'] for p in patch_list], axis=0)  # (B,1,H,W)
    print(batch_tensors.shape)
    outputs = session.run(None, {"input": batch_tensors})[0]            # (B,1,H,W) or (B,H,W)
    # Convert to list of 2D maps
    maps = []
    for i in range(outputs.shape[0]):
        out = outputs[i]
        if out.ndim == 4:
            out = out[0, 0, :, :]   # (H,W)
        elif out.ndim == 3:
            out = out[0, :, :]
        maps.append(out)
    return maps

for i in range(0, len(all_patches), BATCH_SIZE):
    batch = all_patches[i:i+BATCH_SIZE]
    prob_maps = run_batch(batch)   # list of 2D arrays (tile_h, tile_w)

    for patch, prob in zip(batch, prob_maps):
        idx = patch['image_idx']
        if patch['is_tile']:
            y0, x0 = patch['start_y'], patch['start_x']
            accumulators[idx][y0:y0+tile_h, x0:x0+tile_w] += prob
            counts[idx][y0:y0+tile_h, x0:x0+tile_w] += 1.0
        else:
            # resized image – resize prob back to original size
            H_orig, W_orig = patch['orig_shape']
            final_map = cv2.resize(prob, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            final_maps[idx] = final_map

# For tiled images, compute average
for idx in range(num_images):
    if accumulators[idx] is not None:
        final_maps[idx] = accumulators[idx] / np.maximum(counts[idx], 1e-6)

# ----------------------------
# 6. Save results for each page (cv2 overlay)
# ----------------------------
for idx, prob_map in enumerate(final_maps):
    if prob_map is None:
        continue

    # Binary mask: 0 or 255
    binary = (prob_map > 0.5).astype(np.uint8) * 255

    # Save probability map as TIFF (using PIL because it handles multi-page well)

    # Normalize to [0,1]
    # prob_img = Image.fromarray((normalize(prob_map)), mode='F')

    # Raw (shifted)
    prob_img = Image.fromarray((prob_map + prob_map.min()), mode='F')

    prob_img.save(f"prediction_{input_name}_page{idx:03d}.tiff", format='TIFF')
    print(f"Tiff saved: prediction_{input_name}_page{idx:03d}.tiff")

    # --- Pure cv2 overlay: green with 70% opacity, black fully transparent ---
    # Background: original grayscale -> 3‑channel BGR (cv2 default)
    gray_uint8 = (pages[idx] * 255).astype(np.uint8)
    bg = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)

    # Foreground: solid green (BGR = (0, 255, 0))
    fg = np.zeros_like(bg)
    fg[:, :, 1] = 255   # green channel

    # Alpha: 70% opacity where mask is white, 0% where mask is black
    alpha = (binary.astype(np.float32) / 255.0) * 0.7   # range [0, 0.7]
    alpha = alpha[..., np.newaxis]  # add channel for broadcasting

    # Blend: out = bg * (1 - alpha) + fg * alpha
    blended = (bg * (1 - alpha) + fg * alpha).astype(np.uint8)

    # Save using cv2
    cv2.imwrite(f"overlay_{input_name}_page{idx:03d}.png", blended)
    print(f"Overlay saved: overlay_{input_name}_page{idx:03d}.png")

print("All predictions saved.")