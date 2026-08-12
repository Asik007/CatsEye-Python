import numpy as np
from PIL import Image
import onnxruntime as ort
import math
import re
import cv2

# ----------------------------
# 1. Load the ONNX model
# ----------------------------
# model_path = "../VSX_stuff/model_DCA1.onnx"
# model_path = "../VSX_stuff/model_CHASEDB1.onnx"
# model_path = "../VSX_stuff/model_CHUAC.onnx"
model_path = "../VSX_stuff/model_drive.onnx"

session = ort.InferenceSession(model_path)

# # Determine cropping dimensions based on dataset type if necessary
# if self.dataset_path.endswith("DRIVE"):
#     H, W = 584, 565
# elif self.dataset_path.endswith("CHASEDB1"):
#     H, W = 960, 999
# elif self.dataset_path.endswith("DCA1"):
#     H, W = 300, 300
# else:
#     H, W = img.shape[2], img.shape[3]

# if not self.dataset_path.endswith("CHUAC"):
#     img = TF.crop(img, 0, 0, H, W)
#     pre = TF.crop(pre, 0, 0, H, W)

nn_models = {
    "DRIVE": (512, 512),
    "CHASEDB1": (1024, 1024),
    "DCA1": (256, 256),
    "CHUAC": (2048,2048),
}

# Get model input details
input_meta = session.get_inputs()[0]
match = re.search(r"model_([a-z0-9]+)\.onnx", model_path, re.IGNORECASE)
input_name = match.group(1).upper() if match else "idk"
print(f"Input name: {input_name}")
H_in, W_in = nn_models[input_name]
input_shape = [1, 1, H_in, W_in]                    # e.g., ['batch', 'channels', 'height', 'width']
# Convert to integers (ignore batch dimension which may be -1 or 1)
# Usually shape is [1, C, H, W] or [C, H, W] – we assume first is batch.
if input_shape[0] in (1, -1):
    _, num_channels, tile_h, tile_w = input_shape
else:
    # Fallback: assume shape is (batch, C, H, W)
    num_channels, tile_h, tile_w = input_shape[1], input_shape[2], input_shape[3]
tile_h, tile_w = int(tile_h), int(tile_w)

print(f"Model input: {num_channels} channels, tile size = {tile_h}x{tile_w}")

# ----------------------------
# 2. Load full image (original size)
# ----------------------------
image_path = "../output/testing/output.png"
pil_img = Image.open(image_path).convert("RGB")

# Extract green channel (single channel)
green = pil_img.getchannel("G")                  # PIL grayscale
img_np = np.array(green).astype(np.float32) / 255.0   # shape (H, W), range [0,1]

H, W = img_np.shape
print(f"Image size: {H}x{W}")

# ----------------------------
# 3. Decide: resize or tile?
# ----------------------------
# Use tiling if image is > 125% of tile size in either dimension
use_tiling = (H > tile_h * 1.5) or (W > tile_w * 1.5)
if input_name == "CHUAC":
    use_tiling = False

if not use_tiling:
    print("Image is close to model size; resizing and running directly.")
    # Resize to tile size (using bilinear interpolation)
    resized = np.array(green.resize((tile_w, tile_h), Image.BILINEAR))
    resized = resized.astype(np.float32) / 255.0
    # Add batch and channel dims -> (1, 1, tile_h, tile_w) for single channel
    input_data = resized[np.newaxis, np.newaxis, :, :]
    outputs = session.run(None, {"input": input_data})
    # Assume output is (1, 1, tile_h, tile_w) or (1, tile_h, tile_w)
    pred = outputs[0]
    if pred.ndim == 4:
        prob_map = pred[0, 0, :, :]                # (tile_h, tile_w)
    else:
        prob_map = pred[0, :, :]
    # Optionally resize back to original size? We'll keep at tile size for simplicity.
    final_prob = cv2.resize(
        prob_map,
        (W,H),
        interpolation=cv2.INTER_LINEAR
    )
else:
    print(f"Image size ({H}x{W}) exceeds tile size by >25%. Using tiling.")
    overlap_ratio = 0.25          # 25% overlap (adjust as needed)
    stride_h = int(tile_h * (1 - overlap_ratio))
    stride_w = int(tile_w * (1 - overlap_ratio))

    # Number of tiles to cover the whole image
    n_h = math.ceil((H - tile_h) / stride_h) + 1 if H > tile_h else 1
    n_w = math.ceil((W - tile_w) / stride_w) + 1 if W > tile_w else 1

    print(f"There are {n_h * n_w} tiles with {n_h} horizontal and {n_w} vertical.")

    # Starting positions (ensuring full coverage, last tile may exceed image)
    starts_h = [min(i * stride_h, H - tile_h) for i in range(n_h)]
    starts_w = [min(j * stride_w, W - tile_w) for j in range(n_w)]

    # Accumulator and count for averaging overlaps
    accumulator = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for y0 in starts_h:
        for x0 in starts_w:
            tile = img_np[y0:y0+tile_h, x0:x0+tile_w]   # (tile_h, tile_w)
            # Add batch and channel
            input_tile = tile[np.newaxis, np.newaxis, :, :]   # (1,1,tile_h,tile_w)
            outputs = session.run(None, {"input": input_tile})
            prob = outputs[0]
            if prob.ndim == 4:
                prob = prob[0, 0, :, :]              # (tile_h, tile_w)
            else:
                prob = prob[0, :, :]
            # Accumulate (overlaps will be averaged)
            accumulator[y0:y0+tile_h, x0:x0+tile_w] += prob
            count[y0:y0+tile_h, x0:x0+tile_w] += 1.0

    final_prob = accumulator / np.maximum(count, 1e-6)

# ----------------------------
# 4. Post-process and export to TIFF
# ----------------------------
binary_mask = (final_prob > 0.5).astype(np.uint8) * 255

output_tiff = f"prediction_{input_name}.tiff"
mask_img = Image.fromarray(final_prob, mode='L')
mask_img.save(output_tiff, format='TIFF')

binary_mask = Image.fromarray(binary_mask, mode='L')
binary_mask.putalpha(128)
pil_img.paste(binary_mask, (0,0), binary_mask)
pil_img.save(f"guh_{input_name}.png", format='PNG')



print(f"Prediction saved to {output_tiff}")

# Optional: also save raw probability as float TIFF (install tifffile)
# from tifffile import imwrite
# imwrite("probability.tiff", final_prob.astype(np.float32))