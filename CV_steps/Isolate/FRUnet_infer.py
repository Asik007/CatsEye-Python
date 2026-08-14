import math
import re

from PIL import Image
import cv2
import numpy as np
import onnxruntime
from CV_steps.Helpers.utils import normalize
from pathlib import Path


# Temperorary function to solve switching between input sizes bc I export them wrong
def GetModelInputSize(model_path: str) -> tuple[str, int, int]:
    nn_models = {
        "DRIVE": (512, 512),
        "CHASEDB1": (1024, 1024),
        "DCA1": (256, 256),
        "CHUAC": (2048, 2048),
    }

    # Extract the model identifier from the filename
    match = re.search(r"model_([a-z0-9]+)\.onnx", model_path, re.IGNORECASE)
    input_name = match.group(1).upper() if match else None

    if input_name is None:
        raise ValueError(f"Could not extract dataset name from model path: {model_path}")

    if input_name not in nn_models:
        raise ValueError(f"Unknown model name: {input_name} (expected one of {list(nn_models.keys())})")

    H_in, W_in = nn_models[input_name]
    return input_name, H_in, W_in


class FRUnet:
    def __init__(self,output_path: Path, detectionThreshold=.5, model_path="VSX_stuff/model_DCA1.onnx"):
        self.output_path = output_path
        self._session = onnxruntime.InferenceSession(model_path, providers=self.GetProviders())
        self.model_name, self._inputHeight, self._inputWidth = GetModelInputSize(model_path)
        print(f"Model input size: {self._inputHeight}x{self._inputWidth}")

        self._detectionThreshold = detectionThreshold

        # Output size – assume same as input (tile size)
        for output in self._session.get_outputs():
            print(f"Output Name: {output.name}, Shape: {output.shape}")
        # self._outputWidth = self._session.get_outputs()[1].shape[3] 
        # self._outputHeight = self._session.get_outputs()[1].shape[2]
        self._outputWidth = self._inputWidth
        self._outputHeight = self._inputHeight

        self._tile_h = self._inputHeight
        self._tile_w = self._inputWidth

        print(f"FRUnet {self.model_name} defined as: {vars(self)}")


    # Return ONNX Runtime providers
    def GetProviders(self) -> list[str]:
        print(f"Available ONNX Runtime providers: {onnxruntime.get_available_providers()}")
        provider = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider", "DmlExecutionProvider") if provider in onnxruntime.get_available_providers()]
        print(f"Using ONNX Runtime providers: {provider}")
        return provider

    # ----------------------------------------------------------------------
    # Internal patch generation (copied and adapted from the original script)
    # ----------------------------------------------------------------------
    def _generate_patches(self, image_np):
        """
        Generate patches (tiles) for one image.
        Returns a list of dicts: {'tensor': (1,1,tile_h,tile_w),
                                  'start_y': int (if tile else None),
                                  'start_x': int (if tile else None),
                                  'orig_shape': (H, W)}
        """
        H, W = image_np.shape
        patches = []

        # Decide tiling or resize
        use_tiling = (H > self._tile_h * 1.5) or (W > self._tile_w * 1.5)

        if not use_tiling:
            # Resize entire image to model input size
            resized = cv2.resize(image_np, (self._tile_w, self._tile_h), interpolation=cv2.INTER_LINEAR)
            tensor = resized[np.newaxis, :, :]   # (1,1,H,W)
            patches.append({
                'tensor': tensor,
                # 'is_tile': False,
                'start_y': None,
                'start_x': None,
                'orig_shape': (H, W)
            })
        else:
            self.overlap_ratio = 0.25 # Hardcoded for now, but needs to be a parameter
            stride_h = int(self._tile_h * (1 - self.overlap_ratio))
            stride_w = int(self._tile_w * (1 - self.overlap_ratio))

            n_h = math.ceil((H - self._tile_h) / stride_h) + 1 if H > self._tile_h else 1
            n_w = math.ceil((W - self._tile_w) / stride_w) + 1 if W > self._tile_w else 1

            starts_h = [min(i * stride_h, H - self._tile_h) for i in range(n_h)]
            starts_w = [min(j * stride_w, W - self._tile_w) for j in range(n_w)]

            for y0 in starts_h:
                for x0 in starts_w:
                    tile = image_np[y0:y0+self._tile_h, x0:x0+self._tile_w]
                    tensor = tile[np.newaxis, :, :]
                    patches.append({
                        'tensor': tensor,
                        # 'is_tile': True,
                        'start_y': y0,
                        'start_x': x0,
                        'orig_shape': (H, W)
                    })
        return patches

    
    # ----------------------------------------------------------------------
    # Run inference on a batch of patches
    # ----------------------------------------------------------------------
    def _run_batch(self, patch_list):
        """Run inference on a list of patches, return outputs as list of 2D maps."""
        batch_tensors = np.stack([p['tensor'] for p in patch_list], axis=0)  # (B,1,H,W)
        outputs = self._session.run(None, {"input": batch_tensors})[0]       # (B,1,H,W) or (B,H,W)
        maps = []
        for i in range(outputs.shape[0]):
            out = outputs[i]
            if out.ndim == 4:
                out = out[0, 0, :, :]   # (H,W)
            elif out.ndim == 3:
                out = out[0, :, :]
            maps.append(out)
        return maps

    # ----------------------------------------------------------------------
    # Predict: return probability map (float32) for the input image
    # ----------------------------------------------------------------------
    def Predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes a single grayscale image (H, W) and returns a probability map (H, W)
        and a boolean detection mask (H, W) with values in [0, 1].
        """
        # Convert to grayscale if needed
        if image.ndim != 2:
                raise ValueError(f"Unsupported image channels, got {image.shape}, expected 2 dimensions.")
        else:
            gray = normalize(image)

        # Normalize to [0,1] float

        patches = self._generate_patches(gray)

        print(f"Total patches to process: {len(patches)}")

        # If only one patch and it's a resize (not tile), run directly
        if len(patches) == 1:
            # Run inference on the resized image
            patch = patches[0]
            batch_tensor = patch['tensor'][np.newaxis, :, :, :]  # (1,1,H,W)
            output = self._session.run(None, {"input": batch_tensor})[0]
            if output.ndim == 4:
                prob = output[0, 0, :, :]
            elif output.ndim == 3:
                prob = output[0, :, :]
            else:
                prob = output
            # Resize back to original size
            H_orig, W_orig = patch['orig_shape']
            prob_map = cv2.resize(prob, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            return prob_map, (prob_map > self._detectionThreshold)

        # Tiling case: accumulate results
        H_orig, W_orig = patches[0]['orig_shape']
        accumulator = np.zeros((H_orig, W_orig), dtype=np.float32)
        counts = np.zeros((H_orig, W_orig), dtype=np.float32)

        # Process in batches (use a small batch size, e.g., 10)
        self.batch_size = 10
        for i in range(0, len(patches), self.batch_size):
            print(f"Processing patches {i} to {min(i+self.batch_size, len(patches))} of {len(patches)}")
            batch = patches[i:i+self.batch_size]
            prob_maps = self._run_batch(batch)
            for patch, prob in zip(batch, prob_maps):
                # if patch['is_tile']:
                y0, x0 = patch['start_y'], patch['start_x']
                accumulator[y0:y0+self._tile_h, x0:x0+self._tile_w] += prob
                counts[y0:y0+self._tile_h, x0:x0+self._tile_w] += 1.0
                # print(f"Processed patch at ({y0},{x0}) with shape {prob.shape}")

        # Average
        prob_map = accumulator / np.maximum(counts, 1e-6)
        return prob_map, (prob_map > self._detectionThreshold)

    # ----------------------------------------------------------------------
    # Draw green overlay on the input image based on probability map
    # ----------------------------------------------------------------------
    def DrawDetections(self, image: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
        """
        Overlay a green mask with 70% opacity where prob_map > detectionThreshold.
        Returns a BGR image (uint8).
        """
        # Threshold to binary mask
        binary = (prob_map > self._detectionThreshold).astype(np.uint8) * 255

        # Convert input image to BGR (if grayscale) for background
        if image.ndim == 2:
            bg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            bg = image.copy()
        elif image.ndim == 3 and image.shape[2] == 1:
            bg = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("Unsupported image format")

        # Ensure bg is uint8
        if bg.dtype != np.uint8:
            bg = (bg * 255).astype(np.uint8) if bg.max() <= 1.0 else bg.astype(np.uint8)

        # Foreground: solid green (BGR = 0,255,0)
        fg = np.zeros_like(bg)
        fg[:, :, 1] = 255

        # Alpha: 70% opacity where mask is white, 0% where mask is black
        alpha = (binary.astype(np.float32) / 255.0) * 0.7
        alpha = alpha[..., np.newaxis]  # broadcast

        # Blend: out = bg * (1 - alpha) + fg * alpha
        blended = (bg * (1 - alpha) + fg * alpha).astype(np.uint8)
        return blended
    
    def SaveResult(self, image: np.ndarray, output_path: Path, prob_map: np.ndarray, bin_map: np.ndarray):
        """
        Saves both the probability map (as TIFF) and the overlay (as PNG).
        If prob_map is None, it runs inference first.
        """

        tiff_data = prob_map + prob_map.min()
        
        # Save probability map as TIFF
        success = cv2.imwrite(output_path / f"raw_prob_{self.model_name}.tiff", tiff_data)
        if success:
            print(f"TIFF saved: {output_path}/raw_prob_{self.model_name}.tiff")
        else:
            # Fallback: scale to uint16 (preserves high precision) if float32 fails
            print("Warning: Float32 TIFF save failed. Saving as uint16 (scaled 0-65535).")
            tiff_uint16 = (normalize(prob_map) * 65535).astype(np.uint16)
            cv2.imwrite(output_path / f"raw_prob_{self.model_name}.tiff", tiff_uint16)
            print(f"TIFF (uint16) saved: {output_path}/raw_prob_{self.model_name}.tiff")

        
        # Generate and save overlay
        overlay = self.DrawDetections(image, bin_map.astype(np.uint8))
        cv2.imwrite(output_path / f"vessel_overlay_{self.model_name}.png", overlay)
        print(f"Saved probability map and overlay to {output_path}")

        cv2.imwrite(output_path / f"binary_mask_{self.model_name}.png", bin_map.astype(np.uint8) * 255)
        print(f"Saved binary mask to {output_path}")



    # ----------------------------------------------------------------------
    # Execute: full pipeline – predict + overlay
    # ----------------------------------------------------------------------
    save_results = True  # Default to saving results

    def Execute(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prob_map, bin_map = self.Predict(image)
        if self.save_results:
            self.SaveResult(image, self.output_path, prob_map, bin_map)
        return prob_map, bin_map