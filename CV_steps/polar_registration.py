"""
Minimal script for polar-based image registration using phase cross-correlation.
Extracts and aligns rotation between two eye images.
"""

import cv2
import numpy as np



import imreg_dft as ird
import PIL
import scipy as sp
import scipy.misc
# result = cv2.asArray(result)b

im0 = np.array(PIL.Image.open("result1.png").convert("L"))
# im0 =
# the image to be transformed
im1 = np.array(PIL.Image.open("result2.png").convert("L"))

result = ird.similarity(im0, im1, numiter=3, constraints={})
print(f"Detected pixel shift (y, x): {result['tvec']}, rotation (degrees): {result['angle']}, scale: {result['scale']}")
print(f"Error: {result}, diffphas: {result['timg']}")

# calculate the error between im0 and result['timg'] to evaluate the registration quality
def MSE_error(result, im0):
    err = np.sum((im0.astype("float") - result.astype("float")) ** 2)
    err /= float(im0.shape[0] * im0.shape[1])
    return err

err = MSE_error(result['timg'], im0)

print(f"Mean squared error between im0 and registered image: {err:.6f}")
cv2.imwrite("aligned_imreg_dft.png", result['timg'])
cv2.imwrite("diff_imreg_dft.png", np.abs(im0.astype(np.float32) - result['timg'].astype(np.float32)).astype(np.uint8))
# normal cross-correlation for comparison
# from scipy.ndimage import shift as ndi_shift

# # Use FFT-based register_translation for a translation estimate (subpixel)
# shift_cc, error_cc, diffphase_cc = register_translation(im0, im1, upsample_factor=10)
# print(f"Normal cross-correlation detected pixel shift (y, x): {shift_cc}, error {error_cc:.6f}, phase difference: {diffphase_cc}")


# print("Wrote aligned_normal_cc.png and diff_normal_cc.png")

# calculate affine transformation


# ----- SciPy FFT-based cross-correlation (integer-pixel) -----
# from scipy.fft import fft2, ifft2

# def estimate_shift_scipy_fft(a, b):
#     """Estimate integer shift between two images using FFT cross-correlation."""
#     fa = fft2(a)
#     fb = fft2(b)
#     cross = fa * np.conj(fb)
#     cc = np.real(ifft2(cross))
#     max_idx = np.unravel_index(np.argmax(cc), cc.shape)
#     # compute shift relative to center
#     shifts = np.array(max_idx, dtype=np.float64)
#     center = np.array(a.shape, dtype=np.float64) / 2.0
#     shift = shifts - center
#     # wrap-around correction
#     for i in range(2):
#         if shift[i] > a.shape[i] / 2.0:
#             shift[i] -= a.shape[i]
#     # return as (dy, dx)
#     return -shift


# # Run SciPy FFT estimator
# shift_scipy = estimate_shift_scipy_fft(im0, im1)
# print(f"SciPy FFT cross-correlation shift (y, x): {shift_scipy}")

# from scipy.ndimage import shift as ndi_shift
# im1_aligned_scipy = ndi_shift(im1, shift=shift_scipy, mode='constant', cval=0.0)
# diff_scipy = np.abs(im0.astype(np.float32) - im1_aligned_scipy.astype(np.float32))
# diff_scipy_norm = (diff_scipy / (diff_scipy.max() or 1) * 255).astype(np.uint8)
# print("Wrote aligned_scipy_fft.png and diff_scipy_fft.png")

# print(f"SciPy FFT-based shift applied. Mean absolute difference after alignment: {MSE_error(im1_aligned_scipy, im0):.6f}")

# ----- OpenCV phaseCorrelate (subpixel) -----
im0_f = np.float32(im0)
im1_f = np.float32(im1)
shift_cv2, response = cv2.phaseCorrelate(im0_f, im1_f)
# cv2 returns (dx, dy) — convert to (dy, dx)
shift_cv2_yx = (shift_cv2[1], shift_cv2[0])
print(f"OpenCV phaseCorrelate shift (y, x): {shift_cv2_yx}, response: {response}")

# Apply via warpAffine (shift may be fractional)
rows, cols = im1.shape
M = np.float32([[1, 0, shift_cv2_yx[1]], [0, 1, shift_cv2_yx[0]]])
im1_aligned_cv2 = cv2.warpAffine(im1, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
diff_cv2 = np.abs(im0.astype(np.float32) - im1_aligned_cv2.astype(np.float32))
diff_cv2_norm = (diff_cv2 / (diff_cv2.max() or 1) * 255).astype(np.uint8)
print(f"OpenCV phaseCorrelate applied. Mean absolute difference after alignment: {MSE_error(im1_aligned_cv2, im0):.6f}")
print("Wrote aligned_cv2_phase.png and diff_cv2_phase.png")

