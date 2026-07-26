#!/usr/bin/env python3
"""
rolling_ball_imagej.py

Read the first N layers of a multi-layer TIFF and apply rolling‑ball
background subtraction (like ImageJ) with a set of radii.
Optionally handle light background (invert before processing).
Saves a comparison plot as a high-res PNG.
"""

import cv2
import numpy as np


def invert_image(img):
    """Invert image (works for uint8, uint16, float)."""
    if img.dtype == np.uint8:
        return cv2.bitwise_not(img)
    elif img.dtype == np.uint16:
        return (65535 - img).astype(np.uint16)
    elif img.dtype == np.float32 or img.dtype == np.float64:
        return 1.0 - img
    else:
        # fallback: use max - img
        return np.max(img) - img


def rolling_ball(image, radius, light_background=True):
    """
    Apply rolling‑ball background subtraction (morphological top‑hat).
    If light_background is True, invert, process, then invert back.
    """
    # Invert if light background
    if light_background:
        img = invert_image(image)
    else:
        img = image

    # Create disk structuring element
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Morphological top‑hat = original - opening
    if img.ndim == 3 and img.shape[2] > 1:
        channels = cv2.split(img)
        processed = [cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, kernel) for ch in channels]
        result = cv2.merge(processed)
    else:
        result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # Invert back if needed
    if light_background:
        result = invert_image(result)

    return result


def process_rolling_ball(image_stack, radius, light_background):
    """
    Apply rolling ball background subtraction to each 2D slice
    of a 3D uint8 numpy array.

    Parameters
    ----------
    image_stack : 3D numpy array, shape (n_frames, height, width)
    radius : int
    light_background : bool

    Returns
    -------
    processed_stack : 3D numpy array of same shape, dtype float or int
    """
    processed_images = []
    for image in image_stack:  # image is a 2D slice
        processed = rolling_ball(image, radius, light_background)  # call and keep result
        processed_images.append(processed)
    return np.stack(processed_images, axis=0)  # shape (n_frames, H, W)


def clahe(image, kern_size, clip_limit):
    """
    image: 2D grayscale uint8 image
    kern_size: number of tiles per axis, e.g., 8 -> 8x8 grid of tiles
    clip_limit: contrast clipping threshold (typical range ~1.0-4.0)
    """
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=[kern_size, kern_size])
    return clahe_obj.apply(image)


def process_clahe(image_stack, kern_size, clip_limit):
    """
    Apply CLAHE to each 2D slice of a 3D uint8 numpy array.

    Parameters
    ----------
    image_stack : 3D numpy array, shape (n_frames, height, width)
    kern_size : int
    clip_limit : float

    Returns
    -------
    processed_stack : 3D numpy array of same shape, dtype uint8
    """
    processed_images = []
    for image in image_stack:  # image is a 2D slice
        if image.ndim >= 3:
            image = image[:,:,1]
        processed = clahe(image, kern_size, clip_limit)  # call and keep result
        processed_images.append(processed)
    return np.stack(processed_images, axis=0)  # shape (n_frames, H, W)
