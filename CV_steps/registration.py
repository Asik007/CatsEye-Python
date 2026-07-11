import os
from pathlib import Path

import cv2
import numpy as np
from pystackreg import StackReg

from CV_steps.Helpers.fileio import save_tiff


def register_frame_stack(
        frame_stack: np.ndarray,
        output_dir: Path,
        reference: str = 'mean') -> np.ndarray:
    """
    Registers a 3D NumPy array (stack of frames) using pystackreg.
    align to mean frame.

    Parameters:
    -----------
    frame_stack : np.ndarray
        A 3D numpy array of shape (frames, height, width).

    Returns:
    --------
    np.ndarray
        The registered (aligned) frame stack as a numpy array.
    """

    reg_method = StackReg.RIGID_BODY

    # 2. Initialize StackReg with the chosen transformation
    sr = StackReg(reg_method)

    # 3. Perform registration and transformation
    print(f"Registering {len(frame_stack)} frames using rigid alignment...")

    # uhh i guess take out only the green channel?

    frame_stack = frame_stack[:, :, :, 1]

    registered_stack = sr.register_transform_stack(frame_stack, reference=reference)

    save_tiff(registered_stack, output_dir, "stackreg")

    return registered_stack




def dumb_register(
        frame_stack: np.ndarray,
        output_dir: Path,
        ) -> np.ndarray:
    """
    Rigidly register each frame: rotate so the principal axis is vertical,
    then translate the centroid to the image centre.

    Args:
        frame_stack:  Shape (N, H, W) for grayscale or (N, H, W, C) for colour.
        output_dir:   Directory to save registered frames as 'frame_0001.png'.
                      If empty/None, no saving is done.

    Returns:
        registered_stack: Same shape and dtype as input.
    """

    # Determine shape
    if frame_stack.ndim == 3:
        n_frames, H, W = frame_stack.shape
        is_color = False
    elif frame_stack.ndim == 4:
        n_frames, H, W, _ = frame_stack.shape
        is_color = True
    else:
        raise ValueError("frame_stack must be 3D (N,H,W) or 4D (N,H,W,C)")

    # Geometric centre of the image (using pixel indices)
    center_x = (W - 1) / 2.0
    center_y = (H - 1) / 2.0

    registered = np.empty_like(frame_stack)

    for i in range(n_frames):
        frame = frame_stack[i]

        # Convert to grayscale float for moment computation
        if is_color:
            gray = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.astype(np.float32)

        total_intensity = np.sum(gray)
        if total_intensity == 0:
            # Empty frame – pass through unchanged
            registered[i] = frame
            continue

        # Get raw moments (centroid) and central moments (orientation)
        M = cv2.moments(gray)
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        mu20 = M['mu20']
        mu02 = M['mu02']
        mu11 = M['mu11']

        # Angle of the principal axis (in radians, relative to X-axis)
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        # If the intensity distribution is near‑circular, skip rotation
        variance = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
        is_isotropic = variance < 1e-3 * (mu20 + mu02 + 1e-12)

        if is_isotropic:
            angle_deg = 0.0
        else:
            # Rotate so that the major axis aligns with the vertical (Y‑axis)
            # (rotate by +90° minus the current angle)
            angle_deg = np.degrees(np.pi / 2 - theta)

        # Build the rigid transformation matrix (rotation + translation)
        # Step 1: rotation around the centroid
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

        # Step 2: modify translation so that the centroid maps to the image centre
        rot_mat[0, 2] += (center_x - cx)
        rot_mat[1, 2] += (center_y - cy)

        # Apply the affine warp
        if is_color:
            shifted = cv2.warpAffine(
                frame, rot_mat, (W, H),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
        else:
            shifted = cv2.warpAffine(
                frame, rot_mat, (W, H),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # Preserve input dtype (warpAffine returns float64 by default)
        registered[i] = shifted.astype(frame.dtype)

    save_tiff(registered, output_dir, "dumb")

    return registered
