from pathlib import Path
from typing import Optional

import cv2
import numpy as np
# from pystackreg import StackReg

from CV_steps.Helpers.fileio import save_tiff


# def register_frame_stack(
#         frame_stack: np.ndarray,
#         output_dir: Path,
#         reference: str = 'mean') -> np.ndarray:
#     """
#     Registers a 3D NumPy array (stack of frames) using pystackreg.
#     align to mean frame.
#
#     Parameters:
#     -----------
#     frame_stack : np.ndarray
#         A 3D numpy array of shape (frames, height, width).
#
#     Returns:
#     --------
#     np.ndarray
#         The registered (aligned) frame stack as a numpy array.
#     """
#
#     reg_method = StackReg.RIGID_BODY
#
#     # 2. Initialize StackReg with the chosen transformation
#     sr = StackReg(reg_method)
#
#     # 3. Perform registration and transformation
#     print(f"Registering {len(frame_stack)} frames using rigid alignment...")
#
#     # uhh i guess take out only the green channel?
#
#     frame_stack = frame_stack[:, :, :, 1]
#
#     registered_stack = sr.register_transform_stack(frame_stack, reference=reference)
#
#     # save_tiff(registered_stack, output_dir, "stackreg")
#
#     return registered_stack


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




# TODO: this code is so bloated and stupid AI slop

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# NOTE: assumes `save_tiff(stack, output_dir, tag)` is defined/imported
# elsewhere in your module, as in the original code.


# TODO: this really needs to be parallelized if possible
def ecc_rigid_register(
        frame_stack: np.ndarray,
        output_dir: Path,
        ref_img: Optional[np.ndarray] = None,
        border: int = None,
) -> np.ndarray:
    """
    Rigidly register each frame to a reference image using ECC (Enhanced
    Correlation Coefficient) alignment.  This replaces the moment‑based
    ``dumb_register`` while keeping the same interface.

    Parameters
    ----------
    frame_stack : np.ndarray
        Shape (N, H, W) for grayscale or (N, H, W, C) for colour.
    output_dir : Path
        Directory where the registered stack is saved as ``ecc_rigid_register.tif``.
        If None or empty, no file is written.
    ref_img : np.ndarray, optional
        Reference frame to register to. If None, the temporal mean is used.
    border : int, optional
        Amount (in px) by which each frame's *non-zero* footprint is eroded
        before ECC runs. Frames aren't assumed to be zero-padded on a
        rectangle -- the valid area can be an arbitrary shape (e.g. after a
        prior rotation) -- so this shrinks whatever that footprint actually
        is, rather than trimming a fixed rectangular margin. The eroded
        mask is passed to OpenCV as ``inputMask`` so the ECC cost function
        ignores pixels near that (possibly non-rectangular) edge.

    Returns
    -------
    registered_stack : np.ndarray
        Same shape and dtype as input, registered to the reference.
    """
    # ---------- determine shape & convert colour ----------
    if frame_stack.ndim == 3:
        n_frames, H, W = frame_stack.shape
        is_color = False
    elif frame_stack.ndim == 4:
        n_frames, H, W, _ = frame_stack.shape
        is_color = True
    else:
        raise ValueError("frame_stack must be 3D (N,H,W) or 4D (N,H,W,C)")

    def to_gray(img: np.ndarray) -> np.ndarray:
        """Convert a single frame (colour or grayscale) to float32 grayscale."""
        if img.ndim == 3:
            return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32)

    def valid_mask(img: np.ndarray, shrink: int) -> Optional[np.ndarray]:
        """
        uint8 mask (255 = usable, 0 = not) of the true non-zero footprint
        of `img`, eroded by `shrink` px so ECC never samples right at that
        (possibly non-rectangular) boundary. None if no shrinking requested.
        """
        if not shrink:
            return None
        nonzero = np.any(img != 0, axis=-1) if img.ndim == 3 else (img != 0)
        mask = nonzero.astype(np.uint8) * 255
        k = 2 * shrink + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.erode(mask, kernel)
    # TODO: rewrite this as a euclidian distance threshold


    # Build the grayscale stack, then normalise everything (frames AND the
    # reference) by ONE shared scale factor so ECC sees consistent
    # intensities across frames -- fixes the old per-frame-max bug in the
    # colour branch.
    if is_color:
        gray_stack = np.stack(
            [to_gray(frame_stack[i]) for i in range(n_frames)]
        ).astype(np.float32)
    else:
        gray_stack = frame_stack.astype(np.float32)

    gray_max = gray_stack.max()
    norm = gray_max if gray_max > 0 else 1.0
    gray_stack = gray_stack / norm

    # ---------- reference image (now actually respects `ref_img`) ----------
    if ref_img is None:
        template = gray_stack.mean(axis=0).astype(np.float32)
        print("Reference: temporal mean image")
    else:
        template = (to_gray(ref_img) / norm).astype(np.float32)
        print("Reference: user-provided image")

    # ---------- compute transforms (ECC to reference) ----------
    tmats = np.zeros((n_frames, 3, 3), dtype=np.float64)
    warp_init = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    for i in range(n_frames):
        print(f"Frame {i + 1}/{n_frames} → reference")
        mov = gray_stack[i]
        mask_i = valid_mask(frame_stack[i], border)
        warp = warp_init.copy()
        try:
            if mask_i is not None:
                _, warp = cv2.findTransformECC(
                    template, mov, warp, cv2.MOTION_EUCLIDEAN, criteria, mask_i, 5,
                )
            else:
                _, warp = cv2.findTransformECC(
                    template, mov, warp, cv2.MOTION_EUCLIDEAN, criteria,
                )
        except cv2.error as e:
            print(f"    ECC failed for frame {i}, using identity ({e})")
            warp = warp_init.copy()  # ensure we actually fall back to identity

        # OpenCV's findTransformECC returns a warp meant to be paired with
        # WARP_INVERSE_MAP; inverting it up front lets us call warpAffine
        # normally below.
        full = np.vstack([warp, [0, 0, 1]]).astype(np.float64)
        try:
            tmats[i] = np.linalg.inv(full)
        except np.linalg.LinAlgError:
            tmats[i] = np.eye(3, dtype=np.float64)

    # ---------- apply transforms to original (full-resolution) frames ----------
    registered = np.empty_like(frame_stack)
    for i in range(n_frames):
        M = tmats[i][:2, :].astype(np.float32)  # 2×3 affine
        if is_color:
            registered[i] = cv2.warpAffine(
                frame_stack[i], M, (W, H),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            ).astype(frame_stack.dtype)
        else:
            registered[i] = cv2.warpAffine(
                frame_stack[i], M, (W, H),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(frame_stack.dtype)

    if output_dir:
        save_tiff(registered, output_dir, "ECC_reg")

    return registered