import numpy as np

def crop_consistent_rgb(arr, threshold=0):
    """
    Crop all RGB frames to the common bounding box of non‑black pixels.

    Parameters
    ----------
    arr : ndarray, shape (F, H, W, C)
        Stack of RGB (or RGBA) frames.
    threshold : int or float
        Pixels with all channels <= threshold are considered black.

    Returns
    -------
    cropped : ndarray, shape (F, H', W', C)
        Cropped stack (view on the original data).
    """
    # Collapse frame axis (0) and channel axis (-1) -> mask of shape (H, W)
    # True if ANY channel of ANY frame has a pixel > threshold
    mask = np.any(arr > threshold, axis=(0, -1))  # shape (H, W)

    # Find rows and columns that contain at least one non‑black pixel
    rows = np.any(mask, axis=1)  # shape (H,)
    cols = np.any(mask, axis=0)  # shape (W,)

    if not rows.any() or not cols.any():
        # All frames are entirely black – return the original array
        return arr

    # Get the bounding box indices
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop every frame to the same region, keeping all channels
    return arr[:, rmin:rmax+1, cmin:cmax+1, :]