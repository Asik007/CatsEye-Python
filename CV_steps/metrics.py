from typing import Optional

import cv2
import numpy as np


def compute_registration_quality(
        stack: np.ndarray,
        reference: Optional[np.ndarray] = None,
        data_range: Optional[float] = None,
) -> dict:
    """
    Compute per-frame registration-quality metrics for an image stack.

    Metrics
    -------
    MAE  : Mean Absolute Error
    MSE  : Mean Squared Error
    PSNR : Peak Signal-to-Noise Ratio (dB)
    NCC  : Normalised Cross-Correlation (Pearson's r)

    Parameters
    ----------
    stack : np.ndarray
        Input stack, shape (N, H, W) for grayscale or (N, H, W, 3) for colour (BGR).
    reference : np.ndarray or None
        2D reference image to compare against. If None, the temporal mean is used.
    data_range : float or None
        Maximum possible pixel value used for PSNR. If None, inferred from dtype:
        uint8 -> 255, uint16 -> 65535, float -> 1.0 (assumes [0, 1] normalisation).

    Returns
    -------
    dict with keys: n_frames, reference_label, data_range, mae, mse, psnr, ncc
    (the four metric arrays each have shape (N,)).
    """
    if stack.ndim not in (3, 4):
        raise ValueError("stack must be 3D (N,H,W) or 4D (N,H,W,C)")

    n_frames, H, W = stack.shape[:3]
    if n_frames == 0:
        raise ValueError("stack must contain at least one frame")

    is_color = stack.ndim == 4
    if is_color and stack.shape[3] != 3:
        raise ValueError("colour stack must have 3 channels (H,W,3)")

    # ---------- convert to single-channel float ----------
    if is_color:
        # cv2.cvtColor is a per-pixel operation, so N frames can be flattened
        # into one tall image and converted in a single call instead of a loop.
        flat = stack.reshape(-1, W, 3).astype(np.float32)
        gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY).reshape(n_frames, H, W).astype(np.float64)
    else:
        gray = stack.astype(np.float64)

    # ---------- reference image ----------
    if reference is None:
        ref_img = gray.mean(axis=0)
        ref_label = "temporal mean"
    else:
        if reference.shape != (H, W):
            raise ValueError("reference must be 2D with shape matching frame dimensions")
        ref_img = reference.astype(np.float64)
        ref_label = "user supplied"

    # ---------- data range for PSNR ----------
    if data_range is None:
        if stack.dtype == np.uint8:
            data_range = 255.0
        elif stack.dtype == np.uint16:
            data_range = 65535.0
        else:
            data_range = 1.0  # assume float images are already normalised to [0, 1]

    # ---------- vectorised metrics (no per-frame Python loop) ----------
    diff = gray - ref_img  # broadcasts ref_img (H,W) against gray (N,H,W)
    mae_vals = np.abs(diff).mean(axis=(1, 2))
    mse_vals = (diff ** 2).mean(axis=(1, 2))

    with np.errstate(divide="ignore"):
        safe_mse = np.where(mse_vals == 0, 1, mse_vals)
        psnr_vals = np.where(
            mse_vals == 0,
            np.inf,
            20.0 * np.log10(data_range) - 10.0 * np.log10(safe_mse),
        )

    ref_mean = ref_img.mean()
    ref_demean = ref_img - ref_mean
    ref_norm = np.sqrt(np.sum(ref_demean ** 2))

    frame_demean = gray - gray.mean(axis=(1, 2), keepdims=True)
    frame_norm = np.sqrt(np.sum(frame_demean ** 2, axis=(1, 2)))
    numerator = np.sum(frame_demean * ref_demean, axis=(1, 2))
    denom = frame_norm * ref_norm

    valid = (frame_norm >= 1e-12) & (ref_norm >= 1e-12)
    ncc_vals = np.where(valid, numerator / np.where(denom == 0, 1, denom), 0.0)

    return {
        "n_frames": n_frames,
        "reference_label": ref_label,
        "data_range": data_range,
        "mae": mae_vals,
        "mse": mse_vals,
        "psnr": psnr_vals,
        "ncc": ncc_vals,
    }


def _print_stat_row(name: str, vals: np.ndarray) -> None:
    finite = np.isfinite(vals)
    if not np.any(finite):
        print(f"{name:<6}   all inf (perfect alignment)")
        return
    m, s, vmin, vmax = vals[finite].mean(), vals[finite].std(), vals[finite].min(), vals[finite].max()
    suffix = " (+inf)" if not np.all(finite) else ""
    print(f"{name:<6} {m:10.4f} {s:10.4f} {vmin:10.4f} {vmax:10.4f}{suffix}")


def print_registration_quality(
        stack: np.ndarray,
        reference: Optional[np.ndarray] = None,
        data_range: Optional[float] = None,
        verbose: bool = False,
) -> None:
    """
    Compute and print quality metrics for a (possibly) registered image stack.
    See `compute_registration_quality` for parameter and metric details.
    """
    metrics = compute_registration_quality(stack, reference, data_range)
    n_frames = metrics["n_frames"]
    mae_vals, mse_vals, psnr_vals, ncc_vals = (
        metrics["mae"], metrics["mse"], metrics["psnr"], metrics["ncc"]
    )

    print("\nRegistration Quality Report")
    print("===========================")
    print(f"Number of frames  : {n_frames}")
    print(f"Reference         : {metrics['reference_label']}")
    print(f"Data range (PSNR) : {metrics['data_range']}")
    print(f"{'Metric':<6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")

    for name, vals in zip(("MAE", "MSE", "PSNR", "NCC"), (mae_vals, mse_vals, psnr_vals, ncc_vals)):
        _print_stat_row(name, vals)

    if verbose:
        print("\nPer-frame values:")
        print(f"{'Frame':>6} {'MAE':>10} {'MSE':>10} {'PSNR':>10} {'NCC':>10}")
        for i in range(n_frames):
            psnr_str = f"{psnr_vals[i]:10.2f}" if np.isfinite(psnr_vals[i]) else "       inf"
            print(f"{i + 1:6d} {mae_vals[i]:10.4f} {mse_vals[i]:10.4f} {psnr_str} {ncc_vals[i]:10.6f}")