import time

from CV_steps.Render.render import render_compare
import cv2
import numpy as np
import os

# ----------------------------------------------------------------------
# Helper functions – each mimics a specific utility from imreg_dft
# ----------------------------------------------------------------------

def _highpass(img, sigma=1.5):
    """
    Emulates imreg_dft.utils.imfilter() and _logpolar_filter().
    Removes low spatial frequencies using a difference-of-Gaussians
    (DoG) approach.

    Low frequencies often correspond to overall illumination / slowly
    varying background, which phase correlation is insensitive to and
    can swamp the finer detail needed for rotation/scale detection.

    Args:
        img : 2D (h, w) float array.
        sigma : standard deviation for the Gaussian blur.
    Returns:
        High-pass filtered image.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)  # low-pass component
    return img - blurred                             # subtract it → high-pass


def _apodize(img):
    """
    Emulates imreg_dft.utils._apodize().
    Applies a 2D Hann (Hanning) window to make the image borders
    go smoothly to zero. This prevents edge discontinuities from
    creating high-frequency artefacts in the Fourier domain, which
    would corrupt phase correlation.
    """
    r, c = img.shape[:2]
    # Create 1D Hann windows
    win_r = np.hanning(r)
    win_c = np.hanning(c)
    # Multiply to get a 2D window
    mask = np.outer(win_r, win_c)
    if img.ndim == 3:
        mask = mask[:, :, np.newaxis]   # broadcast for multi-channel
    return img * mask


def _logpolar(mag, center, out_shape, max_radius):
    """
    Emulates imreg_dft.imreg._logpolar().
    Performs log-polar transform of the Fourier magnitude.

    This converts rotation and scaling in the spatial domain into
    simple translations in the log-polar domain:
      - Rotation in image → circular shift along the angular axis
      - Scaling in image → shift along the log-radius axis

    Args:
        mag : magnitude spectrum (centered via fftshift).
        center : (x, y) centre of the transform (usually image centre).
        out_shape : (height, width) of the output log-polar image.
        max_radius : maximum radius to map to the outermost edge.
    Returns:
        Log-polar image.
    """
    h, w = out_shape
    return cv2.warpPolar(mag, (w, h), center, max_radius,
                         cv2.WARP_POLAR_LOG)


def _phasecorr_logpolar(lp1, lp2, out_shape, max_radius):
    """
    Performs phase correlation on the log-polar images.
    Converts the shift (dx, dy) into rotation angle and scale factor.

    dx (horizontal shift) corresponds to rotation:
        full range 0..360° is mapped to width `w`.
        A shift of dx pixels → angle = -dx * 360 / w   (negative because
        we want the transformation to *undo* the rotation).

    dy (vertical shift) corresponds to radial change:
        The radius mapping is r = max_radius ** (y / (h-1)).
        A shift dy moves the origin along this exponential scale,
        so the spectral scale factor = exp(dy * ln(max_radius) / (h-1)).
        The image-domain scale factor is the inverse of the spectral factor.

    Returns:
        (angle_deg, scale_img, peak_response)
    """
    h, w = out_shape
    d_vec, response = cv2.phaseCorrelate(lp1, lp2)
    dx, dy = d_vec
    # angle: horizontal shift
    angle = -dx * 360.0 / w

    # # scale: vertical shift → factor in log-polar → image scale = 1 / spectral scale
    # scale_spectrum = np.exp(dy * np.log(max_radius) / (h - 1))
    # scale_img = 1.0 / scale_spectrum

    return angle, response


def _rigid_transform(img, angle_deg, scale, center):
    """
    Rotate and scale an image around a given centre.
    This corresponds to imreg_dft.imreg.transform_img() when
    rotation and scale are applied together.

    Using OpenCV’s getRotationMatrix2D gives a 2×3 affine matrix
    that rotates by `angle_deg` *counter‑clockwise* and scales by `scale`.
    """
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR)


def _translate(img, dx, dy):
    """
    Translate an image by (dx, dy). Used for the final translation
    correction after rotation/scale alignment.
    """
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------
# Main registration function – follows the imreg_dft pipeline step‑by‑step
# ----------------------------------------------------------------------

def imreg_dft_emulate(
    template,
    subject,
    upscale_factor=None,
    logpolar_size=(360, 360),
    highpass_sigma=1.5,
    output_dir="output"
):
    """
    Emulates the image registration pipeline of imreg_dft.

    Parameters
    ----------
    template, subject : ndarray (y, x [, channel])
        Source images as 2D or 3D numpy arrays.
    upscale_factor : float or None
        Optional resampling factor applied to both images before registration
        (imreg_dft.tiles.resample()).
    logpolar_size : tuple (height, width)
        Resolution of the log‑polar image used for rotation/scale detection.
    highpass_sigma : float
        Sigma for the difference‑of‑Gaussians high-pass filter.

    Returns
    -------
    dict with keys:
        'angle'  : clockwise rotation of subject relative to template [degrees]
        'scale'  : isotropic scale factor (subject / template)
        'tvec'   : (dx, dy) translation in pixels (after rotation/scale correction)
        'success': peak phase‑correlation value (higher = better match)
        't_img'  : registered subject image, cropped and scaled back to the
                   original template dimensions.
    """
    # ---------------------------------------------------------------
    # Step 1: Load images (here already provided as numpy arrays)
    # Convert to grayscale float32 – imreg_dft works internally in
    # floating point and usually in single channel for phase correlation.
    # ---------------------------------------------------------------
    if template.ndim == 3:
        template_g = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        template_g = template.astype(np.float32)

    if subject.ndim == 3:
        subject_g = cv2.cvtColor(subject, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        subject_g = subject.astype(np.float32)


    # Remember the original template dimensions *before* extension.
    # They will be used to crop the final registered image back to
    # the original size.
    orig_h, orig_w = template_g.shape[:2]

    # ---------------------------------------------------------------
    # Step 3: Extend images to identical size – emulates
    # imreg_dft.utils.embed_to() and _preprocess_extend().
    # Zero‑padding avoids wraparound artefacts in FFT.
    # ---------------------------------------------------------------
    h_t, w_t = template_g.shape
    h_s, w_s = subject_g.shape
    max_h, max_w = max(h_t, h_s), max(w_t, w_s)

    if h_t < max_h or w_t < max_w:
        template_g = cv2.copyMakeBorder(template_g, 0, max_h - h_t, 0, max_w - w_t,
                                        cv2.BORDER_CONSTANT, value=0)
    if h_s < max_h or w_s < max_w:
        subject_g = cv2.copyMakeBorder(subject_g, 0, max_h - h_s, 0, max_w - w_s,
                                        cv2.BORDER_CONSTANT, value=0)

    # ---------------------------------------------------------------
    # Step 4: High-pass filtering – strips low spatial frequencies,
    # as done by imreg_dft.utils.imfilter() and later by
    # _logpolar_filter() inside the angle/scale stage.
    # ---------------------------------------------------------------
    template_f = _highpass(template_g, sigma=highpass_sigma)
    subject_f  = _highpass(subject_g,  sigma=highpass_sigma)

    # ---------------------------------------------------------------
    # Step 5: Angle & scale estimation (imreg_dft.imreg.similarity())
    # ---------------------------------------------------------------
    # 5a: Apodize (window) to avoid edge artefacts
    tap = _apodize(template_f)
    sap = _apodize(subject_f)

    # Compute Fourier magnitude spectra (centered via fftshift)
    F1 = np.fft.fftshift(np.fft.fft2(tap))
    # remap the magnitude to [0, 255] for visualization (optional)
    F1_img = cv2.normalize(np.log(np.abs(F1) + 1), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "F1.png"), F1_img)  # for visualization
    F2 = np.fft.fftshift(np.fft.fft2(sap))
    F2_img = cv2.normalize(np.log(np.abs(F2) + 1), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "F2.png"), F2_img)  # for visualization
    mag1, mag2 = np.abs(F1), np.abs(F2)

    # Log-polar parameters: centre at the image centre, radius up to
    # the nearest border.
    h, w = mag1.shape
    center = (w / 2, h / 2)
    max_radius = min(center[0], center[1])   # smallest distance to edge
    lp_shape = logpolar_size

    # 5b: Log-polar transform of the magnitude spectra
    lp1 = _logpolar(mag1, center, lp_shape, max_radius)
    lp1_img = cv2.normalize(np.log(np.abs(lp1) + 1), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "LP1.png"), lp1_img)  # for visualization
    lp2 = _logpolar(mag2, center, lp_shape, max_radius)
    lp2_img = cv2.normalize(np.log(np.abs(lp2) + 1), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "LP2.png"), lp2_img)  # for visualization

    # 5c: High-pass filter the log-polar images (imreg_dft's
    # _logpolar_filter() effectively does this to remove remaining
    # low-frequency bias).
    lp1 = _highpass(lp1, sigma=highpass_sigma)
    lp1_filt_img = cv2.normalize(np.log(np.abs(lp1) + 1), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "LP1_filtered.png"), lp1_filt_img)  # for visualization
    lp2 = _highpass(lp2, sigma=highpass_sigma)
    lp2_filt_img = cv2.normalize(np.log(np.abs(lp2) + 1), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "LP2_filtered.png"), lp2_filt_img)  # for visualization

    # 5d: Perform phase correlation on the log-polar images to obtain
    # angle and scale estimates.
    angle_est, _ = _phasecorr_logpolar(lp1, lp2, lp_shape, max_radius)

    # ---------------------------------------------------------------
    # Step 6: Translation refinement (imreg_dft.imreg.translation())
    # ---------------------------------------------------------------
    # First, apply the estimated rotation/scale to the subject so that
    # it is roughly aligned to the template.  This uses the inverse
    # transform (we undo the observed rotation/scale).
    center_img = (w / 2, h / 2)
    sub_aligned = _rigid_transform(subject_f, -angle_est, 1.0, center_img)

    # Re‑apodize both images before the translation phase correlation.
    tap2 = _apodize(template_f)
    sap2 = _apodize(sub_aligned)

    # 6a: Primary translation estimate
    d_vec1, resp1 = cv2.phaseCorrelate(tap2, sap2)
    dx1, dy1 = d_vec1
    # 6b: 180° ambiguity test – rotate the aligned subject by 180°
    # and correlate again.  Because a rotation of θ and θ+180° give
    # identical Fourier magnitude, the angle estimation is ambiguous.
    sub_rot180 = cv2.rotate(sap2, cv2.ROTATE_180)
    d_vec2, resp2 = cv2.phaseCorrelate(tap2, sub_rot180)
    dx2, dy2 = d_vec2

    # 6c: Choose the hypothesis with the higher correlation peak.
    if resp1 >= resp2:
        tvec = (dx1, dy1)
        angle_final = angle_est
        success = resp1
    else:
        # When we pick the 180° rotated case, the translation vector
        # must be sign‑flipped because the rotation reverses the axes.
        tvec = (-dx2, -dy2)
        angle_final = angle_est + 180.0
        success = resp2

    angle_final %= 360.0   # keep angle in [0, 360)

    # ---------------------------------------------------------------
    # Step 7: Build result dictionary
    # ---------------------------------------------------------------
    result = {
        'angle': angle_final,
        # 'scale': scale_est,
        'trans_x': tvec[0],
        'trans_y': tvec[1],
        'success': success,
    }

    # ---------------------------------------------------------------
    # Step 8: Create the fully transformed subject image
    # (optional, but often desired for visual inspection).
    # This undoes the extending and resampling operations,
    # just like imreg_dft’s post-processing.
    # ---------------------------------------------------------------
    # Apply final rotation, scale and translation
    sub_final = _rigid_transform(subject, -angle_final, 1.0, center_img)
    sub_final = _translate(sub_final, tvec[0], tvec[1])

    # Undo upscaling (if applied) and crop to original template size
    # (unextend_by).
    # if upscale_factor is not None and upscale_factor != 1.0:
    #     orig_ow, orig_oh = int(orig_w / upscale_factor), int(orig_h / upscale_factor)
    #     sub_final = cv2.resize(sub_final, (orig_oh, orig_ow),
    #                            interpolation=cv2.INTER_LINEAR)

    sub_final = sub_final[:orig_h, :orig_w]   # remove zero‑padding
    result['t_img'] = sub_final
    M = cv2.getRotationMatrix2D(center_img, -angle_final, 1.0)
    M[0, 2] += tvec[0]  # add translation to the rotation matrix
    M[1, 2] += tvec[1]

    result['transform'] = M
    # render_compare(template_g, subject_g,sub_final, output_dir)

    return result, sub_final


# ----------------------------------------------------------------------
# Example usage (same as before)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    input_frame = r"C:\Users\dragon\Code\CatsEye-Python\output\testing_sclera\frames\frame130.png"
    off_frame = r"C:\Users\dragon\Code\CatsEye-Python\output\testing_sclera\frames\frame003.png"

    tmpl = cv2.imread(input_frame, cv2.IMREAD_GRAYSCALE)
    subj = cv2.imread(off_frame, cv2.IMREAD_GRAYSCALE)



    if tmpl is None or subj is None:
        print("Place 'template.png' and 'subject.png' in the working directory.")
    else:
        output = "output"
        output_dir = os.path.join(output, "results_" + time.strftime("%Y%m%d-%H%M%S"))
        print(f"Saving results to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        res = imreg_dft_emulate(tmpl, subj, upscale_factor=2.0, output_dir=output_dir)
        print(f"Angle: {res['angle']:.2f}°")
        # print(f"Scale: {res['scale']:.4f}")
        print(f"Translation: ({res['trans_x']}, {res['trans_y']})")
        print(f"Peak response: {res['success']:.3f}")

        cv2.imwrite(os.path.join(output_dir, "registered_subject.png"), res['t_img'].astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, "template.png"), tmpl.astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, "subject.png"), subj.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()