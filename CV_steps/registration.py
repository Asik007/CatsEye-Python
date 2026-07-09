import numpy as np
from pystackreg import StackReg
import cv2


def register_frame_stack(
        frame_stack: np.ndarray,
        output_dir: str,
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
    transform_type = 'rigid',

    reg_method = StackReg.RIGID_BODY

    # 2. Initialize StackReg with the chosen transformation
    sr = StackReg(reg_method)

    # 3. Perform registration and transformation
    print(f"Registering {len(frame_stack)} frames using '{transform_type}' alignment...")

    # uhh i guess take out only the green channel?

    frame_stack = frame_stack[:,:,:,1]

    registered_stack = sr.register_transform_stack(frame_stack, reference=reference)

    tiff_path = output_dir + r"\output_stack.tiff"

    print(f"Saving registered stack to {tiff_path}")

    success = cv2.imwritemulti(tiff_path, registered_stack.astype(np.uint8))

    print(f"Registered stack saved to {tiff_path}")

    return registered_stack



import SimpleITK as sitk

# Somehow takes longer

def register_stack_to_mean_sitk(
        frame_stack: np.ndarray,
        output_dir: str,
        channel: int = 1,
        mean_iterations: int = 3) -> np.ndarray:
    """
    Rigidly register all frames to the iterative mean of the stack using SimpleITK.

    Parameters
    ----------
    frame_stack : np.ndarray
        Input stack. Shape can be (frames, H, W) or (frames, H, W, C).
        If 4D, the specified channel is extracted.
    output_dir : str
        Directory where the output TIFF will be saved.
    channel : int
        Channel index to extract if input is 4D (default 1, green).
    mean_iterations : int
        Number of iterative mean-refinement passes (default 3).

    Returns
    -------
    np.ndarray
        Registered stack as uint8 with shape (frames, H, W).
    """
    # ---------- 1. Input validation & channel extraction ----------
    original_shape = frame_stack.shape
    if len(original_shape) == 4:
        if channel >= original_shape[3]:
            raise ValueError(f"Channel {channel} not available. Shape: {original_shape}")
        frame_stack = frame_stack[:, :, :, channel]
        print(f"Extracted channel {channel} for registration.")
    elif len(original_shape) == 3:
        pass  # already (frames, H, W)
    else:
        raise ValueError(f"Input must be 3D or 4D. Got shape {original_shape}")

    num_frames, H, W = frame_stack.shape
    print(f"Rigidly registering {num_frames} frames to iterative mean "
          f"({mean_iterations} iterations)...")

    # Convert to list of SimpleITK images (float for accurate computation)
    images = [sitk.GetImageFromArray(frame_stack[i].astype(np.float32)) for i in range(num_frames)]

    # ---------- 2. Helper: rigid registration of moving to fixed ----------
    def register_pair(fixed_img, moving_img):
        """Return an Euler2DTransform that aligns moving_img to fixed_img."""
        transform = sitk.Euler2DTransform()
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img, moving_img, transform,
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        reg = sitk.ImageRegistrationMethod()
        # Mean squares is fast and works well for same-modality images.
        # For multi-modal, swap to: reg.SetMetricAsMattesMutualInformation()
        reg.SetMetricAsMeanSquares()
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=200
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetInitialTransform(initial_transform, inPlace=False)
        return reg.Execute(fixed_img, moving_img)

    def resample_to_fixed(moving_img, transform, fixed_img):
        """Resample moving image to the grid of fixed image."""
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)
        return resampler.Execute(moving_img)

    # ---------- 3. Iterative mean registration ----------
    # Initial mean (float)
    mean_img = sitk.GetImageFromArray(np.mean(frame_stack, axis=0).astype(np.float32))
    registered_arrays = None

    for it in range(mean_iterations):
        print(f"  Iteration {it+1}/{mean_iterations}...")
        temp_arrays = []

        for moving_img in images:
            transform = register_pair(mean_img, moving_img)
            resampled = resample_to_fixed(moving_img, transform, mean_img)
            temp_arrays.append(sitk.GetArrayFromImage(resampled))

        # Update mean for next iteration, except after the last pass
        if it < mean_iterations - 1:
            mean_img = sitk.GetImageFromArray(
                np.mean(np.array(temp_arrays), axis=0).astype(np.float32)
            )
        else:
            registered_arrays = temp_arrays  # final registered frames

    # Stack results and cast to uint8 (matching original OpenCV output)
    registered_stack = np.stack(registered_arrays, axis=0).astype(np.uint8)

    # ---------- 4. Save as multi‑page TIFF ----------
    # os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + "/registered_to_mean_rigid.tiff"
    sitk.WriteImage(sitk.GetImageFromArray(registered_stack), output_path)
    print(f"Registered stack saved to {output_path}")

    return registered_stack

