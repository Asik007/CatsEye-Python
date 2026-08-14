import os
import time
from pathlib import Path

import cv2
import numpy as np

from CV_steps.Helpers.cropper import crop_consistent_rgb
from CV_steps.Helpers.fileio import extract_vid_seg, save_tiff
from CV_steps.Isolate.FRUnet_infer import FRUnet
from CV_steps.Isolate.binary_cleaning import bin_pipeline, data_bin
from CV_steps.Isolate.pipeline import process_video_ml
from CV_steps.inclass import ProcessingConfig
from CV_steps.metrics import print_registration_quality
from CV_steps.registration import register_frame_stack, dumb_register, ecc_rigid_register
from CV_steps.render import vid2seg
from CV_steps.rollingball import process_rolling_ball, process_clahe


def trim_process_stabilize(config: ProcessingConfig):
    """
    Process video: trim, isolate sclera, and stabilize frames.

    Args:
        config: ProcessingConfig instance containing all required inputs.
    """
    # Use config attributes
    video_path = config.video_path
    output_dir = Path(config.output_dir)
    best_frame = config.best_frame
    model_path = config.model_path

    # Ensure output directory exists (already done in __post_init__, but kept for safety)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("Starting video processing pipeline")
    print(f"  Input video      : {video_path}")
    print(f"  Output directory : {output_dir}")
    print(f"  Best frame       : {best_frame}")
    print(f"  Model path      : {model_path}")
    print("=" * 72)

    config.sclera_mask_path = output_dir / "sclera_mask.mp4"

    print("\nRunning ML sclera isolation")
    print(f"  Sclera mask video saved to    : {config.sclera_mask_path}")

    bad_frames, frame_sizes = process_video_ml(config)


    print("Trim Video Based on Best Frame")
    print("Going to try to cut the video")

    seg_frames = vid2seg(
        config,
        bad_frames,
        frame_sizes,
    )

    print(f"  Trimmed video saving to : {config.trimmed_vid_mask_path}")

    vid_seg_rgb = extract_vid_seg(Path(config.sclera_overlay_path), seg_frames[0], seg_frames[1], output_dir)

    vid_seg_rgb = crop_consistent_rgb(vid_seg_rgb)

    print(f"Trimmed video saved to: {config.trimmed_vid_overlay_color_path}")

    clahe = process_clahe(vid_seg_rgb, 4, .05)
    save_tiff(clahe, output_dir, "clahe")

    print(f"Registering frames and saving to tiff")

    dumb_seg = dumb_register(vid_seg_rgb, output_dir)
    print("dumb reg stats:")
    # print_registration_quality(dumb_seg)
    print("ECC reg stats:")
    ecc_seg = ecc_rigid_register(dumb_seg, output_dir, border=100)
    # print_registration_quality(ecc_seg)

    print(f"Registration complete. Saved to {output_dir / 'reg_stack.tiff'}")

    print("Isolate Vessels via UNet")
    FR_net = FRUnet(output_path=output_dir)

    pred_stack = np.zeros([ecc_seg.shape[0],ecc_seg.shape[1], ecc_seg.shape[2]])
    bin_stack = np.zeros([ecc_seg.shape[0],ecc_seg.shape[1], ecc_seg.shape[2]])

    # just change the range for the full length

    # for frame in range(4):
    for i, frame in enumerate(ecc_seg):
        print(f"Processing frame {i}/{ecc_seg.shape[0] - 1}")
        _preds, bin_map = FR_net.Execute(frame[:, :, 1].astype(np.float32)) # first return is the probability map which we throwout
        _bins = data_bin(bin_map.astype(np.uint8), output_dir)
        pred_stack[i] =_preds
        print(_bins.shape)
        bin_stack[i] = _bins
        # cv2.imshow("bin_map", bin_seg)
    # save pred stack np.arraay[N, H, W] as multpagetiff
    save_tiff(pred_stack, output_dir, "pred_stack")
    save_tiff(bin_stack, output_dir, "bin_stack")





if __name__ == "__main__":
    # Example usage:
    curr_time = time.time()
    config_example = ProcessingConfig(
        video_path=r"uploads\IMG_1745.MOV",
        output_dir=r"output\testing",
        best_frame=10,
        model_path=r"ML_stuff\exports\model_640_False.onnx",
    )

    trim_process_stabilize(config_example)

    print(f"Processing completed in {curr_time - time.time()} seconds")





