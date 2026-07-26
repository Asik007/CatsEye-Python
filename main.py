import os
from pathlib import Path

from CV_steps.Helpers.cropper import crop_consistent_rgb
from CV_steps.Helpers.fileio import extract_vid_seg, save_tiff
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

    # reg_seg = run_all_methods(vid_seg_rgb, reference='mean', output_dir=output_dir)
    # Alternative commented out:
    # reg_seg = register_frame_stack(vid_seg_rgb, output_dir)
    # print("stack reg stats:")
    # print_registration_quality(reg_seg)
    # save_tiff(reg_seg, output_dir, "reg_stack")

    dumb_seg = dumb_register(vid_seg_rgb, output_dir)
    print("dumb reg stats:")
    print_registration_quality(dumb_seg)
    print("ECC reg stats:")
    ecc_seg = ecc_rigid_register(dumb_seg, output_dir, border=100)
    print_registration_quality(ecc_seg)

    print(f"Registration complete. Saved to {output_dir / 'reg_stack.tiff'}")


    print(f"Background Elimination starting")
    
    # bac_seg = process_rolling_ball(ecc_seg, 50, True)
    
    print(f"Background Elimination complete.")
    
    # save_tiff(bac_seg, output_dir, "bac")
    
    print(f"Background Elimination saved to {output_dir / 'bac_stack.tiff'}")


if __name__ == "__main__":
    # Example usage:
    config_example = ProcessingConfig(
        video_path=r"C:\Users\dragon\Code\CatsEye-Python\uploads\IMG_1745.MOV",
        output_dir=r"C:\Users\dragon\Code\CatsEye-Python\output\testing",
        best_frame=10,
        model_path=r"C:\Users\dragon\Code\CatsEye-Python\ML_stuff\exports\model_640_False.onnx",
    )

    trim_process_stabilize(config_example)





