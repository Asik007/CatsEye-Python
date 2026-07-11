import os
from pathlib import Path

from CV_steps.Helpers.cropper import crop_consistent_rgb
from CV_steps.Helpers.fileio import extract_vid_seg
from CV_steps.Isolate.pipeline import process_video_ml
from CV_steps.inclass import ProcessingConfig
from CV_steps.registration import register_frame_stack
from CV_steps.render import vid2seg




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
    print("=" * 72)

    overlay_path = output_dir / "sclera_overlay.mp4"
    mask_path = output_dir / "sclera_mask.mp4"

    print("\nRunning ML sclera isolation")
    print(f"  Sclera overlay video saved to : {overlay_path}")
    print(f"  Sclera mask video saved to    : {mask_path}")

    bad_frames, frame_sizes = process_video_ml(
        video_path=video_path,
        model_path=model_path,
        output_mask_path=str(mask_path),
        output_overlay_path=str(overlay_path),
    )

    print("\nTrim Video Based on Best Frame")
    print("\nGoing to try to cut the video")

    trimmed_vid_path = output_dir / "trimmed_video.mp4"
    print(f"  Trimmed video saving to : {trimmed_vid_path}")

    seg_frames = vid2seg(
        str(mask_path),
        bad_frames,
        frame_sizes,
        str(trimmed_vid_path),
        best_frame_idx=best_frame,
    )

    vid_seg_rgb = extract_vid_seg(overlay_path, seg_frames[0], seg_frames[1], output_dir)

    vid_seg_rgb = crop_consistent_rgb(vid_seg_rgb)

    print(f"Trimmed video saved to: {trimmed_vid_path}")

    print(f"Registering frames and saving to tiff")

    # reg_seg = run_all_methods(vid_seg_rgb, reference='mean', output_dir=output_dir)
    # Alternative commented out:
    reg_seg = register_frame_stack(vid_seg_rgb, output_dir)
