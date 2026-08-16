from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessingConfig:
    """
    Configuration for the video processing pipeline.

    This dataclass holds all parameters required to process a video, including
    input sources, model settings, and output paths. It distinguishes between
    required fields (no default) and optional ones.

    Path handling behavior:
        - Fields with a default of `None` indicate that the corresponding output
          should **not** be saved.
        - Fields with a default of `""` (empty string) indicate that a default
          naming scheme should be used for the output file.
        - The `sclera_mask_path` is an exception: it cannot be `None` (it
          defaults to an empty string, implying default naming).

    Note:
        The `bad_frames` field is a mutable list. It is initialised to `None`
        to avoid the common mutable-default pitfall. A verifier class should be
        used to populate it with a list of frame indices (length equal to the
        total number of frames) after validation.

    TODO:
        Consider replacing this with a verify script that renames/adds files
        instead of relying on these path flags.
    """

    # None => don't save
    # Blank string => default naming
    # TODO: should probably change this to add a verify script and rename/add instead
    video_path: str
    output_dir: str
    best_frame: int
    seg_max_len: int = 50
    outlier_std: float = 1.0
    model_path: str = "./ML_stuff/exports/model_640_False.onnx"  # default remains
    vessel_model_path: str = "./VSX_stuff/model_DCA1.onnx"  # default remains

    # All stuff that can be inferred
    sclera_mask_path: Path = "" # except this one can't be None
    trimmed_vid_mask_path: str = ""
    sclera_overlay_path: str = ""
    trimmed_vid_overlay_color_path: str = ""

    # These can be None, but above can't be

    # this one is funky because its a mutable but I can just make it a list that is as long as the amount of frames
    # which needs a verifier class
    bad_frames: list[int] | None = None


