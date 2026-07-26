from dataclasses import dataclass
from pathlib import Path

from torch._C import OptionalType


@dataclass
class ProcessingConfig:
    """Configuration for the video processing pipeline."""
    # None => don't save
    # Blank string => default naming
    # TODO: should probably change this to add a verify script and rename/add instead
    video_path: str
    output_dir: str
    best_frame: int
    seg_max_len: int = 50
    outlier_std: float = 1.0
    model_path: str = r"C:\Users\dragon\Code\CatsEye-Python\ML_stuff\best.pt"  # default remains

    # All stuff that can be inferred
    sclera_mask_path: Path = "" # except this one can't be None
    trimmed_vid_mask_path: str = ""
    sclera_overlay_path: str = ""
    trimmed_vid_overlay_color_path: str = ""

    # These can be None, but above can't be

    # this one is funky because its a mutable but I can just make it a list that is as long as the amount of frames
    # which needs a verifier class
    bad_frames: OptionalType[list] = None

