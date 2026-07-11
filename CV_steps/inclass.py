from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    """Configuration for the video processing pipeline."""
    video_path: str
    output_dir: str
    best_frame: int
    model_path: str = r"C:\Users\dragon\Code\CatsEye-Python\ML_stuff\best.pt"  # default remains
