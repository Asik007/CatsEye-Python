from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_parent(path: Path) -> None:
    """Create parent directories if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _open_video(path: str) -> cv2.VideoCapture:
    """Open a video file and raise if it's unreadable."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def _time_to_frame(time_s: float | None, fps: float) -> int:
    """Convert time in seconds to a frame index."""
    if time_s is None:
        return 0
    return max(0, int(round(time_s * fps)))


def _resolve_frame_range(
    cap: cv2.VideoCapture,
    start_s: float | None,
    end_s: float | None,
    start_frame: int | None,
    end_frame: int | None,
) -> tuple[int, int | None]:
    """
    Determine start and end frame indices from mixed time/frame arguments.
    Returns (start, end) where end may be None (meaning "to the end").
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 and (start_s is not None or end_s is not None):
        raise ValueError("Cannot convert time to frames because FPS is unavailable.")

    s_frame = start_frame if start_frame is not None else _time_to_frame(start_s, fps)
    e_frame = end_frame if end_frame is not None else (
        _time_to_frame(end_s, fps) if end_s is not None else None
    )

    if e_frame is not None and e_frame < s_frame:
        raise ValueError("end must be greater than or equal to start")
    return s_frame, e_frame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cut_video(
    input_path: str,
    output_path: str,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> str:
    """Extract a segment from a video and write it to a new file."""
    cap = _open_video(input_path)

    # Fallback values if FPS is unavailable (won't be used for time conversion)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    s_frame, e_frame = _resolve_frame_range(cap, start_s, end_s, start_frame, end_frame)

    _ensure_parent(Path(output_path))
    ext = Path(output_path).suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext in {".mp4", ".m4v", ".mov"} else "XVID"))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
        current = s_frame
        while True:
            if e_frame is not None and current >= e_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            current += 1
    finally:
        writer.release()
        cap.release()
    print(f"Video cut saved to {output_path}")
    return output_path


def export_frames(
    input_path: str,
    output_dir: str,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    image_format: str = "png",
    prefix: str = "frame_",
    every_n: int = 1,
) -> str:
    """Export selected frames to individual image files."""
    if every_n < 1:
        raise ValueError("every_n must be >= 1")

    cap = _open_video(input_path)
    s_frame, e_frame = _resolve_frame_range(cap, start_s, end_s, start_frame, end_frame)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fmt = image_format.lower().lstrip(".")
    if fmt not in {"png", "tif", "tiff"}:
        raise ValueError("image_format must be png, tif, or tiff")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
        current = s_frame
        saved_count = 0
        while True:
            if e_frame is not None and current >= e_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            # Save every Nth frame, counting from the first frame in the range.
            if (current - s_frame) % every_n == 0:
                out_file = output / f"{prefix}{current:06d}.{fmt}"
                cv2.imwrite(str(out_file), frame)
                saved_count += 1
            current += 1
    finally:
        cap.release()
    print(f"Exported {saved_count} frames to {output}")
    return str(output)


def export_tiff_stack(
    input_path: str,
    output_path: str,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> str:
    """Export selected frames to a single multi-page TIFF stack."""
    cap = _open_video(input_path)
    s_frame, e_frame = _resolve_frame_range(cap, start_s, end_s, start_frame, end_frame)

    _ensure_parent(Path(output_path))
    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
        current = s_frame
        while True:
            if e_frame is not None and current >= e_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            current += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError("No frames were read from the input video")

    if not cv2.imwritemulti(output_path, frames):
        raise RuntimeError("Could not write multi-page TIFF stack")
    print(f"TIFF stack with {len(frames)} frames saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_frame_range_args(parser: argparse.ArgumentParser) -> None:
    """Add shared arguments for specifying a frame range."""
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--start-frame", type=int, default=None, help="Start frame index (0-based)")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index (exclusive)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video cut/export utilities using OpenCV")
    '''
    Pretty much tell it if you want to cut/export frames/export tiff stack, 
    then specify the input video and output path, 
    and optionally specify the frame range using either time or frame indices.
    For exporting frames, you can also choose the
    image format, filename prefix, and how frequently to save frames.
    '''

    sub = parser.add_subparsers(dest="command", required=True)

    # cut
    p_cut = sub.add_parser("cut", help="Cut a segment from a video into a new video")
    p_cut.add_argument("input", help="Input video file")
    p_cut.add_argument("output", help="Output video file")
    _add_frame_range_args(p_cut)


    # frames
    p_frames = sub.add_parser("frames", help="Export frames to a folder")
    p_frames.add_argument("input", help="Input video file")
    p_frames.add_argument("output_dir", help="Output folder")
    _add_frame_range_args(p_frames)
    p_frames.add_argument("--format", default="png", choices=["png", "tif", "tiff"], help="Output image format")
    p_frames.add_argument("--prefix", default="frame_", help="Filename prefix")
    p_frames.add_argument("--every-n", type=int, default=1, help="Save every Nth frame")

    # stack
    p_stack = sub.add_parser("stack", help="Export frames to a single multi-page TIFF stack")
    p_stack.add_argument("input", help="Input video file")
    p_stack.add_argument("output", help="Output TIFF file")
    _add_frame_range_args(p_stack)


    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "cut":
        cut_video(args.input, args.output, args.start, args.end, args.start_frame, args.end_frame)
    elif args.command == "frames":
        export_frames(
            args.input,
            args.output_dir,
            args.start,
            args.end,
            args.start_frame,
            args.end_frame,
            args.format,
            args.prefix,
            args.every_n,
        )
    elif args.command == "stack":
        export_tiff_stack(args.input, args.output, args.start, args.end, args.start_frame, args.end_frame)

    return 0


if __name__ == "__main__":
    main()
    raise SystemExit()