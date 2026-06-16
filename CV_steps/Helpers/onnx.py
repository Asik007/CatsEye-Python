"""
Export a YOLO model to ONNX for multiple image sizes, with optional cleanup.
Can be used as both a CLI tool and an importable function.
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Union

from ultralytics import YOLO


def export_yolo_to_onnx(
    model_path: Union[str, Path],
    img_sizes: List[int] = (640, 320, 160),
    half: bool = False,
    export_root: Optional[Union[str, Path]] = None,
    dynamic: bool = False,
    simplify: bool = True,
    opset: int = 12,
    format: str = "onnx",
    cleanup: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    **extra_export_kwargs,
) -> List[Path]:
    """
    Export a YOLO model to the specified format for several image sizes,
    optionally cleaning up temporary folders and consolidating outputs.

    Args:
        model_path: Path to the .pt model file.
        img_sizes: List of image sizes (int) for which to export.
        half: Enable FP16 half-precision export.
        export_root: Directory where per‑size export subdirectories are created.
                     Defaults to '<model_dir>/exports'.
        dynamic: Enable dynamic ONNX axes.
        simplify: Simplify ONNX model.
        opset: ONNX opset version.
        format: Export format (default 'onnx').
        cleanup: If True, move all exported files to a single folder
                 (output_dir) and delete the temporary per‑size folders.
        output_dir: Target folder when cleanup=True (defaults to export_root).
        **extra_export_kwargs: Additional arguments passed to model.export().

    Returns:
        List of paths to the final exported files (after cleanup if requested).

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if export_root is None:
        export_root = model_path.parent / "exports"
    else:
        export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    exported_paths = []
    final_paths = []

    for sz in img_sizes:
        export_dir = export_root / f"model_{sz}_{half}"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Work on a copy to avoid locking / modifying the original file
        copied_model_path = export_dir / model_path.name
        shutil.copy2(model_path, copied_model_path)

        model = YOLO(str(copied_model_path))
        fin_name = f"model_{sz}_{half}.{format}"
        exported_path = model.export(
            format=format,
            imgsz=sz,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            project=str(export_dir),
            name= fin_name,
            exist_ok=True,
            **extra_export_kwargs,
        )

        print(f"Exported {sz}px model to: {exported_path} -> {fin_name}")
        exported_paths.append(Path(exported_path))
        final_paths.append(fin_name)

    if cleanup:
        # Determine the single output folder
        final_output_dir = Path(output_dir) if output_dir else export_root
        final_output_dir.mkdir(parents=True, exist_ok=True)

        cleaned_paths = []
        for i, orig_path in enumerate(exported_paths):
            # Destination filename is already named correctly
            dest_path = final_output_dir / final_paths[i]
            if dest_path.exists():
                # If file already exists (e.g. from a previous run), overwrite
                dest_path.unlink()
            shutil.move(str(orig_path), str(dest_path))
            cleaned_paths.append(dest_path)

            # Remove the temporary per‑size folder (contains the copied .pt, etc.)
            temp_dir = orig_path.parent
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Optionally remove empty export_root if it only contained the temp dirs
        # (not strictly necessary, but keeps things tidy)
        if export_root != final_output_dir:
            try:
                export_root.rmdir()  # only if empty
            except OSError:
                pass

        # Update returned list with the final paths
        exported_paths = cleaned_paths
        print(f"Cleanup complete. All exports moved to: {final_output_dir}")

    return exported_paths


def main():
    parser = argparse.ArgumentParser(
        description="Export a YOLO model to ONNX for multiple image sizes."
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the .pt model file",
    )
    parser.add_argument(
        "--img-sizes",
        nargs="+",
        type=int,
        default=[640, 320, 160],
        help="Image sizes to export (default: 640 320 160)",
    )
    parser.add_argument(
        "--half", action="store_true", help="Use FP16 half precision"
    )
    parser.add_argument(
        "--export-root",
        type=str,
        default=None,
        help="Root directory for per-size exports (default: <model_dir>/exports)",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Export with dynamic axes"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_false",
        dest="simplify",
        help="Disable ONNX simplification",
    )
    parser.add_argument(
        "--opset", type=int, default=12, help="ONNX opset version (default: 12)"
    )
    parser.add_argument(
        "--format",
        default="onnx",
        help="Export format (default: onnx)",
    )
    # Cleanup options
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="After export, move all ONNX files to a single folder and delete temporary folders/weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Single output folder when using --cleanup (default: same as --export-root)",
    )

    args = parser.parse_args()

    export_yolo_to_onnx(
        model_path=args.model_path,
        img_sizes=args.img_sizes,
        half=args.half,
        export_root=args.export_root,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        format=args.format,
        cleanup=args.cleanup,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()