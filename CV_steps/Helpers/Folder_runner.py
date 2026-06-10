"""Batch runner for CV_steps.Register.registration.

Runs the registration CLI once for each video file in a folder and writes
each result into its own output subdirectory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from CV_steps.Register.registration import chosen_pipeline


DEFAULT_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}


def iter_input_files(input_dir: Path, recursive: bool, extensions: set[str]) -> list[Path]:
	if recursive:
		candidates = (path for path in input_dir.rglob("*") if path.is_file())
	else:
		candidates = (path for path in input_dir.iterdir() if path.is_file())

	return sorted(
		path for path in candidates if not extensions or path.suffix.lower() in extensions
	)


def run_registration_for_folder(
	input_dir: Path,
	output_root: Path,
	recursive: bool = False,
	extensions: set[str] | None = None,
) -> int:
	extensions = extensions or DEFAULT_EXTENSIONS
	input_files = iter_input_files(input_dir, recursive=recursive, extensions=extensions)

	if not input_files:
		print(f"No matching video files found in {input_dir}")
		return 1

	output_root.mkdir(parents=True, exist_ok=True)

	
	for file_path in input_files:
		output_dir = output_root / file_path.stem
		output_dir.mkdir(parents=True, exist_ok=True)

		chosen_pipeline(str(file_path), str(output_dir))

	print(f"\nFinished registration for {len(input_files)} file(s).")
	return 0


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run the registration pipeline on every video in a folder."
	)
	parser.add_argument(
		"input_dir",
		nargs="?",
		default="uploads/Unstabilized",
		help="Folder containing the videos to process.",
	)
	parser.add_argument(
		"--output-root",
		default="output/Unstabilized",
		help="Root folder where per-file result folders will be created.",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Search for video files recursively under the input folder.",
	)
	parser.add_argument(
		"--extensions",
		nargs="*",
		default=None,
		help="Optional list of file extensions to include, for example .mp4 .avi .mov.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	input_dir = Path(args.input_dir).expanduser().resolve()
	output_root = Path(args.output_root).expanduser().resolve()

	if not input_dir.exists():
		print(f"Input folder does not exist: {input_dir}")
		return 1

	extensions = None
	if args.extensions is not None:
		extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}

	return run_registration_for_folder(
		input_dir=input_dir,
		output_root=output_root,
		recursive=args.recursive,
		extensions=extensions,
	)


if __name__ == "__main__":
	main()
	raise SystemExit()

