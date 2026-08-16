"""
Integrated Tkinter GUI for the video processing pipeline.

This GUI wires up directly to `trim_process_stabilize()` (imported from `main.py`).
It lets a user:
  - pick an input video and preview any frame from it,
  - choose an output folder (or auto-generate a timestamped one),
  - pick a detection/tracking model file (.pt / .onnx),
  - choose a "best frame" index and whether to save tracking data as TIFF,
  - run processing on a background thread without freezing the UI.

Usage:
    python video_processor_gui_tkinter.py
"""

from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from CV_steps.inclass import ProcessingConfig
from main import  trim_process_stabilize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Constants (previously scattered "magic numbers"/strings through the class)
# ──────────────────────────────────────────────────────────────────────────
WINDOW_TITLE = "Video Processor -- CatsEye"
WINDOW_SIZE = "900x780"

PREVIEW_MAX_WIDTH = 850
PREVIEW_MAX_HEIGHT = 250

VIDEO_FILETYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
    ("MP4", "*.mp4"),
    ("AVI", "*.avi"),
    ("MOV", "*.mov"),
    ("MKV", "*.mkv"),
    ("All files", "*.*"),
]

MODEL_FILETYPES = [
    ("Model files", "*.onnx"),
    ("ONNX model", "*.onnx"),
    ("All files", "*.*"),
]

# Pulled from ProcessingConfig itself so the GUI default and the pipeline
# default can never drift out of sync.
DEFAULT_MODEL_PATH = ProcessingConfig.__dataclass_fields__["model_path"].default
DEFAULT_VESSEL_MODEL_PATH = ProcessingConfig.__dataclass_fields__["vessel_model_path"].default


class VideoProcessorApp:
    """Main application window and controller."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)

        # ── State ──
        self.frame_idx = 0
        self.vid_len = 0
        self.processing = False
        self.cancel_requested = False
        self.current_photo: ImageTk.PhotoImage | None = None  # must keep a ref or Tk garbage-collects it

        self.setup_ui()

        # Confirm before closing while a background run is active.
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ────────────────────────────────────────────────────────────────
    def setup_ui(self) -> None:
        """Create and lay out all UI elements."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        self._build_io_section(main_frame, row=1)
        self._build_preview_section(main_frame, row=2)
        self._build_params_section(main_frame, row=3)
        self._build_processing_section(main_frame, row=4)
        self._build_status_bar(main_frame, row=5)

    def _build_io_section(self, parent: ttk.Frame, row: int) -> None:
        """Video input + output folder selectors."""
        input_frame = ttk.LabelFrame(parent, text="Input & Output", padding="10")
        input_frame.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        input_frame.columnconfigure(1, weight=1)

        # Video file
        ttk.Label(input_frame, text="Video file:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        self.video_path_var = tk.StringVar()
        self.video_path_ctrl = ttk.Entry(input_frame, textvariable=self.video_path_var, state="readonly")
        self.video_path_ctrl.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        ttk.Button(input_frame, text="Browse…", command=self.on_browse_video).grid(row=0, column=2)

        # Output folder
        ttk.Label(input_frame, text="Output folder:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)
        self.output_path_var = tk.StringVar()
        self.output_path_ctrl = ttk.Entry(input_frame, textvariable=self.output_path_var, state="readonly")
        self.output_path_ctrl.grid(row=1, column=1, sticky="ew", padx=(0, 5))
        ttk.Button(input_frame, text="Browse…", command=self.on_browse_output).grid(row=1, column=2)
        # "Auto" fills in a timestamped folder next to the source video so the
        # user doesn't have to create/name one by hand on every run.
        ttk.Button(input_frame, text="Auto…", width=6, command=self.on_auto_output).grid(
            row=1, column=3, padx=(5, 0)
        )

    def _build_preview_section(self, parent: ttk.Frame, row: int) -> None:
        preview_frame = ttk.LabelFrame(parent, text="Frame Preview", padding="10")
        preview_frame.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        preview_frame.columnconfigure(0, weight=1)

        self.image_panel = ttk.Label(preview_frame, background="#e0e0e0", relief="sunken", anchor="center")
        self.image_panel.grid(row=0, column=0, sticky="ew", pady=10)

    def _build_params_section(self, parent: ttk.Frame, row: int) -> None:
        params_frame = ttk.LabelFrame(parent, text="Processing Parameters", padding="10")
        params_frame.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        params_frame.columnconfigure(1, weight=1)

        # Frame index slider
        ttk.Label(params_frame, text="Best Frame Index:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        slider_frame = ttk.Frame(params_frame)
        slider_frame.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        slider_frame.columnconfigure(0, weight=1)

        self.idx_slider = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal", command=self.on_idx_change)
        self.idx_slider.grid(row=0, column=0, sticky="ew")

        self.idx_unit = ttk.Label(slider_frame, text="Frame 0", width=10)
        self.idx_unit.grid(row=0, column=1, sticky="w", padx=(10, 0))

        # Save-as-TIFF checkbox
        # NOTE: ProcessingConfig has no save_tiff field yet, so this value is
        # collected in the UI but not currently forwarded to the pipeline.
        # Add a `save_tiff: bool = False` field to ProcessingConfig once the
        # pipeline supports it, then wire it up in collect_settings().
        check_frame = ttk.Frame(params_frame)
        check_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        self.tiff_var = tk.BooleanVar()
        ttk.Checkbutton(check_frame, text="Save tracking as TIFF", variable=self.tiff_var).pack(side="left")

        # Model path -- entry + browse dialog filtered to .pt / .onnx
        ttk.Label(params_frame, text="Model path:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=5)
        model_row = ttk.Frame(params_frame)
        model_row.grid(row=2, column=1, sticky="ew")
        model_row.columnconfigure(0, weight=1)

        self.model_path_var = tk.StringVar(value=DEFAULT_MODEL_PATH)
        self.model_path_ctrl = ttk.Entry(model_row, textvariable=self.model_path_var)
        self.model_path_ctrl.grid(row=0, column=0, sticky="ew")
        ttk.Button(model_row, text="Browse…", command=self.on_browse_model).grid(row=0, column=1, padx=(5, 0))

                # Model path -- entry + browse dialog filtered to .pt / .onnx
        ttk.Label(params_frame, text="Vessel Model path:").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=5)
        model_row = ttk.Frame(params_frame)
        model_row.grid(row=3, column=1, sticky="ew")
        model_row.columnconfigure(0, weight=1)

        self.vessel_model_path_var = tk.StringVar(value=DEFAULT_VESSEL_MODEL_PATH)
        self.model_path_ctrl = ttk.Entry(model_row, textvariable=self.vessel_model_path_var)
        self.model_path_ctrl.grid(row=0, column=0, sticky="ew")
        ttk.Button(model_row, text="Browse…", command=self.on_browse_model_vessel).grid(row=0, column=1, padx=(5, 0))

    def _build_processing_section(self, parent: ttk.Frame, row: int) -> None:
        processing_frame = ttk.LabelFrame(parent, text="Processing", padding="10")
        processing_frame.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        processing_frame.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(processing_frame, mode="determinate", maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.progress_text = ttk.Label(processing_frame, text="Ready")
        self.progress_text.grid(row=1, column=0, sticky="w", pady=(0, 10))

        button_frame = ttk.Frame(processing_frame)
        button_frame.grid(row=2, column=0, sticky="ew")

        self.process_btn = ttk.Button(
            button_frame, text="▶ Start Processing", command=self.on_process_click, width=20
        )
        self.process_btn.pack(side="left", padx=5, pady=5)

    def _build_status_bar(self, parent: ttk.Frame, row: int) -> None:
        """Bottom status bar -- shows the last significant UI event."""
        self.status_text = ttk.Label(parent, text="Ready", relief="sunken", anchor="w", padding=(5, 2))
        self.status_text.grid(row=row, column=0, sticky="ew")

    # ────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ────────────────────────────────────────────────────────────────
    def on_browse_video(self) -> None:
        """Open a file dialog to select the source video and preview a frame from it."""
        file_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=VIDEO_FILETYPES,
            initialdir="./uploads/",
        )
        if not file_path:
            return

        self.video_path_var.set(file_path)
        cap = None
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise IOError("OpenCV could not open this file (unsupported codec/container?).")
            self.vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Frame count can legitimately be 0 for some containers/codecs --
            # guard against handing the slider a negative/degenerate range.
            self.idx_slider.config(to=max(self.vid_len - 1, 0))
            self.update_image_display()
            self.status_text.config(text=f"Loaded video: {Path(file_path).name} ({self.vid_len} frames)")
        except Exception as exc:
            logger.exception("Failed to load video: %s", file_path)
            messagebox.showerror("Error", f"Could not load video:\n{exc}")
            self.status_text.config(text="Failed to load video.")
        finally:
            if cap is not None:
                cap.release()

    def on_browse_output(self) -> None:
        """Open a directory dialog to select the output folder."""
        folder_path = filedialog.askdirectory(title="Select output directory", initialdir="./output/")
        if folder_path:
            self.output_path_var.set(folder_path)
            self.status_text.config(text=f"Output: {folder_path}")

    def on_auto_output(self) -> None:
        """
        Auto-generate an output folder next to the source video, named
        'results_YYYYMMDD-HHMMSS', and create it immediately.
        """
        video_path = self.video_path_var.get()
        base_dir = Path(video_path).parent.parent if video_path else Path(".")
        folder_name = f"results_{datetime.now():%Y%m%d-%H%M%S}"
        target = base_dir / "output" / folder_name

        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Error", f"Could not create output directory:\n{exc}")
            return

        self.output_path_var.set(str(target))
        self.status_text.config(text=f"Output: {target}")

    def on_browse_model(self) -> None:
        """Open a file dialog restricted to common model formats (.pt / .onnx)."""
        file_path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=MODEL_FILETYPES,
            initialdir="./ML_stuff",
        )
        if file_path:
            self.model_path_var.set(file_path)
            self.status_text.config(text=f"Model: {Path(file_path).name}")

    def on_browse_model_vessel(self) -> None:
            """Open a file dialog restricted to common model formats (.pt / .onnx)."""
            file_path = filedialog.askopenfilename(
                title="Select model file",
                filetypes=MODEL_FILETYPES,
                initialdir="./ML_stuff",
            )
            if file_path:
                self.vessel_model_path_var.set(file_path)
                self.status_text.config(text=f"Vessel Model: {Path(file_path).name}")

    def on_idx_change(self, value: str) -> None:
        """Update the frame index label/preview when the slider moves."""
        val = int(float(value))
        self.idx_unit.config(text=f"Frame {val}")
        self.frame_idx = val
        self.update_image_display()

    def update_image_display(self) -> None:
        """Extract the currently selected frame from the video and show it in the preview panel."""
        vid_path = self.video_path_var.get()
        if not vid_path or not os.path.exists(vid_path):
            return

        cap = None
        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, frame = cap.read()

            if not ret:
                self.image_panel.config(image="", text="Could not read selected frame.")
                return

            frame = self._resize_for_preview(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Keep a reference on `self` -- Tk's PhotoImage has no internal
            # strong reference of its own, so a local-only variable gets
            # garbage-collected right after this function returns and the
            # label silently goes blank.
            self.current_photo = ImageTk.PhotoImage(pil_image)
            self.image_panel.config(image=self.current_photo, text="")
        except Exception as exc:
            logger.exception("Error loading preview frame")
            self.image_panel.config(image="", text=f"Error loading frame: {exc}")
        finally:
            if cap is not None:
                cap.release()

    @staticmethod
    def _resize_for_preview(frame: np.ndarray) -> np.ndarray:
        """Scale a BGR frame down (never up) to fit inside the preview panel bounds."""
        height, width = frame.shape[:2]
        if width <= PREVIEW_MAX_WIDTH and height <= PREVIEW_MAX_HEIGHT:
            return frame
        scale = min(PREVIEW_MAX_WIDTH / width, PREVIEW_MAX_HEIGHT / height)
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(frame, new_size)

    # ────────────────────────────────────────────────────────────────
    # SETTINGS / VALIDATION
    # ────────────────────────────────────────────────────────────────
    def collect_settings(self) -> ProcessingConfig:
        """Gather all relevant widget state into a single ProcessingConfig for the pipeline."""
        return ProcessingConfig(
            video_path=self.video_path_var.get(),
            output_dir=self.output_path_var.get(),
            best_frame=self.frame_idx,
            model_path=self.model_path_var.get() or DEFAULT_MODEL_PATH,
        )

    def validate_settings(self, config: ProcessingConfig) -> str | None:
        """Return a human-readable error message if the config is invalid, else None."""
        if not config.video_path or not os.path.exists(config.video_path):
            return "Please select a valid video file."
        if not config.output_dir:
            return "Please select an output directory."
        if config.model_path and not os.path.exists(config.model_path):
            # Not necessarily fatal (the pipeline may resolve relative paths
            # differently), but worth flagging before burning time on a run.
            return f"Model path does not exist:\n{config.model_path}"
        return None

    # ────────────────────────────────────────────────────────────────
    # PROCESSING LIFECYCLE
    # ────────────────────────────────────────────────────────────────
    def on_process_click(self) -> None:
        """Validate inputs and launch processing on a background thread."""
        if self.processing:
            messagebox.showinfo("Info", "Processing already in progress.")
            return

        config = self.collect_settings()
        error = self.validate_settings(config)
        if error:
            messagebox.showerror("Error", error)
            return

        try:
            os.makedirs(config.output_dir, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Error", f"Could not create output directory:\n{exc}")
            return

        self._set_processing_state(True)

        logger.info(
            "Starting processing | video=%s output_dir=%s best_frame=%s model=%s",
            config.video_path,
            config.output_dir,
            config.best_frame,
            config.model_path,
        )

        thread = threading.Thread(target=self._run_pipeline, args=(config,), daemon=True)
        thread.start()

    def _run_pipeline(self, config: ProcessingConfig) -> None:
        """
        Runs on a background thread.

        IMPORTANT: never touch a Tk widget directly from here -- Tkinter is
        not thread-safe. Always hop back to the main thread via
        `self.root.after(...)`, as the success/error handlers below do.
        """
        try:
            # trim_process_stabilize now takes a single ProcessingConfig
            # instead of discrete positional/keyword arguments.
            trim_process_stabilize(config)
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the UI
            logger.exception("Pipeline raised an exception")
            self.root.after(0, self._on_pipeline_error, exc)
        else:
            self.root.after(0, self._on_pipeline_success)

    def _on_pipeline_success(self) -> None:
        """Runs on the main thread after a successful run."""
        self._set_processing_state(False)
        self.progress_bar["value"] = 100
        self.progress_text.config(text="Done.")
        self.status_text.config(text="Processing complete.")
        messagebox.showinfo("Done", "Processing finished successfully.")

    def _on_pipeline_error(self, exc: Exception) -> None:
        """Runs on the main thread after a failed run."""
        self._set_processing_state(False)
        self.progress_text.config(text="Failed.")
        self.status_text.config(text="Processing failed -- see error dialog.")
        messagebox.showerror("Processing failed", f"{exc}\n\n{traceback.format_exc()}")

    def _set_processing_state(self, processing: bool) -> None:
        """Toggle every control that should be locked while a run is in progress."""
        self.processing = processing
        if processing:
            self.cancel_requested = False

        self.process_btn.config(state=("disabled" if processing else "normal"))
        # self.cancel_btn.config(state=("normal" if processing else "disabled"))

        if processing:
            # We don't have a granular progress signal from the pipeline, so an
            # indeterminate ("marching ants") bar is more honest than a
            # determinate one that never actually moves.
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start(10)
            self.progress_text.config(text="Processing...")
        else:
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate")

    def on_close(self) -> None:
        """Confirm before closing the window while a background run is active."""
        if self.processing and not messagebox.askyesno("Quit", "Processing is still running. Quit anyway?"):
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    VideoProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()