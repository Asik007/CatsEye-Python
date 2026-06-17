"""
Integrated Toga GUI for the video processing pipeline.
This version directly connects to process_and_stabilize() and other pipeline functions.

Usage:
  python video_processor_gui_integrated.py
"""

import os
import cv2
import time
from PIL import Image
import asyncio
import threading
from pathlib import Path
from datetime import datetime

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, LEFT, RIGHT, CENTER

# Import your pipeline functions
# try:
from main import process_and_stabilize

class VideoProcessorApp(toga.App):
    frame_idx = 0
    vid_len = 0

    def startup(self):
        """Create the main application window and construct the UI."""
        self.main_window = toga.MainWindow(title="Video Processor -- CatsEye", size=(900, 750))

        self.processing = False

        # ────────────────────────────────────────────────────────────────
        # UI CONSTRUCTION
        # ────────────────────────────────────────────────────────────────
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10))

        # Header
        header = toga.Label(
            "Video Processing Pipeline",
            style=Pack(font_family="sans-serif", font_weight="bold", font_size=16, padding_bottom=10)
        )
        main_box.add(header)

        # -- Input Section --
        input_box = toga.Box(style=Pack(direction=COLUMN, padding_bottom=15))

        # Video File
        video_row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        video_row.add(toga.Label("Video file:", style=Pack(width=100, padding_top=5)))
        self.video_path_ctrl = toga.TextInput(readonly=True, style=Pack(flex=1, padding_right=5))
        video_browse_btn = toga.Button("Browse…", on_press=self.on_browse_video)
        video_row.add(self.video_path_ctrl)
        video_row.add(video_browse_btn)

        # Output Folder
        output_row = toga.Box(style=Pack(direction=ROW))
        output_row.add(toga.Label("Output folder:", style=Pack(width=100, padding_top=5)))
        self.output_path_ctrl = toga.TextInput(readonly=True, style=Pack(flex=1, padding_right=5))
        output_browse_btn = toga.Button("Browse…", on_press=self.on_browse_output)
        output_row.add(self.output_path_ctrl)
        output_row.add(output_browse_btn)

        input_box.add(video_row)
        input_box.add(output_row)
        main_box.add(input_box)

        # -- Image Preview Section --
        self.image_panel = toga.ImageView(style=Pack(flex=1, height=250, padding_bottom=15))
        main_box.add(self.image_panel)

        # -- Parameters Section --
        params_box = toga.Box(style=Pack(direction=COLUMN, padding_bottom=15))

        # Smoothing Slider
        idx_row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        idx_row.add(toga.Label("Best Frame Index:", style=Pack(width=130, padding_top=5)))
        self.idx_slider = toga.Slider(min=0, max=self.vid_len, value=self.frame_idx, on_release=self.on_idx_release, style=Pack(flex=1, padding_right=10))
        self.idx_unit = toga.Label("Frame 50", style=Pack(width=80, padding_top=5))
        idx_row.add(self.idx_slider)
        idx_row.add(self.idx_unit)

        # Checkboxes (Toga uses Switches for booleans)
        check_row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        # self.debug_checkbox = toga.Switch("Debug mode", style=Pack(padding_right=20))
        self.tiff_checkbox = toga.Switch("Save tracking as TIFF")
        # check_row.add(self.debug_checkbox)
        check_row.add(self.tiff_checkbox)

        # ML Model
        ml_row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        ml_row.add(toga.Label("Model path:", style=Pack(width=100, padding_top=5)))
        self.model_path_ctrl = toga.TextInput(value="ML_stuff/best.pt", style=Pack(flex=1))
        ml_row.add(self.model_path_ctrl)

        # Confidence
        # conf_row = toga.Box(style=Pack(direction=ROW))
        # conf_row.add(toga.Label("Confidence:", style=Pack(width=100, padding_top=5)))
        # self.conf_ctrl = toga.NumberInput(min=1, max=100, step=1, value=25)
        # conf_row.add(self.conf_ctrl)
        # conf_row.add(toga.Label("%", style=Pack(padding_left=5, padding_top=5)))

        params_box.add(idx_row)
        params_box.add(check_row)
        params_box.add(ml_row)
        # params_box.add(conf_row)
        main_box.add(params_box)

        # -- Processing Section --
        processing_box = toga.Box(style=Pack(direction=COLUMN, padding_bottom=15))

        self.progress_bar = toga.ProgressBar(max=100, style=Pack(padding_bottom=5))
        self.progress_text = toga.Label("Ready", style=Pack(padding_bottom=5))

        btn_row = toga.Box(style=Pack(direction=ROW, alignment=CENTER))
        self.process_btn = toga.Button("▶ Start Processing", on_press=self.on_process_click,
                                       style=Pack(width=150, padding=5))
        btn_row.add(self.process_btn)

        processing_box.add(self.progress_bar)
        processing_box.add(self.progress_text)
        processing_box.add(btn_row)
        main_box.add(processing_box)

        # # -- Output Log Section --
        # log_box = toga.Box(style=Pack(direction=COLUMN, flex=1, padding_bottom=10))
        # self.log_ctrl = toga.MultilineTextInput(readonly=True, style=Pack(flex=1, font_family="monospace"))
        #
        # clear_btn = toga.Button("Clear log", on_press=lambda w: setattr(self.log_ctrl, 'value', ""),
        #                         style=Pack(width=100, padding_top=5))
        #
        # log_box.add(self.log_ctrl)
        # log_box.add(clear_btn)
        # main_box.add(log_box)

        # -- Status Footer --
        self.status_text = toga.Label("Ready", style=Pack(padding_top=5, padding_bottom=5))
        main_box.add(toga.Divider())
        main_box.add(self.status_text)

        self.main_window.content = main_box
        self.main_window.show()

    # ────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ────────────────────────────────────────────────────────────────

    async def on_browse_video(self, widget):
        """Open async file dialog to select video."""
        try:
            ask_file_path = toga.OpenFileDialog("Select Video to Process", ".")
            # Toga dialogs are asynchronous
            file_path = await self.main_window.open_file_dialog(
                "Select video file",
                file_types=['mp4', 'avi', 'mov', 'mkv']
            )
            if file_path:
                self.video_path_ctrl.value = str(file_path)
                cap = cv2.VideoCapture(file_path)
                self.vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.idx_slider.max = self.vid_len
                cap.release()
                self.update_image_display()

        except ValueError:
            pass  # User canceled the dialog

    async def on_browse_output(self, widget):
        """Open async directory dialog to select output folder."""
        try:
            folder_path = await self.main_window.select_folder_dialog("Select output directory")
            if folder_path:
                self.output_path_ctrl.value = str(folder_path)
        except ValueError:
            pass  # User canceled

    def on_idx_release(self, widget):
        """Update status when index selection changes."""
        val = int(widget.value)
        self.idx_unit.text = f"Frame {val}"
        self.frame_idx = val
        print(f"slider index: {val}")
        # self.status_text.text = f"Smoothing radius: {val} pixels"
        self.update_image_display()

    def update_image_display(self):
        """Extract a frame from the video and set it to the ImageView."""
        vid_path = self.video_path_ctrl.value
        if not vid_path or not os.path.exists(vid_path):
            return
        cap = cv2.VideoCapture(vid_path)
        # self.vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        # self.idx_slider.min = self.vid_len
        ret, frame = cap.read()
        cap.release()

        if ret:
            print("writing new image")
            self.image_panel.image = toga.Image(Image.fromarray(frame))
        else:
            print("Could not read selected frame.")

    async def on_process_click(self, widget):
        """Handle process button click. Executes validation, then launches thread."""
        if self.processing:
            await self.main_window.info_dialog("Info", "Processing already in progress.")
            return

        # Validate inputs
        video_path = self.video_path_ctrl.value
        output_path = self.output_path_ctrl.value

        if not video_path or not os.path.exists(video_path):
            await self.main_window.error_dialog("Error", "Please select a valid video file.")
            return

        if not output_path:
            await self.main_window.error_dialog("Error", "Please select an output directory.")
            return

        try:
            os.makedirs(output_path, exist_ok=True)
        except Exception as e:
            await self.main_window.error_dialog("Error", f"Could not create output directory:\n{e}")
            return

        # Disable controls
        self.processing = True
        self.process_btn.enabled = False
        self.progress_bar.value = 0

        print(f"\n{'=' * 60}")
        print("Starting processing…")
        print(f"Video: {video_path}")
        print(f"Output: {output_path}")
        print(f"Smoothing radius: {int(self.idx_slider.value)}")
        print(f"{'=' * 60}\n")

        # Call ML pipeline
        process_and_stabilize(
            video_path=video_path,
            # model_path=model_path,
            output_dir=output_path,
        )


def main():
    return VideoProcessorApp("Video Processor", "org.beeware.videoprocessor")


if __name__ == "__main__":
    main().main_loop()