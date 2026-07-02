"""
Toga GUI for the video processing pipeline.

Built on the framework-agnostic state/logic in video_core.py -- this file
only builds widgets and reacts to AppState changes.

Usage:
    python video_gui_toga.py
"""

import asyncio
import os
from typing import Optional

import toga
from toga.style import Pack
from toga.style.pack import CENTER, COLUMN, ROW

from main import process_and_stabilize
# from video_core import AppState, ProcessingStatus, VideoPreview, run_pipeline, validate


class VideoProcessorApp(toga.App):
    def startup(self):
        self.state = AppState()
        self.preview: Optional[VideoPreview] = None
        self.on_exit = self.handle_exit

        self.main_window = toga.MainWindow(title="Video Processor -- CatsEye", size=(900, 750))

        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10))
        main_box.add(self._build_header())
        main_box.add(self._build_input_section())

        self.image_panel = toga.ImageView(style=Pack(flex=1, height=250, padding_bottom=15))
        main_box.add(self.image_panel)

        main_box.add(self._build_params_section())
        main_box.add(self._build_processing_section())

        self.status_label = toga.Label(self.state.status_message, style=Pack(padding_top=5, padding_bottom=5))
        main_box.add(toga.Divider())
        main_box.add(self.status_label)

        self.main_window.content = main_box
        self.main_window.show()

    # ------------------------------------------------------------------
    # WIDGET CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_header(self):
        return toga.Label(
            "Video Processing Pipeline",
            style=Pack(font_family="sans-serif", font_weight="bold", font_size=16, padding_bottom=10),
        )

    def _build_input_section(self):
        input_box = toga.Box(style=Pack(direction=COLUMN, padding_bottom=15))

        video_row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        video_row.add(toga.Label("Video file:", style=Pack(width=100, padding_top=5)))
        self.video_path_ctrl = toga.TextInput(readonly=True, style=Pack(flex=1, padding_right=5))
        video_row.add(self.video_path_ctrl)
        video_row.add(toga.Button("Browse…", on_press=self.on_browse_video))

        output_row = toga.Box(style=Pack(direction=ROW))
        output_row.add(toga.Label("Output folder:", style=Pack(width=100, padding_top=5)))
        self.output_path_ctrl = toga.TextInput(readonly=True, style=Pack(flex=1, padding_right=5))
        output_row.add(self.output_path_ctrl)
        output_row.add(toga.Button("Browse…", on_press=self.on_browse_output))

        input_box.add(video_row)
        input_box.add(output_row)
        return input_box

    def _build_params_section(self):
        params_box = toga.Box(style=Pack(direction=COLUMN, padding_bottom=15))

        idx_row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        idx_row.add(toga.Label("Best Frame Index:", style=Pack(width=130, padding_top=5)))
        self.idx_slider = toga.Slider(
            min=0, max=0, value=0, on_release=self.on_idx_release, style=Pack(flex=1, padding_right=10)
        )
        self.idx_label = toga.Label("Frame 0", style=Pack(width=80, padding_top=5))
        idx_row.add(self.idx_slider)
        idx_row.add(self.idx_label)

        check_row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        self.tiff_switch = toga.Switch("Save tracking as TIFF")
        check_row.add(self.tiff_switch)

        ml_row = toga.Box(style=Pack(direction=ROW, padding_bottom=5))
        ml_row.add(toga.Label("Model path:", style=Pack(width=100, padding_top=5)))
        self.model_path_ctrl = toga.TextInput(value=self.state.model_path, style=Pack(flex=1))
        ml_row.add(self.model_path_ctrl)

        params_box.add(idx_row)
        params_box.add(check_row)
        params_box.add(ml_row)
        return params_box

    def _build_processing_section(self):
        processing_box = toga.Box(style=Pack(direction=COLUMN, padding_bottom=15))

        self.progress_bar = toga.ProgressBar(max=100, style=Pack(padding_bottom=5))
        self.progress_label = toga.Label("Ready", style=Pack(padding_bottom=5))

        btn_row = toga.Box(style=Pack(direction=ROW, alignment=CENTER))
        self.process_btn = toga.Button(
            "▶ Start Processing", on_press=self.on_process_click, style=Pack(width=150, padding=5)
        )
        btn_row.add(self.process_btn)

        processing_box.add(self.progress_bar)
        processing_box.add(self.progress_label)
        processing_box.add(btn_row)
        return processing_box

    # ------------------------------------------------------------------
    # EVENT HANDLERS
    # ------------------------------------------------------------------

    async def on_browse_video(self, widget):
        try:
            path = await self.main_window.open_file_dialog(
                "Select video file", file_types=["mp4", "avi", "mov", "mkv"]
            )
        except ValueError:
            return  # user canceled
        if not path:
            return
        path = str(path)

        try:
            new_preview = VideoPreview(path)
        except ValueError as e:
            await self.main_window.error_dialog("Error", str(e))
            return

        if self.preview is not None:
            self.preview.close()
        self.preview = new_preview

        self.state.video_path = path
        self.state.frame_count = self.preview.frame_count
        self.state.frame_idx = 0

        self.video_path_ctrl.value = path
        self.idx_slider.max = max(self.state.frame_count - 1, 1)
        self.idx_slider.value = 0
        self.idx_label.text = "Frame 0"

        self._show_frame(0)

    async def on_browse_output(self, widget):
        try:
            path = await self.main_window.select_folder_dialog("Select output directory")
        except ValueError:
            return  # user canceled
        if path:
            self.state.output_path = str(path)
            self.output_path_ctrl.value = str(path)

    def on_idx_release(self, widget):
        idx = int(widget.value)
        self.state.frame_idx = idx
        self.idx_label.text = f"Frame {idx}"
        self._show_frame(idx)

    def _show_frame(self, idx: int):
        if self.preview is None:
            return
        image = self.preview.get_frame(idx)
        if image is None:
            return
        self.image_panel.image = toga.Image(image)

    async def on_process_click(self, widget):
        if self.state.status == ProcessingStatus.RUNNING:
            await self.main_window.info_dialog("Info", "Processing already in progress.")
            return

        self.state.model_path = self.model_path_ctrl.value
        self.state.save_tiff = self.tiff_switch.value

        errors = validate(self.state)
        if errors:
            await self.main_window.error_dialog("Error", "\n".join(errors))
            return

        try:
            os.makedirs(self.state.output_path, exist_ok=True)
        except OSError as e:
            await self.main_window.error_dialog("Error", f"Could not create output directory:\n{e}")
            return

        self.state.status = ProcessingStatus.RUNNING
        self.process_btn.enabled = False
        self.progress_bar.value = 0
        self.progress_label.text = "Processing…"
        self.status_label.text = "Processing…"

        loop = asyncio.get_event_loop()
        try:
            # process_and_stabilize() is blocking, so it runs on a worker
            # thread to keep the UI responsive.
            await loop.run_in_executor(None, run_pipeline, process_and_stabilize, self.state)
        except Exception as e:
            self.state.status = ProcessingStatus.ERROR
            await self.main_window.error_dialog("Processing failed", str(e))
            self.progress_label.text = "Failed"
            self.status_label.text = "Processing failed."
            self.progress_bar.value = 0
        else:
            self.state.status = ProcessingStatus.SUCCESS
            self.progress_label.text = "Done"
            self.status_label.text = "Processing complete."
            self.progress_bar.value = 100
        finally:
            self.process_btn.enabled = True

    def handle_exit(self) -> bool:
        if self.preview is not None:
            self.preview.close()
        return True


def main():
    return VideoProcessorApp("Video Processor", "org.beeware.videoprocessor")


if __name__ == "__main__":
    main().main_loop()