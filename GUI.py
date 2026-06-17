"""
Integrated wxPython GUI for the video processing pipeline.
This version directly connects to process_and_stabilize() and other pipeline functions.

Usage:
  python video_processor_gui_integrated.py

Requirements:
  pip install wxPython
"""

import wx
import os
import threading
import cv2
import time
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

# Import your pipeline functions
# try:
from CV_steps.stabilize_frame import stabilize_video
from CV_steps.Isolate.pipeline import process_video_ml

IMPORTS_AVAILABLE = True
# except ImportError as e:
#     IMPORTS_AVAILABLE = False
#     IMPORT_ERROR = str(e)


class VideoProcessorFrame(wx.Frame):
    """Main application window for the video processing pipeline."""

    frame_idx = 0
    frame_img = wx.NullBitmap
    def __init__(self):
        super().__init__(
            parent=None,
            title="Video Processor -- CatsEye",
            size=wx.Size(900, 750)
        )

        self.processing = False
        self.worker_thread: Optional[threading.Thread] = None
        # self.frame_index = 0
        # self.image_panel: Optional[wx.StaticBitmap] = None
        # self.original_bitmap: Optional[wx.Bitmap] = None

        # Create the UI
        self._create_ui()
        self._set_defaults()
        self._bind_events()

        # Check imports and warn if needed
        if not IMPORTS_AVAILABLE:
            self._log(f"⚠ Warning: Pipeline modules not available:\n")
            self._log("  You can still use this interface, but processing will fail.")
            self._log("  Ensure all dependencies are installed and paths are correct.\n")

        # Center on screen
        self.Centre()

    def _create_ui(self):
        """Build the user interface."""
        main_panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # ────────────────────────────────────────────────────────────────
        # HEADER
        # ────────────────────────────────────────────────────────────────
        header_font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        header = wx.StaticText(main_panel, label="Video Processing Pipeline")
        header.SetFont(header_font)
        main_sizer.Add(header, 0, wx.ALL, 12)

        # ────────────────────────────────────────────────────────────────
        # INPUT SECTION
        # ────────────────────────────────────────────────────────────────
        input_box = wx.StaticBoxSizer(wx.VERTICAL, main_panel, "Input")

        # Video file selection
        video_hbox = wx.BoxSizer(wx.HORIZONTAL)
        video_label = wx.StaticText(main_panel, label="Video file:", size=(100, -1))
        self.video_path_ctrl = wx.TextCtrl(main_panel, style=wx.TE_READONLY)
        video_browse_btn = wx.Button(main_panel, label="Browse…")
        video_browse_btn.Bind(wx.EVT_BUTTON, self._on_browse_video)

        video_hbox.Add(video_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        video_hbox.Add(self.video_path_ctrl, 1, wx.EXPAND | wx.RIGHT, 8)
        video_hbox.Add(video_browse_btn, 0, wx.EXPAND)
        input_box.Add(video_hbox, 0, wx.EXPAND | wx.ALL, 8)

        # Output directory selection
        output_hbox = wx.BoxSizer(wx.HORIZONTAL)
        output_label = wx.StaticText(main_panel, label="Output folder:", size=(100, -1))
        self.output_path_ctrl = wx.TextCtrl(main_panel, style=wx.TE_READONLY)
        output_browse_btn = wx.Button(main_panel, label="Browse…")
        output_browse_btn.Bind(wx.EVT_BUTTON, self._on_browse_output)

        output_hbox.Add(output_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        output_hbox.Add(self.output_path_ctrl, 1, wx.EXPAND | wx.RIGHT, 8)
        output_hbox.Add(output_browse_btn, 0, wx.EXPAND)
        input_box.Add(output_hbox, 0, wx.EXPAND | wx.ALL, 8)

        main_sizer.Add(input_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        # ────────────────────────────────────────────────────────────────
        # IMAGE DISPLAY SECTION
        # ────────────────────────────────────────────────────────────────
        image_box = wx.StaticBoxSizer(wx.VERTICAL, main_panel, "Frame Preview")

        self.image_panel = wx.StaticBitmap(main_panel, size=(320, 320), bitmap=self.frame_img)
        self.image_panel.SetBackgroundColour(wx.Colour(50, 50, 50))
        image_box.Add(self.image_panel, 1, wx.EXPAND | wx.ALL, 8)

        main_sizer.Add(image_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        # ────────────────────────────────────────────────────────────────
        # PARAMETERS SECTION
        # ────────────────────────────────────────────────────────────────
        params_box = wx.StaticBoxSizer(wx.VERTICAL, main_panel, "Parameters")

        # Smoothing radius slider
        smooth_hbox = wx.BoxSizer(wx.HORIZONTAL)
        smooth_label = wx.StaticText(main_panel, label="Smoothing radius:", size=(120, -1))
        self.idx_slider = wx.Slider(
            main_panel,
            value=50,
            minValue=1,
            maxValue=150,
            size=(200, -1),
            style=wx.SL_HORIZONTAL | wx.SL_LABELS
        )
        self.idx_slider.Bind(wx.EVT_SLIDER, self._on_idx_changed)
        smooth_unit = wx.StaticText(main_panel, label="pixels")

        smooth_hbox.Add(smooth_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        smooth_hbox.Add(self.idx_slider, 1, wx.EXPAND | wx.RIGHT, 8)
        smooth_hbox.Add(smooth_unit, 0, wx.ALIGN_CENTER_VERTICAL)
        params_box.Add(smooth_hbox, 0, wx.EXPAND | wx.ALL, 8)

        # Checkboxes
        checkbox_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.debug_checkbox = wx.CheckBox(main_panel, label="Debug mode")
        self.tiff_checkbox = wx.CheckBox(main_panel, label="Save tracking as TIFF")

        checkbox_sizer.Add(self.debug_checkbox, 0, wx.RIGHT, 24)
        checkbox_sizer.Add(self.tiff_checkbox, 0)
        params_box.Add(checkbox_sizer, 0, wx.ALL, 8)

        # ML Model parameters
        ml_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ml_label = wx.StaticText(main_panel, label="Model path:", size=(100, -1))
        self.model_path_ctrl = wx.TextCtrl(main_panel)
        self.model_path_ctrl.SetValue("ML_stuff/best.pt")
        ml_sizer.Add(ml_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        ml_sizer.Add(self.model_path_ctrl, 1, wx.EXPAND)
        params_box.Add(ml_sizer, 0, wx.EXPAND | wx.ALL, 8)

        # Confidence threshold
        conf_sizer = wx.BoxSizer(wx.HORIZONTAL)
        conf_label = wx.StaticText(main_panel, label="Confidence:", size=(100, -1))
        self.conf_ctrl = wx.SpinCtrl(main_panel, value="25", min=1, max=100)
        conf_unit = wx.StaticText(main_panel, label="%")
        conf_sizer.Add(conf_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        conf_sizer.Add(self.conf_ctrl, 0, wx.RIGHT, 4)
        conf_sizer.Add(conf_unit, 0, wx.ALIGN_CENTER_VERTICAL)
        params_box.Add(conf_sizer, 0, wx.ALL, 8)

        main_sizer.Add(params_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        # ────────────────────────────────────────────────────────────────
        # PROCESSING SECTION
        # ────────────────────────────────────────────────────────────────
        processing_box = wx.StaticBoxSizer(wx.VERTICAL, main_panel, "Processing")

        # Progress bar
        self.progress_bar = wx.Gauge(main_panel, range=100, style=wx.GA_HORIZONTAL)
        self.progress_text = wx.StaticText(main_panel, label="Ready")

        processing_box.Add(self.progress_bar, 0, wx.EXPAND | wx.ALL, 8)
        processing_box.Add(self.progress_text, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Process button
        self.process_btn = wx.Button(main_panel, label="▶ Start Processing", size=(150, 35))
        process_font = self.process_btn.GetFont()
        process_font.PointSize = 11
        self.process_btn.SetFont(process_font)
        self.process_btn.Bind(wx.EVT_BUTTON, self._on_process_click)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(self.process_btn, 0, wx.ALIGN_CENTER)
        processing_box.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 8)

        main_sizer.Add(processing_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        # ────────────────────────────────────────────────────────────────
        # LOG / RESULTS SECTION
        # ────────────────────────────────────────────────────────────────
        log_box = wx.StaticBoxSizer(wx.VERTICAL, main_panel, "Output Log")

        self.log_ctrl = wx.TextCtrl(
            main_panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_WORDWRAP,
            size=(-1, 200)
        )
        log_font = wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.log_ctrl.SetFont(log_font)

        log_box.Add(self.log_ctrl, 1, wx.EXPAND | wx.ALL, 8)

        # Clear log button
        clear_btn = wx.Button(main_panel, label="Clear log")
        clear_btn.Bind(wx.EVT_BUTTON, lambda e: self.log_ctrl.Clear())
        log_box.Add(clear_btn, 0, wx.RIGHT | wx.BOTTOM, 8)

        main_sizer.Add(log_box, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        # ────────────────────────────────────────────────────────────────
        # FOOTER / STATUS
        # ────────────────────────────────────────────────────────────────
        main_sizer.Add(wx.StaticLine(main_panel), 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 8)

        status_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.status_text = wx.StaticText(main_panel, label="Ready")
        status_sizer.Add(self.status_text, 1, wx.LEFT, 8)
        main_sizer.Add(status_sizer, 0, wx.EXPAND | wx.BOTTOM, 8)

        main_panel.SetSizer(main_sizer)

    def _set_defaults(self):
        """Set default values for controls."""
        cwd = os.getcwd()
        default_video = os.path.join(cwd, "uploads", "output_001.mp4")
        default_output = os.path.join(cwd, "output")

        if os.path.exists(default_video):
            self.video_path_ctrl.SetValue(default_video)
        if os.path.exists(default_output):
            self.output_path_ctrl.SetValue(default_output)

        self._log("Application started. Ready to process.")

    def _bind_events(self):
        """Bind custom events for inter-thread communication."""
        self.Bind(wx.EVT_WINDOW_DESTROY, self._on_window_close)
        self.Bind(wx.EVT_SIZE, self._on_window_resize)

    # ────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ────────────────────────────────────────────────────────────────

    def _on_browse_video(self, event):
        """Open file dialog to select video."""
        wildcard = "Video files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|All files (*.*)|*.*"
        dlg = wx.FileDialog(
            self,
            "Select video file",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        )

        if dlg.ShowModal() == wx.ID_OK:
            self.video_path_ctrl.SetValue(dlg.GetPath())
        dlg.Destroy()

    def _on_browse_output(self, event):
        """Open directory dialog to select output folder."""
        dlg = wx.DirDialog(
            self,
            "Select output directory",
            style=wx.DD_DEFAULT_STYLE
        )

        if dlg.ShowModal() == wx.ID_OK:
            self.output_path_ctrl.SetValue(dlg.GetPath())
        dlg.Destroy()

    def _on_idx_changed(self, event):
        """Update status when index selection changes."""
        value = self.idx_slider.GetValue()
        self.frame_index = value
        self.status_text.SetLabel(f"Smoothing radius: {self.frame_index} pixels")

    def _on_window_resize(self, event):
        """Handle window resize to update image display."""
        # if self.original_bitmap is not None:
        self._update_image_display()
        # event.Skip()

    def _update_image_display(self):
        """Scale and display the image to fit the current panel size."""
        # if self.original_bitmap is None or self.image_panel is None:
        #     return

        panel_width, panel_height = self.image_panel.GetSize()
        if panel_width <= 0 or panel_height <= 0:
            return

        vid_path = self.video_path_ctrl.GetValue()
        print(vid_path)
        if vid_path == "":
            return

        # chosen frame is equal to opening up the video and then using the selected frame number
        # import cv2
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_height, img_width = frame_rgb.shape[:2]
        else:
            self._log("Could not read selected frame.")
            return

        img = wx.Bitmap.FromBuffer(img_width, img_height, frame_rgb)
        img_width = img.GetWidth()
        img_height = img.GetHeight()

        # Calculate scaling factor to fit image within panel while maintaining aspect ratio
        scale_w = panel_width / img_width
        scale_h = panel_height / img_height
        scale = min(scale_w, scale_h)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        if new_width > 0 and new_height > 0:
            scale_img = img.Rescale(img, (new_width, new_height))
            self.image_panel.SetBitmap(scale_img)

    def _on_process_click(self, event):
        """Handle process button click."""
        if self.processing:
            wx.MessageBox("Processing already in progress.", "Info", wx.OK | wx.ICON_INFORMATION)
            return

        # Validate inputs
        video_path = self.video_path_ctrl.GetValue()
        output_path = self.output_path_ctrl.GetValue()

        if not video_path:
            wx.MessageBox("Please select a video file.", "Error", wx.OK | wx.ICON_ERROR)
            return

        if not os.path.exists(video_path):
            wx.MessageBox(f"Video file not found:\n{video_path}", "Error", wx.OK | wx.ICON_ERROR)
            return

        if not output_path:
            wx.MessageBox("Please select an output directory.", "Error", wx.OK | wx.ICON_ERROR)
            return

        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path, exist_ok=True)
            except Exception as e:
                wx.MessageBox(f"Could not create output directory:\n{e}", "Error", wx.OK | wx.ICON_ERROR)
                return

        # Check if we have pipeline modules
        if not IMPORTS_AVAILABLE:
            wx.MessageBox(
                f"Pipeline modules not available:\n{IMPORT_ERROR}\n\nCannot start processing.",
                "Error",
                wx.OK | wx.ICON_ERROR
            )
            return

        # Disable controls during processing
        self.processing = True
        self.process_btn.Enable(False)
        self.progress_bar.SetValue(0)
        self._log(f"\n{'=' * 60}")
        self._log(f"Starting processing…")
        self._log(f"Video: {video_path}")
        self._log(f"Output: {output_path}")
        self._log(f"Smoothing radius: {self.idx_slider.GetValue()}")
        self._log(f"{'=' * 60}\n")

        # Start processing in background thread
        self.worker_thread = threading.Thread(
            target=self._process_worker,
            args=(video_path, output_path),
            daemon=True
        )
        self.worker_thread.start()

    def _process_worker(self, video_path: str, output_path: str):
        """Background worker thread for processing."""
        try:
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = os.path.join(output_path, f"results_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            start_time = time.perf_counter()

            # Get parameters
            smooth_radius = self.idx_slider.GetValue()
            model_path = self.model_path_ctrl.GetValue()
            conf = self.conf_ctrl.GetValue() / 100.0

            # Define output paths
            overlay_path = os.path.join(output_dir, "sclera_overlay.mp4")
            mask_path = os.path.join(output_dir, "sclera_mask.mp4")
            stabilized_path = os.path.join(output_dir, "stabilized.mp4")

            self._log("► Step 1: Running ML sclera isolation…")
            self._update_progress(20, "ML isolation in progress…")

            # Call ML pipeline
            process_video_ml(
                video_path=video_path,
                model_path=model_path,
                output_mask_path=mask_path,
                output_overlay_path=overlay_path,
                conf=conf,
                imgsz=512,
            )

            self._log("  ✓ Sclera isolation complete")
            self._log("  • Overlay video saved")
            self._log("  • Mask video saved")

            self._log("► Step 2: Stabilizing video…")
            self._update_progress(60, "Stabilizing video…")

            # Call stabilization pipeline
            stabilize_video(overlay_path, stabilized_path, smoothing_radius=smooth_radius)

            self._log("  ✓ Video stabilization complete")
            self._log("  • Stabilized video saved")

            elapsed = time.perf_counter() - start_time

            # Display results
            self._update_progress(100, "Processing complete!")
            self._log("\n" + "=" * 60)
            self._log("✓ All processing complete")
            self._log(f"Total time: {elapsed:.2f} seconds")
            self._log(f"Output directory: {output_dir}")
            self._log("=" * 60)

            self._log("\nGenerated files:")
            self._log(f"  • sclera_overlay.mp4")
            self._log(f"  • sclera_mask.mp4")
            self._log(f"  • stabilized.mp4")

        except Exception as e:
            self._log(f"\n✗ Error during processing: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            self._update_progress(0, "Error occurred during processing")

        finally:
            # Re-enable controls
            wx.CallAfter(self._processing_complete)

    def _update_progress(self, value: int, message: str):
        """Update progress bar and message (thread-safe)."""
        wx.CallAfter(lambda: self._do_update_progress(value, message))

    def _do_update_progress(self, value: int, message: str):
        """Actually update progress bar (must be called from main thread)."""
        self.progress_bar.SetValue(min(value, 100))
        self.progress_text.SetLabel(message)
        self.status_text.SetLabel(message)

    def _processing_complete(self):
        """Called when processing finishes."""
        self.processing = False
        self.process_btn.Enable(True)

    def _log(self, message: str = ""):
        """Add message to log (thread-safe)."""
        wx.CallAfter(lambda: self._do_log(message))

    def _do_log(self, message: str):
        """Actually add to log (must be called from main thread)."""
        if message:
            self.log_ctrl.AppendText(message + "\n")
        else:
            self.log_ctrl.AppendText("\n")

    def _on_window_close(self, event):
        """Handle window close."""
        if self.processing:
            dlg = wx.MessageDialog(
                self,
                "Processing is still running. Exit anyway?",
                "Confirm exit",
                wx.YES_NO | wx.ICON_QUESTION
            )
            if dlg.ShowModal() != wx.ID_YES:
                event.Veto()
                dlg.Destroy()
                return
            dlg.Destroy()

        event.Skip()


class VideoProcessorApp(wx.App):
    """Main application class."""

    def OnInit(self):
        self.frame = VideoProcessorFrame()
        self.frame.Show()
        return True

def GUI():
    app = VideoProcessorApp()
    app.MainLoop()

if __name__ == "__main__":
    GUI()