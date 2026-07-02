"""
Integrated Tkinter GUI for the video processing pipeline.
This version directly connects to process_and_stabilize() and other pipeline functions.

Usage:
  python video_processor_gui_tkinter.py
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from pathlib import Path
from datetime import datetime

# Import your pipeline functions
from main import process_and_stabilize


class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor -- CatsEye")
        self.root.geometry("900x750")

        self.frame_idx = 0
        self.vid_len = 0
        self.processing = False
        self.current_photo = None  # Keep reference to avoid garbage collection

        # ────────────────────────────────────────────────────────────────
        # UI CONSTRUCTION
        # ────────────────────────────────────────────────────────────────
        self.setup_ui()

    def setup_ui(self):
        """Create and layout all UI elements."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Header
        header_label = ttk.Label(
            main_frame,
            text="Video Processing Pipeline",
            font=("TkDefaultFont", 16, "bold")
        )
        header_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # ── Input Section ──
        input_frame = ttk.LabelFrame(main_frame, text="Input & Output", padding="10")
        input_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        input_frame.columnconfigure(1, weight=1)

        # Video File
        ttk.Label(input_frame, text="Video file:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        self.video_path_var = tk.StringVar()
        self.video_path_ctrl = ttk.Entry(input_frame, textvariable=self.video_path_var, state="readonly")
        self.video_path_ctrl.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        video_browse_btn = ttk.Button(input_frame, text="Browse…", command=self.on_browse_video)
        video_browse_btn.grid(row=0, column=2)

        # Output Folder
        ttk.Label(input_frame, text="Output folder:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)
        self.output_path_var = tk.StringVar()
        self.output_path_ctrl = ttk.Entry(input_frame, textvariable=self.output_path_var, state="readonly")
        self.output_path_ctrl.grid(row=1, column=1, sticky="ew", padx=(0, 5))
        output_browse_btn = ttk.Button(input_frame, text="Browse…", command=self.on_browse_output)
        output_browse_btn.grid(row=1, column=2)

        # ── Image Preview Section ──
        preview_frame = ttk.LabelFrame(main_frame, text="Frame Preview", padding="10")
        preview_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        preview_frame.columnconfigure(0, weight=1)

        self.image_panel = ttk.Label(preview_frame, background="#e0e0e0", relief="sunken")
        self.image_panel.grid(row=0, column=0, sticky="ew", pady=10)
        # Set a minimum height for the image panel
        # self.image_panel.config(height=250)

        # ── Parameters Section ──
        params_frame = ttk.LabelFrame(main_frame, text="Processing Parameters", padding="10")
        params_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        params_frame.columnconfigure(1, weight=1)

        # Frame Index Slider
        ttk.Label(params_frame, text="Best Frame Index:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        slider_frame = ttk.Frame(params_frame)
        slider_frame.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        slider_frame.columnconfigure(0, weight=1)

        self.idx_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=100,
            orient="horizontal",
            command=self.on_idx_change
        )
        self.idx_slider.grid(row=0, column=0, sticky="ew")

        self.idx_unit = ttk.Label(slider_frame, text="Frame 0", width=10)
        self.idx_unit.grid(row=0, column=1, sticky="w", padx=(10, 0))

        # Checkbox
        check_frame = ttk.Frame(params_frame)
        check_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        self.tiff_var = tk.BooleanVar()
        self.tiff_checkbox = ttk.Checkbutton(check_frame, text="Save tracking as TIFF", variable=self.tiff_var)
        self.tiff_checkbox.pack(side="left")

        # ML Model Path
        ttk.Label(params_frame, text="Model path:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=5)
        self.model_path_var = tk.StringVar(value="ML_stuff/best.pt")
        self.model_path_ctrl = ttk.Entry(params_frame, textvariable=self.model_path_var)
        self.model_path_ctrl.grid(row=2, column=1, sticky="ew")

        # ── Processing Section ──
        processing_frame = ttk.LabelFrame(main_frame, text="Processing", padding="10")
        processing_frame.grid(row=4, column=0, sticky="ew", pady=(0, 15))
        processing_frame.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(
            processing_frame,
            mode="determinate",
            maximum=100,
            length=400
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.progress_text = ttk.Label(processing_frame, text="Ready")
        self.progress_text.grid(row=1, column=0, sticky="w", pady=(0, 10))

        button_frame = ttk.Frame(processing_frame)
        button_frame.grid(row=2, column=0, sticky="ew")
        button_frame.columnconfigure(0, weight=1)

        self.process_btn = ttk.Button(
            button_frame,
            text="▶ Start Processing",
            command=self.on_process_click,
            width=20
        )
        self.process_btn.pack(side="left", padx=5, pady=5)

        # ── Status Footer ──
        ttk.Separator(main_frame, orient="horizontal").grid(row=5, column=0, sticky="ew", pady=(0, 5))
        self.status_text = ttk.Label(main_frame, text="Ready")
        self.status_text.grid(row=6, column=0, sticky="w", pady=5)

    # ────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ────────────────────────────────────────────────────────────────

    def on_browse_video(self):
        """Open file dialog to select video."""
        file_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("MP4", "*.mp4"),
                ("AVI", "*.avi"),
                ("MOV", "*.mov"),
                ("MKV", "*.mkv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.video_path_var.set(file_path)
            try:
                cap = cv2.VideoCapture(file_path)
                self.vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.idx_slider.config(to=self.vid_len)
                cap.release()
                self.update_image_display()
                self.status_text.config(text=f"Loaded video: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load video:\n{e}")

    def on_browse_output(self):
        """Open directory dialog to select output folder."""
        folder_path = filedialog.askdirectory(title="Select output directory")
        if folder_path:
            self.output_path_var.set(folder_path)
            self.status_text.config(text=f"Output: {folder_path}")

    def on_idx_change(self, value):
        """Update status when index selection changes."""
        val = int(float(value))
        self.idx_unit.config(text=f"Frame {val}")
        self.frame_idx = val
        print(f"slider index: {val}")
        self.update_image_display()

    def update_image_display(self):
        """Extract a frame from the video and display it."""
        vid_path = self.video_path_var.get()
        if not vid_path or not os.path.exists(vid_path):
            return

        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                print("updating image display")
                # Resize frame to fit panel
                height, width = frame.shape[:2]
                max_width, max_height = 850, 250

                if width > max_width or height > max_height:
                    scale = min(max_width / width, max_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                self.current_photo = ImageTk.PhotoImage(pil_image)
                self.image_panel.config(image=self.current_photo, text="")
            else:
                self.image_panel.config(text="Could not read selected frame.")

        except Exception as e:
            self.image_panel.config(text=f"Error loading frame: {e}")

    def on_process_click(self):
        """Handle process button click. Validate inputs and launch processing thread."""
        if self.processing:
            messagebox.showinfo("Info", "Processing already in progress.")
            return

        # Validate inputs
        video_path = self.video_path_var.get()
        output_path = self.output_path_var.get()

        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        if not output_path:
            messagebox.showerror("Error", "Please select an output directory.")
            return

        try:
            os.makedirs(output_path, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not create output directory:\n{e}")
            return

        # Disable controls and start processing thread
        self.processing = True
        self.process_btn.config(state="disabled")
        self.progress_bar["value"] = 0
        self.progress_text.config(text="Processing...")

        print(f"\n{'=' * 60}")
        print("Starting processing…")
        print(f"Video: {video_path}")
        print(f"Output: {output_path}")
        print(f"Frame index: {int(self.idx_slider.get())}")
        print(f"Save as TIFF: {self.tiff_var.get()}")
        print(f"{'=' * 60}\n")

        # Run processing in separate thread to keep UI responsive
        thread = threading.Thread(
            target=self.run_processing,
            args=(video_path, output_path),
            daemon=True
        )
        thread.start()

    def run_processing(self, video_path, output_path):
        """Run the video processing pipeline in a background thread."""
        # try:
            # Call ML pipeline
        process_and_stabilize(
            video_path=video_path,
            output_dir=output_path,
        )

        # Update UI on completion
        self.root.after(self.on_processing_complete, "success")

        # except Exception as e:
        #     print(f"Processing error: {e}")
        #     self.root.after(self.on_processing_complete, "error", str(e))

    def on_processing_complete(self, status, error_msg=None):
        """Handle processing completion."""
        self.processing = False
        self.process_btn.config(state="normal")

        if status == "success":
            self.progress_bar["value"] = 100
            self.progress_text.config(text="Processing completed!")
            self.status_text.config(text="Ready")
            messagebox.showinfo("Success", "Video processing completed successfully!")
        else:
            self.progress_bar["value"] = 0
            self.progress_text.config(text="Processing failed")
            self.status_text.config(text="Ready")
            messagebox.showerror("Error", f"Processing failed:\n{error_msg}")


def main():
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()