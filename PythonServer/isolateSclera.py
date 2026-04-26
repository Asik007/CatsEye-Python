# %%
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

video_source = r"uploads\IMG_1735.MOV"  # Change this to your video file path or camera index

def draw_contour(frame, largest_contour):
    contour_img = frame.copy()
    cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 2)
    return contour_img


def process_frame(frame):
    frame_low_res = cv2.resize(frame, (0, 0), fx=0.05, fy=0.05)
    frame_gaussian_low_res = cv2.GaussianBlur(frame_low_res, (3, 3), 3)
    frame_gaussian_low_res = cv2.resize(
        frame_gaussian_low_res,
        (frame.shape[1], frame.shape[0]),
    )

    frame_hsv = cv2.cvtColor(frame_gaussian_low_res, cv2.COLOR_BGR2HSV)
    thresh_hsv = cv2.inRange(frame_hsv, (0, 1, 180), (180, 35, 255))
    close_hsv = cv2.morphologyEx(
        thresh_hsv,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )

    contours, _ = cv2.findContours(close_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    return draw_contour(frame, largest_contour)


cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {video_source}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Could not read frames from the video source.")

print(
    f"Resolution: {first_frame.shape}, FPS: {fps}, "
    f"Total Frames: {total_frames}, duration: {total_frames / fps if fps else 0} seconds"
)

# Save annotated output video to the output folder.
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_video_path = output_dir / f"sclera_contours_{timestamp}.mp4"

height, width = first_frame.shape[:2]
effective_fps = fps if fps and fps > 0 else 30
writer = cv2.VideoWriter(
    str(output_video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    effective_fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {output_video_path}")

processed_frames = 0
first_processed = process_frame(first_frame)
writer.write(first_processed if first_processed is not None else first_frame)
if first_processed is not None:
    processed_frames += 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = process_frame(frame)
    writer.write(processed if processed is not None else frame)
    if processed is not None:
        processed_frames += 1

cap.release()
writer.release()

print(f"Processed {processed_frames} frames with detected contours.")
print(f"Saved annotated video to: {output_video_path}")

