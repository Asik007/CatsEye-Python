import cv2
import numpy as np


def select_roi_and_show(first_img: np.ndarray):
    """Show the first frame and let the user select an ROI rectangle."""
    window_name = "Select ROI (press Enter/Space to confirm)"
    roi = cv2.selectROI(window_name, first_img, False, False)
    cv2.destroyWindow(window_name)

    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise ValueError("No ROI selected.")

    # first_gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
    template = first_img[y : y + h, x : x + w]
    center = (x + w // 2, y + h // 2)
    return (x, y, w, h), template, center


def process_video(video_path):
    print(f"[process_video] Opening video: {video_path}")
    orig_vid = cv2.VideoCapture(video_path)
    if not orig_vid.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    # Take the first frame for ROI selection.
    ret, first_frame = orig_vid.read()
    if not ret:
        orig_vid.release()
        raise Exception("Could not read frame from video")

    # first_img = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    (roi_x, roi_y, roi_w, roi_h), template, origin_center = select_roi_and_show(first_frame)
    print(f"[process_video] ROI selected: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

    # Restart from frame 0 so displacement starts at the source frame.
    orig_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracked_points = []
    frame_index = 0
    total_frames = int(orig_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[process_video] Tracking started. Total frames: {total_frames}")

    while frame_index < orig_vid.get(cv2.CAP_PROP_FRAME_COUNT):  # Limit to first 100 frames for testing
        ret, frame = orig_vid.read()

        if not ret:
            break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        match = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(match)

        top_left = max_loc
        center = (top_left[0] + roi_w // 2, top_left[1] + roi_h // 2)
        displacement = (center[0] - origin_center[0], center[1] - origin_center[1])

        tracked_points.append(
            {
                "frame": frame_index,
                "center": center,
                "displacement": displacement,
                "match_score": float(max_val),
            }
        )
        frame_index += 1
        print(f"[process_video] Processed {frame_index}/{total_frames} frames...")

    orig_vid.release()
    cv2.destroyAllWindows()
    print(f"[process_video] Tracking complete. Points tracked: {len(tracked_points)}")
    
    return {
        "video_path": video_path,
        "roi": (roi_x, roi_y, roi_w, roi_h),
        "origin_center": origin_center,
        "tracked_points": tracked_points,
    }


def save_tracking_results(results, output_path):
    # save it as a csv file with columns: frame, center_x, center_y, disp_x, disp_y, match_score, and then calculate the mean location and standard deviation of the tracked points and save it as a json file
    import csv

    print(f"[save_tracking_results] Writing CSV: {output_path}")

    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'center_x', 'center_y', 'disp_x', 'disp_y', 'match_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for point in results['tracked_points']:
            writer.writerow({
                'frame': point['frame'],
                'center_x': point['center'][0],
                'center_y': point['center'][1],
                'disp_x': point['displacement'][0],
                'disp_y': point['displacement'][1],
                'match_score': point['match_score'],
            })
        
        # Calculate mean location and standard deviation
        centers = np.array([point['center'] for point in results['tracked_points']])
        mean_location = centers.mean(axis=0)
        std_location = centers.std(axis=0)
        # save mean and std in the same csv file as a new row with frame = 'mean' and frame = 'std'
        writer.writerow({
            'frame': 'mean',
            'center_x': mean_location[0],
            'center_y': mean_location[1],
            'disp_x': '',
            'disp_y': '',
            'match_score': '',
        })
        writer.writerow({
            'frame': 'std',
            'center_x': std_location[0],
            'center_y': std_location[1],
            'disp_x': '',
            'disp_y': '',
            'match_score': '',
        })

    print(f"[save_tracking_results] CSV saved. Rows: {len(results['tracked_points']) + 3}")


def process_and_save(video_path, output_csv):
    import os

    print(f"[process_and_save] Start processing video: {video_path}")

    results = process_video(video_path)
    # save_tracking_results(results, output_csv)

    # Create a video that overlays each tracked center point.
    orig_vid = cv2.VideoCapture(video_path)
    if not orig_vid.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    fps = orig_vid.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_size = (
        int(orig_vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(orig_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    output_dir = os.path.dirname(output_csv) or "."
    output_video = os.path.join(output_dir, "output_video.mp4")
    print(f"[process_and_save] Creating visualization video: {output_video}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    if not out_vid.isOpened():
        orig_vid.release()
        raise Exception(f"Could not open output video for writing: {output_video}")

    frame_index = 0
    total_frames = int(orig_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    trail_points = []
    while frame_index < orig_vid.get(cv2.CAP_PROP_FRAME_COUNT):
        ret, frame = orig_vid.read()
        if not ret:
            break

        if frame_index < len(results["tracked_points"]):
            center = tuple(results["tracked_points"][frame_index]["center"])
            trail_points.append(center)

            if len(trail_points) > 1:
                pts = np.array(trail_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 0), 2)

            cv2.circle(frame, center, 10, (0, 255, 0), -1)

        out_vid.write(frame)
        frame_index += 1

        if frame_index % 50 == 0:
            print(f"[process_and_save] Wrote {frame_index}/{total_frames} frames to output video...")

    orig_vid.release()
    out_vid.release()
    cv2.destroyAllWindows()
    print(f"[process_and_save] Video render complete: {output_video}")

    # play the output video
    print("[process_and_save] Starting output video playback window. Press 'q' to close.")
    cap = cv2.VideoCapture(output_video)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {output_video}")
    while cap.isOpened():
        # 3. Read frame-by-frame
        ret, frame = cap.read()

    # If ret is False, the video has ended or there's an error
        if not ret:
            break

    # 4. Display the resulting frame
        cv2.imshow('Video Playback', frame)

    # 5. Wait for a key press and set the frame rate
    # waitKey(25) provides a 25ms delay (approx 40 FPS)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[process_and_save] Playback closed. Processing done.")
    
    return {
        "csv_path": output_csv,
        "video_path": output_video,
        "frames_visualized": frame_index,
        "tracking_points": len(results["tracked_points"]),
    }

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ThreadPoolExecutor
import threading

"""
Test Code for stabilization

from Pipeline import stabilize_video
import os
import datetime
input_video = os.getcwd() + "/uploads/IMG_1702.mov"
output_video = os.getcwd() + f"/output/stabilized_output_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
stabilize_video(input_video, output_video, smooth_radius=50)



"""

def stabilize_video(input_path: str, output_path: str, smooth_radius: int = 50):
    cap = cv2.VideoCapture(input_path)

    # --- Metadata (all known upfront) ---
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Pass 1: Buffer all frames + compute transforms in one loop ---
    # Pre-allocate — avoids incremental realloc from list.append()
    frames     = np.empty((n_frames, h, w, 3), dtype=np.uint8)
    transforms = np.empty((n_frames - 1, 3),   dtype=np.float64)  # dx, dy, angle

    ret, frames[0] = cap.read()
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, n_frames):
        ret, frames[i] = cap.read()
        if not ret:
            n_frames = i  # handle truncated files
            break

        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        transforms[i - 1] = _estimate_transform(prev_gray, curr_gray)
        prev_gray = curr_gray  # no copy — just reassign reference

    cap.release()

    # --- Trajectory smoothing (fully vectorized, no Python loop) ---
    trajectory = np.cumsum(transforms, axis=0)           # shape (N-1, 3)
    # uniform_filter1d is a fast O(N) box filter — no padding gymnastics needed
    smoothed   = uniform_filter1d(trajectory, size=smooth_radius * 2 + 1, axis=0, mode='nearest')
    corrections = smoothed - trajectory                  # shape (N-1, 3)

    # --- Pass 2: Warp + write, pipelined ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Build all warp matrices at once (vectorized)
    cos_a = np.cos(corrections[:, 2])
    sin_a = np.sin(corrections[:, 2])
    dx    = corrections[:, 0]
    dy    = corrections[:, 1]

    # Shape: (N-1, 2, 3)
    Ms = np.stack([
        np.stack([ cos_a, -sin_a, dx], axis=1),
        np.stack([ sin_a,  cos_a, dy], axis=1),
    ], axis=1)

    # Write frame 0 unchanged, then apply corrections from frame 1 onward
    writer.write(frames[0])

    # Use a queue + background thread to overlap warp (CPU) with disk write (I/O)
    write_queue  = []
    write_lock   = threading.Semaphore(0)

    def writer_thread():
        for _ in range(n_frames - 1):
            write_lock.acquire()
            writer.write(write_queue.pop(0))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(writer_thread)
        for i in range(n_frames - 1):
            stabilized = cv2.warpAffine(
                frames[i + 1], Ms[i], (w, h),
                borderMode=cv2.BORDER_REFLECT
            )
            write_queue.append(stabilized)
            write_lock.release()
        future.result()

    writer.release()


def _estimate_transform(prev_gray, curr_gray):
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,qualityLevel=0.05, minDistance=30)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    mask     = status.ravel() == 1
    M, _     = cv2.estimateAffinePartial2D(prev_pts[mask], curr_pts[mask])

    # return M[0, 2], M[1, 2], np.arctan2(M[1, 0], M[0, 0]) # with rotation
    return M[0, 2], M[1, 2], 0.0 # without rotation


# Here is the code to run the code with CLI
import os
import sys


def _run_cli() -> None:
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    def _clean_path(p: str) -> str:
        p = os.path.expanduser(p.strip().strip('"').strip("'"))
        if not os.path.isabs(p):
            p = os.path.join(cwd, p)
        return os.path.abspath(p)

    def _ask_yes_no(prompt: str, default: bool = True) -> bool:
        suffix = " [Y/n]: " if default else " [y/N]: "
        raw = input(prompt + suffix).strip().lower()
        if not raw:
            return default
        return raw in {"y", "yes"}

    # Ask for source video (relative paths are resolved from current working directory)
    while True:
        src_input = input("Enter source video path (relative to current directory): ").strip()
        video_path = _clean_path(src_input)
        if os.path.isfile(video_path):
            break
        print("Invalid file path. Please try again.")

    # Ask for destination folder (relative paths are resolved from current working directory)
    default_dest = os.path.join(cwd, "output")
    dest_input = input(f"Enter destination folder [{default_dest}]: ").strip()
    dest_folder = _clean_path(dest_input) if dest_input else default_dest
    os.makedirs(dest_folder, exist_ok=True)

    # Ask whether to show playback window
    show_output_video = _ask_yes_no("Show output video after processing?", default=True)

    # process_and_save uses dirname(output_csv) for output location
    output_csv = os.path.join(dest_folder, "tracking_results.csv")

    # If user does not want playback, bypass window calls used in process_and_save
    if not show_output_video:
        def _noop_imshow(*_args, **_kwargs):
            return None

        def _quit_waitkey(_delay=0):
            return ord("q")

        cv2.imshow = _noop_imshow
        cv2.waitKey = _quit_waitkey

    result = process_and_save(video_path, output_csv)

    print("\nDone.")
    print(f"CSV path:   {result['csv_path']}")
    print(f"Video path: {result['video_path']}")


if __name__ == "__main__":
    if sys.stdin.isatty() and sys.stdout.isatty():
        _run_cli()
    else:
        print("Run this script from a terminal to use the interactive prompts.")
