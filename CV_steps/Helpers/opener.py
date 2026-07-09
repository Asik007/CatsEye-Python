import cv2
import numpy as np


def extract_vid_seg(
        video_path,
        start_frame,
        end_frame,
        output_dir
    ):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open source video.")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Boundary check
    if end_frame >= total_frames:
        end_frame = total_frames - 1

    # Jump to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_list = []
    current_frame = start_frame

    # 1. Extract frames into a list
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(frame)
        current_frame += 1

    cap.release()

    if not frames_list:
        print("No frames were extracted.")
        return None

    # 2. Convert list to a NumPy stack
    # Shape will be: (num_frames, height, width, channels)
    frame_stack = np.stack(frames_list, axis=0)
    print(f"NumPy stack created with shape: {frame_stack.shape}")

    # 3. Save the NumPy stack as a video file
    # 'mp4v' is a widely compatible codec for .mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_filename = output_dir + "/trimmed.mp4"
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Iterate through axis 0 of the numpy array to write frames
    for i in range(frame_stack.shape[0]):
        out.write(frame_stack[i])

    out.release()
    print(f"Successfully saved video to {output_filename}")

    # Returns the numpy array in case you need it for further processing
    return frame_stack


# --- Example Usage ---
# Extracts frames 100 to 150, saves them as 'trimmed.mp4', and returns the array
