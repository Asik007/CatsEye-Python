import cv2
import matplotlib.pyplot as plt
import numpy as np
import os




def find_seg(
        num_frames: int,
        bad_frames: list[int],
        # video_path: str,
        best_frame_idx: int | None = None,
        max_len: int = 50,
    ) -> tuple[int, int]:
    # we want to use bad frames as walls to divide the frames into segments
    boundaries = [-1] + bad_frames + [num_frames]
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1] - 1
        if segment_eval(start, end, max_len):
            segments.append((start, end))

    if not segments:
        raise ValueError("No valid video segments found after filtering out bad frames.")

    # 2. Select the target segment
    chosen_seg = None
    if best_frame_idx is not None:
        # Find the segment that contains our target best frame
        for start, end in segments:
            if start <= best_frame_idx <= end:
                chosen_seg = (start, end)
                break

    # Fallback: if no best_frame_idx is provided (or it landed on a bad frame), pick the longest segment
    if chosen_seg is None:
        chosen_seg = max(segments, key=lambda x: x[1] - x[0] + 1)

    return chosen_seg

def segment_eval(
        start: int,
        end: int,
        max_len: int,
    ) -> bool:
    isGood = (start <= end and end - start <= max_len)
    return isGood


def vid2seg(
        video_path: str,  # this is the video with blanks for bad frames
        bad_frames: list[int],
        frame_sizes: list[int],
        output_path: str,
        max_len: int = 50,
        best_frame_idx: int | None = None,
):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate mean and std dev of frame sizes, excluding already bad frames
    # valid_sizes = [frame_sizes[i] for i in range(len(frame_sizes)) if i not in bad_frames]
    valid_sizes = []
    for i, size in enumerate(frame_sizes):
        if i not in bad_frames:
            if size.size > 0:
                valid_sizes.append(size)
            else:
                bad_frames.append(i)
    #             this is tl_x, tl_y, br_x, br_y


    if valid_sizes:
        mean_size = np.mean(valid_sizes)
        std_size = np.std(valid_sizes)

        # Find outlier frames (outside 1 stdev from mean)
        lower_bound = mean_size - std_size
        upper_bound = mean_size + std_size

        for i, size in enumerate(frame_sizes):
            if i not in bad_frames and (size < lower_bound or size > upper_bound):
                bad_frames.append(i)

    # Remove duplicates and sort
    bad_frames = sorted(set(bad_frames))

    # Find the best segment using the helper function
    start_frame, end_frame = find_seg(
        num_frames=num_frames,
        bad_frames=bad_frames,
        best_frame_idx=best_frame_idx,
        max_len=max_len,
    )

    # Write the selected segment to output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Write frame only if it's within the selected segment
        if start_frame <= frame_idx <= end_frame:
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

    print(f"Segment saved: frames {start_frame}-{end_frame} to {output_path}")

    return (start_frame, end_frame)





