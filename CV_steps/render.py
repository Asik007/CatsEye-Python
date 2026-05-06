import cv2
import numpy as np


def _longest_continuous_segment(
    tracked_points: list[dict], best_frame_idx: int | None
) -> list[dict]:
    """Return the longest run of tracked frames with consecutive frame numbers.

    If best_frame_idx is provided, return the segment containing that frame
    instead of the overall longest segment.
    """
    segments = _build_consecutive_segments(tracked_points)

    if best_frame_idx is not None:
        for segment in segments:
            if any(p.get("frame") == best_frame_idx for p in segment):
                return segment

    return max(segments, key=len, default=[])


def _build_consecutive_segments(tracked_points: list[dict]) -> list[list[dict]]:
    """Split tracked points into groups of consecutive frame numbers."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    prev_frame: int | None = None

    for point in tracked_points:
        frame_idx = point.get("frame") if point else None

        if frame_idx is None:
            if current:
                segments.append(current)
                current = []
            prev_frame = None
            continue

        if prev_frame is not None and frame_idx != prev_frame + 1:
            segments.append(current)
            current = []

        current.append(point)
        prev_frame = frame_idx

    if current:
        segments.append(current)

    return segments

def _render_tracking_video(
    video_path: str,
    tracked_points: list[dict],
    output_path: str,
    roi: tuple,
) -> None:
    """
    Re-reads the original video and draws on every frame:
      • Green polyline trail of the tracked center
      • Colored ROI box at the current matched position
      • Current-center dot
      • Displacement (dx / dy) text in the top-left corner
    """
    _x, _y, roi_w, roi_h = roi
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trail: list[tuple[int, int]] = []

    for idx in range(n):
        ret, frame = cap.read()
        if not ret:
            break

        if idx < len(tracked_points):
            pt     = tracked_points[idx]
            center = tuple(pt["center"])
            dx, dy = pt["displacement"]
            trail.append(center)

            # --- draw trail ---
            if len(trail) > 1:
                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

            # --- draw moving ROI box ---
            tl = (center[0] - roi_w // 2, center[1] - roi_h // 2)
            br = (center[0] + roi_w // 2, center[1] + roi_h // 2)
            cv2.rectangle(frame, tl, br, color=(0, 200, 255), thickness=2)

            # --- center dot ---
            cv2.circle(frame, center, radius=6, color=(0, 255, 0), thickness=-1)

            # --- displacement label ---
            # label = f"dx={dx:+d}  dy={dy:+d}"
            # cv2.putText(
            #     frame, label, (10, 34),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
            # )

        writer.write(frame)

        if (idx + 1) % 50 == 0:
            print(f"  [render]   {idx + 1}/{n} frames written")

    cap.release()
    writer.release()


def _render_stabilized_video(
    video_path: str,
    tracked_points: list[dict],
    output_path: str,
    best_frame_idx: int | None = None,
) -> None:

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    segment = _longest_continuous_segment(tracked_points, best_frame_idx)
    if not segment:
        cap.release()
        writer.release()
        raise ValueError("No continuous tracked frame segment was found to render.")

    start_frame = segment[0]["frame"]
    end_frame = segment[-1]["frame"]
    print(f"  [render] using continuous frames {start_frame}..{end_frame} ({len(segment)} frames)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for segment_idx, point in enumerate(segment):
        # frame_idx = point["frame"]
        ret, frame = cap.read()
        if not ret:
            break
        # print(f"Frame {frame_idx}")
        M = point["transform"]
        # print(f"matrix: {segment_idx} / {len(segment)}")
        # print(f"M shape: {M.shape}")

        
        #  shift the frame in the opposite direction of the displacement to stabilize
        # M = np.float32([[1, 0, -del_x], [0, 1, -del_y]])
        # frame = cv2.warpAffine(frame, M, (w, h))
        # test a homography-based stabilization (not currently better than simple translation)
        # src_pts = np.float32([tracked_points[idx]["center"]])
        # dst_pts = np.float32([tracked_points[0]["center"]])
        # H, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
        # if its a homography matrix, use warpPerspective; if it's affine, use warpAffine
        if M.shape == (2, 3):
            print(f"  [render] applying affine transform for frame {point['frame']}")

            frame = cv2.warpAffine(frame, M, (w, h))
        elif M.shape == (3, 3):
            print(f"  [render] applying homography transform for frame {point['frame']}")
            frame = cv2.warpPerspective(frame, M, (w, h))
        writer.write(frame)

        if (segment_idx + 1) % 50 == 0:
            print(f"  [render]   {segment_idx + 1}/{len(segment)} frames written")

    cap.release()
    writer.release()


def render_tracking_video(
    video_path: str,
    tracked_points: list[dict],
    output_path: str,
    roi: tuple,
) -> None:
    _render_tracking_video(video_path, tracked_points, output_path, roi)

def render_stabilized_video(
    video_path: str,
    tracked_points: list[dict],
    output_path: str,
    best_frame_idx: int | None = None,
) -> None:
    _render_stabilized_video(video_path, tracked_points, output_path)

def select_wanted_frame(
    input_video: str,
) -> tuple[int, np.ndarray]:
    '''
    Display the video and let the user go back and forth and select the
    desired frame they want, then return the frame_number and the frame itself.

    Controls:
        D  → next frame
        A  → previous frame
        \ or Enter  → confirm selection
        Q or Esc      → quit without selecting
    '''
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get frame dimensions if needed:
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    new_height = min(600, height)
    new_width = int(aspect_ratio * new_height)

    idx        = 0
    best_frame = None

    def read_frame(index: int) -> np.ndarray:
        print(f"Reading frame {index} / {total_frames}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        return frame if ret else None

    window = "Select Frame  |  A/D step  |  \\ or Enter confirm  |  Q or Esc quit"
    print(window)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # idx = 0

    while True:
        frame = read_frame(idx)

        if frame is None:
            break
        key = None
        resize_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow(window, resize_frame)

        # ── key handling ─────────────────────────────────────────────────────
        # key = cv2.waitKey(1)          # flush any queued eventsb
        key = cv2.waitKey(0) & 0xFF

        print(f"Key pressed: {key} (frame index: {idx})")

        if key == ord('\\') or key == 13:          # \ or Enter → confirm
            print(f"Selected frame {idx}")
            best_frame = frame
            break

        elif key == ord('q') or key == 27:         # ESC / Q → abort
            print("Selection cancelled by user.")
            idx, best_frame = 50, read_frame(50)  # default to frame 50 if user cancels
            break

        elif key == ord('a'):         # LEFT arrow
            print(f"Left key pressed. Current index: {idx}")
            idx = max(0, idx - 1)

        elif key == ord('d'):         # RIGHT arrow
            print(f"Right key pressed. Current index: {idx}")
            idx = min(total_frames - 1, idx + 1)

    cap.release()
    cv2.destroyAllWindows()
    return idx, best_frame


def show(img_rgb, scale_factor=0.25, title=None):
    from PIL import Image
    from IPython.display import display

    # if greyscale, convert to RGB
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    scaled_img = cv2.resize(img_rgb, (img_rgb.shape[1] // int(1/scale_factor), img_rgb.shape[0] // int(1/scale_factor)))
    display(Image.fromarray(scaled_img))
    return scaled_img