import argparse
from CV_steps.Render.render import render_tiff
from CV_steps.Isolate.Vessel_IP import normalize_and_enhance
from CV_steps.Register.XCorr_phase import imreg_dft_emulate
from CV_steps.Helpers.Image_Input import select_roi_scaling
from CV_steps.Helpers.tracking_io import save_tracking_csv
from Vid_Obj import VidObj          # <-- using your PyAV‑based class
import cv2
import numpy as np
import os

try:
    from CV_steps.Render.render import render_tracking_video, render_stabilized_video, select_wanted_frame
except ImportError:
    from CV_steps.Render.render import render_tracking_video, render_stabilized_video, select_wanted_frame


def _get_inscribed_square(img: np.ndarray) -> tuple:
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    M = cv2.moments(mask)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    print(f"Center: ({cX}, {cY})")
    per_sel = 0.2
    side = int(min(img.shape[0] * per_sel, img.shape[1] * per_sel))
    template = img[int(cY - side // 2):int(cY + side // 2),
                   int(cX - side // 2):int(cX + side // 2)].copy()

    print(f"Inscribed square ROI: center=({cX}, {cY}), side={side}")
    roi_img = img.copy()
    print("if you like this auto-selected ROI, press 'y' to confirm. Otherwise, press 'n' to select manually.")
    cv2.rectangle(roi_img, (cX - side // 2, cY - side // 2),
                  (cX + side // 2, cY + side // 2), (0, 255, 0), 2)
    cv2.imshow("y = confirm, n = select manually", roi_img)
    key = cv2.waitKey(0)
    if key == ord('y'):
        print("auto ROI confirmed by user.")
    if key == ord('n'):
        print("auto ROI rejected by user. Please select ROI manually.")
        cv2.destroyAllWindows()
        return select_roi_scaling(img)

    return (cX, cY, side, side), template, (cX, cY)


def gen_mask(frame: np.ndarray) -> np.ndarray:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_binary = cv2.threshold(frame_gray, 1, 255, cv2.THRESH_BINARY)[1]
    shrink_dist = min(frame.shape[1] * 0.1, frame.shape[0] * 0.1)
    dist = cv2.distanceTransform(frame_binary, cv2.DIST_L2, 5)
    mask = (dist > shrink_dist).astype(np.uint8) * 255
    return mask


# ----------------------------------------------------------------------
#  Tracking functions now use VidObj instead of cv2.VideoCapture
# ----------------------------------------------------------------------

def track_with_cross_correlation(
    video_path: str,
    roi: tuple,
    template: np.ndarray,
    origin_center: tuple,
) -> list[dict]:
    _x, _y, roi_w, roi_h = roi
    vid = VidObj(video_path)                     # <-- VidObj
    n_frames = vid.len
    tracked = []

    for idx, frame in enumerate(vid.frames_gen()):  # <-- generator yields BGR frames
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        cx = max_loc[0] + roi_w // 2
        cy = max_loc[1] + roi_h // 2
        tracked.append({
            "frame": idx,
            "center": (cx, cy),
            "displacement": (cx - origin_center[0], cy - origin_center[1]),
            "match_score": float(score),
        })
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [tracking] {idx + 1}/{n_frames} frames")
    return tracked


def track_with_homography(video_path: str, best_frame: np.ndarray = None) -> list[dict]:
    vid = VidObj(video_path)                     # <-- VidObj
    n_frames = vid.len
    tracked = []

    orb = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    prev_frame = None
    prev_kp = None
    prev_des = None

    if best_frame is not None:
        print("Using provided best_frame for tracking.")
        mask = gen_mask(best_frame)
        prev_kp, prev_des = orb.detectAndCompute(best_frame, mask=mask)
        prev_frame = best_frame.copy()

    for idx, frame in enumerate(vid.frames_gen()):  # <-- generator
        mask = gen_mask(frame)
        kp2, des2 = orb.detectAndCompute(frame, mask=mask)

        if prev_frame is None:
            prev_frame, prev_kp, prev_des = frame.copy(), kp2, des2
            tracked.append(None)
            continue

        if prev_des is None or des2 is None:
            prev_frame, prev_kp, prev_des = frame.copy(), kp2, des2
            tracked.append(None)
            continue

        matches = sorted(bf.match(prev_des, des2), key=lambda m: m.distance)
        if len(matches) < 4:
            prev_frame, prev_kp, prev_des = frame.copy(), kp2, des2
            tracked.append(None)
            continue

        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, what = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if H is None:
            prev_frame, prev_kp, prev_des = frame.copy(), kp2, des2
            tracked.append(None)
            continue

        tx = H[0, 2]
        ty = H[1, 2]
        sx = np.linalg.norm(H[0:2, 0])
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        sy = det / sx
        rotation_deg = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
        shear = (H[0, 0] * H[0, 1] + H[1, 0] * H[1, 1]) / det

        tracked.append({
            "frame": idx,
            "transform": H,
            "trans X": tx,
            "trans Y": ty,
            "Scale X": sx,
            "Scale Y": sy,
            "rotation_deg": rotation_deg,
            "shear": shear,
        })

        prev_frame = frame.copy()
        prev_kp = kp2
        prev_des = des2

        if (idx + 1) % 1 == 0 or idx == 0:
            print(f"  [tracking] {idx + 1}/{n_frames} frames")

    return tracked


EDGE_MARGIN_RATIO = 0.05
FEATURE_PARAMS    = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
LK_PARAMS         = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
REDETECT_EVERY    = 30


def track_with_homography2(video_path: str, best_frame: np.ndarray = None) -> list[dict]:
    def build_mask(gray: np.ndarray) -> np.ndarray:
        binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        margin = min(gray.shape[:2]) * EDGE_MARGIN_RATIO
        dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        return (dist > margin).astype(np.uint8) * 255

    def detect(gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(gray, mask=build_mask(gray), **FEATURE_PARAMS)
        return pts.astype(np.float32) if pts is not None else np.empty((0, 1, 2), np.float32)

    vid = VidObj(video_path)                     # <-- VidObj
    n_frames = vid.len
    tracked = []

    if best_frame is not None:
        print("Seeding tracker with provided best_frame.")
        prev_gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
        prev_pts  = detect(prev_gray)
    else:
        prev_gray = prev_pts = None

    for idx, frame in enumerate(vid.frames_gen()):  # <-- generator
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray, prev_pts = gray, detect(gray)
            tracked.append(None)
            continue

        if len(prev_pts) < 10 or idx % REDETECT_EVERY == 0:
            prev_pts = detect(prev_gray)

        if len(prev_pts) < 4:
            prev_gray, prev_pts = gray, detect(gray)
            tracked.append(None)
            continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)
        # (visualisation code left unchanged)
        im_match = cv2.drawMatches(prev_gray, [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=1) for p in prev_pts],
                                   gray, [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=1) for p in curr_pts],
                                   [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(prev_pts)) if status[i] == 1],
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        resize_match = cv2.resize(im_match, (800, 600))
        cv2.imshow("tracking", resize_match)
        cv2.waitKey(1)

        good_old = prev_pts[status == 1]
        good_new = curr_pts[status == 1]

        if len(good_old) < 4:
            prev_gray, prev_pts = gray, detect(gray)
            tracked.append(None)
            continue

        H, _ = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 1, 2),
            good_new.reshape(-1, 1, 2),
            method=cv2.RANSAC,
        )

        if H is None:
            prev_gray = gray
            prev_pts  = good_new.reshape(-1, 1, 2)
            tracked.append(None)
            continue

        tx  = H[0, 2]
        ty  = H[1, 2]
        sx  = np.linalg.norm(H[0:2, 0])
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        sy  = det / sx if sx != 0 else 0.0
        rotation_deg = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
        shear        = (H[0, 0] * H[0, 1] + H[1, 0] * H[1, 1]) / det if det != 0 else 0.0

        tracked.append({
            "frame":        idx,
            "transform":    H,
            "trans X":      tx,
            "trans Y":      ty,
            "Scale X":      sx,
            "Scale Y":      sy,
            "rotation_deg": rotation_deg,
            "shear":        shear,
        })

        prev_gray = gray
        prev_pts  = good_new.reshape(-1, 1, 2)

        if (idx + 1) % 30 == 0:
            print(f"  [tracking] {idx + 1}/{n_frames} frames")

    return tracked


def track_with_phase(
    video_path: str,
    template: np.ndarray,
    output_dir: str,
) -> list[dict]:
    vid = VidObj(video_path)                     # <-- VidObj
    n_frames = vid.len
    tracked = []

    if template is not None:
        prev_tmp = normalize_and_enhance(template)
    else:
        print("Warning: No template provided for phase correlation tracking. Results will be empty.")
        prev_tmp = None

    for idx, frame in enumerate(vid.frames_gen()):  # <-- generator
        if prev_tmp is not None:
            result, _ = imreg_dft_emulate(
                normalize_and_enhance(frame, gen_mask(frame)),
                prev_tmp,
                output_dir=output_dir
            )
            result["frame"] = idx
            tracked.append(result)
        else:
            tracked.append(None)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [tracking] {idx + 1}/{n_frames} frames")

    return tracked


# ----------------------------------------------------------------------
#  Pipelines – video objects now created via VidObj
# ----------------------------------------------------------------------

def xCorr_pipeline_OG(video_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # First frame: use VidObj.read_frame(0) instead of VideoCapture
    vid = VidObj(video_path)
    first_frame = vid.read_frame(0)
    if first_frame is None:
        raise IOError("Could not read first frame from video.")

    print("► Select an ROI on the first frame, then press Enter / Space to confirm.")
    roi, template, origin_center = _get_inscribed_square(first_frame)
    print(f"  ROI  : x={roi[0]}  y={roi[1]}  w={roi[2]}  h={roi[3]}")
    print(f"  Center : {origin_center}")

    print("\n► Tracking ROI across all frames (cross-correlation)…")
    tracked_points = track_with_cross_correlation(video_path, roi, template, origin_center)
    print(f"  Tracked {len(tracked_points)} frames.")

    tracking_video = os.path.join(output_dir, "motion_tracking.mp4")
    print(f"\n► Rendering motion-tracking video…\n  → {tracking_video}")
    render_tracking_video(video_path, tracked_points, tracking_video, roi)

    stabilized_video = os.path.join(output_dir, "sclera_stabilized_XC.mp4")
    print(f"\n► Rendering stabilized video…\n  → {stabilized_video}")
    render_stabilized_video(video_path, tracked_points, stabilized_video)

    return


def xCorr_pipeline_Homo(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # select_wanted_frame originally used cv2.VideoCapture; you'll need to adapt it
    # to accept a VidObj or we can extract the frame ourselves.
    # For now, we use VidObj to pick a frame (for example, the first one).
    vid = VidObj(video_path)
    template_frame = vid.read_frame(0)          # adapt as needed
    print("Using first frame as template (modify if you want interactive selection).")

    tracked_points = track_with_homography2(video_path, best_frame=template_frame)

    print("\n► Rendering stabilized video…")
    render_stabilized_video(video_path, tracked_points,
                            os.path.join(output_dir, "sclera_stabilized_XC.mp4"))
    print(f"  Stabilized video → {os.path.join(output_dir, 'sclera_stabilized_XC.mp4')}")

    csv_path = os.path.join(output_dir, "tracking_results.csv")
    save_tracking_csv(tracked_points, csv_path)
    print(f"\n  Tracking CSV → {csv_path}")


def xCorr_pipeline_Phase(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    vid = VidObj(video_path)
    first_frame = vid.read_frame(0)
    if first_frame is None:
        raise IOError("Could not read first frame from video.")

    roi, template, origin_center = _get_inscribed_square(first_frame)
    print("\n► Tracking ROI across all frames (cross-correlation)…")
    tracked_points = track_with_cross_correlation(video_path, roi, template, origin_center)
    print(f"  Tracked {len(tracked_points)} frames.")

    print("\n► Rendering stabilized video…")
    render_tracking_video(video_path, tracked_points,
                          os.path.join(output_dir, "motion_tracking.mp4"), roi)
    render_tiff(video_path, tracked_points, output_dir)

    csv_path = os.path.join(output_dir, "tracking_results.csv")
    save_tracking_csv(tracked_points, csv_path)
    print(f"\n  Tracking CSV → {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track an ROI via cross-correlation and render outputs.")
    parser.add_argument("--video", default="output\\results_20260502-001448\\sclera_overlay.mp4",
                        help="Path to the input video.")
    parser.add_argument("--output-dir", default="output/jupyter_test/",
                        help="Directory to save outputs (motion video + CSV).")
    return parser.parse_args()


def chosen_pipeline(video_path: str, output_dir: str, *args):
    xCorr_pipeline_Phase(video_path=video_path, output_dir=output_dir)


if __name__ == "__main__":
    args = parse_args()
    chosen_pipeline(video_path=args.video, output_dir=args.output_dir)