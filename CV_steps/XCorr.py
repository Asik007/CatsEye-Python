import argparse
import cv2
import numpy as np
import os
import csv

try:
    from CV_steps.render import render_tracking_video, render_stabilized_video, select_wanted_frame
except ImportError as e:
    from render import render_tracking_video, render_stabilized_video, select_wanted_frame

def _select_roi(first_img: np.ndarray):
    """
    Show the first frame and let the user draw an ROI rectangle.
    Returns (x, y, w, h), template crop, and center point.
    """
    window_name = "Select ROI  –  Enter / Space to confirm"
    shrink = 0.5
    display_img = cv2.resize(first_img, (0, 0), fx=shrink, fy=shrink)
    roi = cv2.selectROI(window_name, display_img, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(window_name)
    print(f"Selected ROI (on displayed image): {roi}")
    # x, y, w, h = map(int, roi)
    # scale back up to original image coordinates
    x,y,w,h = (np.array(roi) / shrink).astype(int)
    # x, y, w, h = scaled_roi

    if w <= 0 or h <= 0:
        raise ValueError("No ROI selected.")
    template = first_img[y : y + h, x : x + w].copy()
    center   = (x + w // 2, y + h // 2)
    return (x, y, w, h), template, center

def _get_inscribed_square(first_img: np.ndarray) -> tuple:
    mask = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    M = cv2.moments(mask)

    # Calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    print(f"Center: ({cX}, {cY})")

    per_sel = 0.2



    side = int(min(first_img.shape[0] * per_sel, first_img.shape[1] * per_sel))
    template = first_img[int(cY - side // 2):int(cY + side // 2), int(cX - side // 2):int(cX + side // 2)].copy()

    print(f"Inscribed square ROI: center=({cX}, {cY}), side={side}")
    # draw the given ROI on the image and show it
    roi_img = first_img.copy()
    print(f"if you like this auto-selected ROI, press 'y' to confirm. Otherwise, press 'n' to select manually.")
    cv2.rectangle(roi_img, (cX - side // 2, cY - side // 2), (cX + side // 2, cY + side // 2), (0, 255, 0), 2)
    cv2.imshow("y = confirm, n = select manually", roi_img)
    key = cv2.waitKey(0)
    if key == ord('y'):
        print("auto ROI confirmed by user.")
    if key == ord('n'):
        print("auto ROI rejected by user. Please select ROI manually.")
        cv2.destroyAllWindows()
        return _select_roi(first_img)


    return (cX, cY, side, side), template, (cX,cY)

def gen_mask(
    frame: np.ndarray,
    ) -> np.ndarray:
    
    frame_gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame_binary = cv2.threshold(frame_gray, 1, 255, cv2.THRESH_BINARY)[1]

    # shrink it by 50 px (MAKE THIS DEPENDENT ON THE SIZE OF THE VIDEO — 50px is arbitrary and may not work for all resolutions)
    shrink_dist = min(frame.shape[1] * 0.1, frame.shape[0] * 0.1)
    dist = cv2.distanceTransform(frame_binary, cv2.DIST_L2, 5) # calculate distance from edge for each pixel
    mask = (dist > shrink_dist).astype(np.uint8) * 255  # keep only pixels >50px from any edge
    return mask


def track_with_cross_correlation(
    video_path: str,
    roi: tuple,
    template: np.ndarray,
    origin_center: tuple,
) -> list[dict]:
    """
    Match `template` in every frame via TM_CCOEFF_NORMED.

    Returns a list of dicts:
        { frame, center, displacement, match_score }
    """
    _x, _y, roi_w, roi_h = roi
    cap      = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracked  = []

    for idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        result   = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)

        cx = max_loc[0] + roi_w // 2
        cy = max_loc[1] + roi_h // 2
        tracked.append({
            "frame":        idx,
            "center":       (cx, cy),
            "displacement": (cx - origin_center[0], cy - origin_center[1]),
            "match_score":  float(score),
        })

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [tracking] {idx + 1}/{n_frames} frames")

    cap.release()
    return tracked

def track_with_homography(video_path: str, best_frame: np.ndarray = None) -> list[dict]:
    """
    Track motion across frames using ORB feature matching + RANSAC homography.
    Returns a list of {frame, transform} dicts.
    """
    cap      = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracked  = []

    # orb = cv2.ORB_create()    
    # bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    # flann = cv2.FlannBasedMatcher(index_params,search_params)

    orb = cv2.SIFT_create()  # try SIFT instead of ORB for better feature detection (but slower)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # use L2 norm for SIFT descriptors
    # orb = cv2.xfeatures2d.FREAK_create()
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # orb = cv2.xfeatures2d.SURF_create(400)
    # orb = cv2.xfeatures2d.FREAK.create()
    
    prev_frame = None
    prev_kp    = None
    prev_des   = None

    if best_frame is not None:
        print("Using provided best_frame for tracking.")
        mask = gen_mask(best_frame)
        # detect & describe features in this frame
        prev_kp, prev_des = orb.detectAndCompute(best_frame, mask=mask)
        prev_frame = best_frame.copy()
        


    for idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        mask = gen_mask(frame)

        # detect & describe features in this frame
        kp2, des2 = orb.detectAndCompute(frame, mask=mask)

        # skip the very first frame — nothing to match against yet
        if prev_frame is None:
            prev_frame, prev_kp, prev_des = frame.copy(), kp2, des2
            tracked.append(None)
            continue

        if prev_des is None or des2 is None:
            prev_frame, prev_kp, prev_des = frame.copy(), kp2, des2
            tracked.append(None)
            continue

        # match, draw, filter
        # flann = cv2.FlannBasedMatcher(index_params,search_params)
        # matches = flann.knnMatch(prev_des,des2,k=2)

        # --- lowes test for flann
        # ratio_thresh = 0.7
        # good_matches = []
        # for m,n in matches:
        #     if m.distance < ratio_thresh * n.distance:
        #         good_matches.append(m)
        
        
        # The standard way
        matches = sorted(bf.match(prev_des, des2), key=lambda m: m.distance)
        # viewer  = cv2.drawMatches(prev_frame, prev_kp, frame, kp2,
        #                    matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # viewer = cv2.drawKeypoints(frame, kp2, None)
        # draw the mask roi
        # green = np.zeros_like(viewer)
        # green[:, :, 1] = mask
        # viewer = cv2.addWeighted(viewer, 1.0, green, 0.5, 0)
        # resize_viewer = cv2.resize(viewer, (800, 600))
        # cv2.imshow("matches", resize_viewer)
        # cv2.waitKey(1)

        if len(matches) < 4:
            prev_frame, prev_kp, prev_des = frame.copy(), kp2, des2
            tracked.append(None)
            continue


        #-- Localize the object
        # obj = np.empty((len(good_matches),2), dtype=np.float32)
        # scene = np.empty((len(good_matches),2), dtype=np.float32)
        # for i in range(len(good_matches)):
        #     #-- Get the keypoints from the good matches
        #     obj[i,0] = prev_kp[good_matches[i].queryIdx].pt[0]
        #     obj[i,1] = prev_kp[good_matches[i].queryIdx].pt[1]
        #     scene[i,0] = kp2[good_matches[i].trainIdx].pt[0]
        #     scene[i,1] = kp2[good_matches[i].trainIdx].pt[1]
        # estimate homography

        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # H, what = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
        H, what = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)  # try affine first (translation + rotation + scale)
        
        

        # print(what)
        if H is None:
            prev_frame, prev_kp, prev_des = frame.copy(), kp2, des2
            tracked.append(None)
            continue
        # 1. Translation: Directly from the last column
        tx = H[0, 2]
        ty = H[1, 2]
        
        # 2. Scaling: Magnitude of the basis vectors
        # sx is the length of the first column (x-basis)
        sx = np.linalg.norm(H[0:2, 0])
        
        # sy uses the determinant to correctly identify reflection/flipping
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        sy = det / sx
        
        # 3. Rotation: Angle of the first column
        rotation_rad = np.arctan2(H[1, 0], H[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        
        # 4. Shear: How much the axes are "un-square"
        shear = (H[0, 0] * H[0, 1] + H[1, 0] * H[1, 1]) / det

        tracked.append({"frame": idx, "transform": H,
                        "trans X": tx, "trans Y": ty,
                        "Scale X": sx, "Scale Y": sy,
                        "rotation_deg": rotation_deg,
                        "shear": shear})

        if (idx + 1) % 1 == 0 or idx == 0:
            print(f"  [tracking] {idx + 1}/{n_frames} frames")

    cap.release()
    return tracked


EDGE_MARGIN_RATIO = 0.05
FEATURE_PARAMS    = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
LK_PARAMS         = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
REDETECT_EVERY    = 30


def track_with_homography2(video_path: str, best_frame: np.ndarray = None) -> list[dict]:
    """
    Track motion across frames using LK optical flow + RANSAC affine transform.
    Drop-in replacement for the SIFT version — same signature and return format.
    Returns a list of {frame, transform, ...} dicts (None where tracking failed).
    """
    def build_mask(gray: np.ndarray) -> np.ndarray:
        binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        margin = min(gray.shape[:2]) * EDGE_MARGIN_RATIO
        dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        return (dist > margin).astype(np.uint8) * 255

    def detect(gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(gray, mask=build_mask(gray), **FEATURE_PARAMS)
        return pts.astype(np.float32) if pts is not None else np.empty((0, 1, 2), np.float32)

    cap      = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracked  = []

    # ── Seed with best_frame if provided ──────────────────────────────────────
    if best_frame is not None:
        print("Seeding tracker with provided best_frame.")
        prev_gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
        prev_pts  = detect(prev_gray)
    else:
        prev_gray = prev_pts = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    for idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame with no seed — nothing to track against yet
        if prev_gray is None:
            prev_gray, prev_pts = gray, detect(gray)
            tracked.append(None)
            continue

        # Re-detect when points run low or on schedule
        if len(prev_pts) < 10 or idx % REDETECT_EVERY == 0:
            prev_pts = detect(prev_gray)

        if len(prev_pts) < 4:
            prev_gray, prev_pts = gray, detect(gray)
            tracked.append(None)
            continue

        # Track prev_pts forward into current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)
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

        # ── Decompose (identical format to the original) ──────────────────────
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

    cap.release()
    return tracked


def _save_tracking_csv(tracked_points: list[dict], output_path: str) -> None:
    """Save per-frame tracking data plus mean / std summary rows."""
    # fieldnames = ["frame", "center_x", "center_y", "disp_x", "disp_y", "match_score"]
    # value_types = [type(v) for v in tracked_points[0].values()]

    # centers = np.array([p["center"] for p in tracked_points])
    # mean = centers.mean(axis=0)
    # std  = centers.std(axis=0)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tracked_points[0].keys())
        writer.writeheader()

        for p in tracked_points:
            if p is not None:
                writer.writerow(p)
            else:
                writer.writerow({"frame": "tracking_failed"})
            # writer.writerow({
            #     "frame":       p["frame"],
            #     "center_x":   p["center"][0],
            #     "center_y":   p["center"][1],
            #     "disp_x":     p["displacement"][0],
            #     "disp_y":     p["displacement"][1],
            #     "match_score": p["match_score"],
            # })

        # writer.writerow({"frame": "mean", "center_x": mean[0], "center_y": mean[1]})
        # writer.writerow({"frame": "std",  "center_x": std[0],  "center_y": std[1]})


def xCorr_pipeline_debug(
    video_path: str,
    output_dir: str,
) -> dict:
    """
    Full pipeline:

      1. Open the video and show the first frame for ROI selection.
         The chosen ROI is used as the cross-correlation template.
      2. Track the ROI across every frame → displacement / motion data.
      3. Render a *motion-tracking video* that overlays the trail,
         moving ROI box, and per-frame displacement on the original footage.

    Returns a dict with paths to all outputs and summary counts.
    """

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. First frame → ROI selection ────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("Could not read first frame from video.")

    print("► Select an ROI on the first frame, then press Enter / Space to confirm.")
    # roi, template, origin_center = _select_roi(first_frame)
    roi, template, origin_center = _get_inscribed_square(first_frame)
    print(f"  ROI  : x={roi[0]}  y={roi[1]}  w={roi[2]}  h={roi[3]}")
    print(f"  Center : {origin_center}")
    
    # ── 2. Cross-correlation tracking ─────────────────────────────────────────
    print("\n► Tracking ROI across all frames (cross-correlation)…")
    tracked_points = track_with_cross_correlation(video_path, roi, template, origin_center)
    print(f"  Tracked {len(tracked_points)} frames.")
    
    # ── 3. Motion-tracking video ───────────────────────────────────────────────
    tracking_video = os.path.join(output_dir, "motion_tracking.mp4")
    print(f"\n► Rendering motion-tracking video…\n  → {tracking_video}")
    render_tracking_video(video_path, tracked_points, tracking_video, roi)
    
    # stabilize the video based on the tracked points
    stabilized_video = os.path.join(output_dir, "sclera_stabilized_XC.mp4")
    print(f"\n► Rendering stabilized video…\n  → {stabilized_video}")
    render_stabilized_video(video_path, tracked_points, stabilized_video)

    # CSV
    # csv_path = os.path.join(output_dir, "tracking_results.csv")
    # _save_tracking_csv(tracked_points, csv_path)
    # print(f"\n  Tracking CSV → {csv_path}")
    return

def xCorr_pipeline(
    video_path: str,
    output_dir: str,
):
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. First frame → ROI selection ────────────────────────────────────────

    idx, template_frame = select_wanted_frame(video_path)
    print(f"Selected frame index: {idx}")

    # select the largest square inscribed within the mask
    # roi, template, origin_center = _get_inscribed_square(first_frame)
    # print("\n► Tracking ROI across all frames (cross-correlation)…")
    tracked_points = track_with_homography2(video_path, best_frame=template_frame)
    # print(f"  Tracked {len(tracked_points)} frames.")
    # print(tracked_points)

    print("\n► Rendering stablized video…")
    render_stabilized_video(video_path, tracked_points, os.path.join(output_dir, "sclera_stabilized_XC.mp4"))
    print(f"  Stabilized video → {os.path.join(output_dir, 'sclera_stabilized_XC.mp4')}")

    # write csv
    csv_path = os.path.join(output_dir, "tracking_results.csv")
    _save_tracking_csv(tracked_points, csv_path)
    print(f"\n  Tracking CSV → {csv_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track an ROI via cross-correlation and render outputs."
    )
    parser.add_argument(
        "--video",
        default="output\\results_20260502-001448\\sclera_overlay.mp4",
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/jupyter_test/",
        help="Directory to save outputs (motion video + CSV).",
    )
    # parser.add_argument(
    #     "--smooth-radius",
    #     type=int,
    #     default=50,
    #     help="Smoothing radius (currently unused).",
    # )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    xCorr_pipeline_debug(
        video_path=args.video,
        output_dir=args.output_dir,
        # smooth_radius=args.smooth_radius,
    )


