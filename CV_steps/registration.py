import argparse
from CV_steps.Helpers.Image_Input import select_roi_scaling
import cv2
import numpy as np

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track an ROI via cross-correlation and render outputs.")
    parser.add_argument("--video", default="output\\results_20260502-001448\\sclera_overlay.mp4",
                        help="Path to the input video.")
    parser.add_argument("--output-dir", default="output/jupyter_test/",
                        help="Directory to save outputs (motion video + CSV).")
    return parser.parse_args()


def chosen_pipeline(video_path: str, output_dir: str, *args):
    # xCorr_pipeline_Phase(video_path=video_path, output_dir=output_dir)
    print("uhhh no pipelines")

if __name__ == "__main__":
    args = parse_args()
    chosen_pipeline(video_path=args.video, output_dir=args.output_dir)