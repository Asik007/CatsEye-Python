from __future__ import annotations

import argparse
from pathlib import Path
from line_profiler import profile
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


def load_segmentation_model(model_path: str | Path) -> YOLO:
    return YOLO(str(model_path))


def infer_mask(
    image_bgr: np.ndarray,
    model: YOLO,
    target_class: Optional[str] = "Eye",
    conf: float = 0.25,
    imgsz: int = 160,
) -> np.ndarray:
    """
    Returns a binary mask (uint8, 0/255) for the target class.
    If target_class is None, combines all predicted instance masks.
    """
    results = model.predict(source=image_bgr, conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    result = results[0]
    # print(f"Inference results: {result.con}")
    h, w = image_bgr.shape[:2]
    out_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return out_mask

    class_ids = None

    # get the ids of the predicted classes if they exist
    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

    names = getattr(model, "names", {})
    target_id = None
    if target_class is not None:
        for k, v in names.items():
            if str(v).lower() == target_class.lower():
                target_id = int(k)
                break

    mask_data = result.masks.data.detach().cpu().numpy()  # (n, mask_h, mask_w)
    print(f"Mask data shape: {result.boxes.conf}, Class IDs: {class_ids}")

    for i, mask in enumerate(mask_data):
        if target_class is not None and target_id is not None and class_ids is not None:
            if i >= len(class_ids) or class_ids[i] != target_id:
                continue

        mask_resized = cv2.resize(
            mask.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        out_mask = np.maximum(out_mask, (mask_resized > 0.5).astype(np.uint8) * 255)

    return out_mask


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def process_image(
    image_bgr: np.ndarray,
    # model: str | Path,
    # output_mask_path: Optional[str | Path] = None,
    # output_overlay_path: Optional[str | Path] = None,
    target_class: Optional[str] = "sclera",
    conf: float = 0.25,
    imgsz: int = 640,
    model: Optional[YOLO] = None,
) -> tuple[np.ndarray, np.ndarray]:
    # image_path = Path(image_path)
    # model_path = Path(model_path)

    # image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError("Could not read image")

    mask = infer_mask(image_bgr, model, target_class=target_class, conf=conf, imgsz=imgsz)

    overlay = apply_mask(image_bgr, mask)

    return mask, overlay


@profile
def process_video(
    video_path: str | Path,
    model_path: str | Path,
    output_mask_path: Optional[str | Path] = None,
    output_overlay_path: Optional[str | Path] = None,
    target_class: Optional[str] = "sclera",
    conf: float = 0.25,
    imgsz: int = 640,
) -> None:
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # ✅ Read BEFORE release
    if fps <= 0:
        fps = 30.0
    
    print(f"Processing video: {video_path}")
    print(f"Frames: {n_frames}, Resolution: {w}x{h}, FPS: {fps}")
    
    model = load_segmentation_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # apply the inference and save the output videos
    # mask
    if output_mask_path is not None:
        output_mask_path = Path(output_mask_path)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_writer = cv2.VideoWriter(
            str(output_mask_path), fourcc, fps, (w, h), isColor=False
        )
        print(f"Mask output: {output_mask_path}")
    
    # overlay
    if output_overlay_path is not None:
        output_overlay_path = Path(output_overlay_path)
        output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_writer = cv2.VideoWriter(
            str(output_overlay_path), fourcc, fps, (w, h)
        )
        print(f"Overlay output: {output_overlay_path}")
    
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i}, stopping early.")
            break

        mask, overlay = process_image(
            image_bgr=frame,
            model=model,
            target_class=target_class,
            conf=conf,
            imgsz=imgsz,
        )



        if output_mask_path is not None:
            mask_writer.write(mask)

        if output_overlay_path is not None:
            overlay_writer.write(overlay)
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{n_frames} frames processed")

    cap.release()

    if output_mask_path is not None:
        mask_writer.release()
    if output_overlay_path is not None:
        overlay_writer.release()
    
    print("Video processing complete!")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a sclera mask using a YOLO segmentation model."
    )
    parser.add_argument("--image", help="Input image path", default=None)
    parser.add_argument("--video", help="Input video path", default="uploads/IMG_1673.mov")
    parser.add_argument("--model", help="YOLO segmentation model path", default="ML_stuff/best.pt")
    parser.add_argument("--out-mask", help="Path to save binary mask", default="output/testing_sclera/sclera_mask_ML.mp4")
    parser.add_argument("--out-overlay", help="Path to save masked overlay", default="output/testing_sclera/sclera_overlay_ML.mp4")
    parser.add_argument(
        "--class-name",
        default="Eye",
        help="Target class name in the model. Use 'all' to combine all masks.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    return parser.parse_args()
    



def run_image(
        image_bgr: np.ndarray,
        model: Optional[YOLO],
        target_class: Optional[str],
        conf: float,
        imgsz: int
) -> None:
    # args = parse_args()
    target_class = None if args.class_name.lower() == "all" else args.class_name
    model = load_segmentation_model(args.model)
    image_bgr = cv2.imread(str(args.image))

    mask, overlay = process_image(
        image_bgr=image_bgr,
        model=model,
        # output_mask_path=args.out_mask,
        # output_overlay_path=args.out_overlay,
        target_class=target_class,
        conf=args.conf,
        imgsz=args.imgsz,
    )

    
    cv2.imwrite(str(args.out_mask), mask)
    cv2.imwrite(str(args.out_overlay), overlay)





DEBUG = False

if __name__ == "__main__":
    # DEBUG = input("Run in debug mode? (y/N): ").strip().lower() == "y"
    args = parse_args()

    if args.image is not None:
        run_image(
            image_bgr=None,
            model=None,
            # target_class=None,
            conf=args.conf,
            imgsz=args.imgsz
        )
    else:

        process_video(
        video_path=args.video,
        model_path=args.model,
        output_mask_path=args.out_mask,
        output_overlay_path=args.out_overlay,
        # target_class=args.target_class,
        conf=args.conf,
        imgsz=args.imgsz,
        )


# test command:
# python CV_steps/sclera_ML.py --image uploads/frames/frame0009.jpg --out-mask output/mask.png --out-overlay output/overlay.png

# python CV_steps/sclera_ML.py --video uploads/IMG_1673.mov --imgsz 256
# kernprof -l -v CV_steps/sclera_ML.py --video uploads/IMG_1673.mov --imgsz 256

# this script is essentially the equivalent of this command:
# yolo predict model=ML_stuff/best.pt source=uploads/IMG_1673.mov conf=0.25 imgsz=64 save_txt=True save_mask=True name=sclera_ML_output