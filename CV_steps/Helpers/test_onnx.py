"""
Test exported YOLO ONNX models: speed comparison and raw‑output MSE.
Processes a video, generates a CSV report, and shows bounding boxes.
"""

import csv
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ------------------------------ Configuration ------------------------------
PT_MODEL_PATH = r"C:\Users\dragon\Code\CatsEye-Python\ML_stuff\best.pt"  # original .pt model
# ONNX_MODEL_PATH = r"C:\Users\dragon\Code\CatsEye-Python\ML_stuff\best.onnx"  # exported ONNX
ONNX_MODEL_PATH = r"C:\Users\dragon\Code\CatsEye-Python\ML_stuff\exports\model_640_False.onnx"  # exported ONNX

VIDEO_PATH = r"C:\Users\dragon\Code\CatsEye-Python\uploads\IMG_1744.MOV"  # path to test video
IMGSZ = 640  # must match export size
CSV_OUTPUT = r"C:\Users\dragon\Code\CatsEye-Python\output\onnx_test_results.csv"


def main():
    # Load models
    model_pt = YOLO(PT_MODEL_PATH)  # PyTorch
    model_onnx = YOLO(ONNX_MODEL_PATH)
    # net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)  # ONNX (using OpenCV DNN)

    # The Magic:
    net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)

    import requests

    LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    labels = {int(key): value for (key, value)
              in requests.get(LABELS_URL).json().items()}

    print("The class", biggest_pred_index, "correspond to", labels[biggest_pred_index])




    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    # Prepare CSV
    csv_path = Path(CSV_OUTPUT)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_index", "pt_time_s", "onnx_time_s", "mse"])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        t0 = time.perf_counter()
        pt_out = model_pt(frame, imgsz=IMGSZ, verbose=False, task="segment",)
        pt_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (IMGSZ, IMGSZ), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        biggest_pred_index = np.array(preds)[0].argmax()
        print("Predicted class:", biggest_pred_index)

        # onnx_out = model_onnx(frame, imgsz=IMGSZ, verbose=False, task="segment")
        onnx_time = time.perf_counter() - t0
        print(f"Frame {frame_idx}: pt_time={pt_time:.6f}s, onnx_time={onnx_time:.6f}s")
        # print(f"MSE: {np.mean((onnx_out[0].masks - pt_out[0].masks) ** 2)}")

        print(onnx_out[0].masks)
        # MSE between raw output masks of each model
        try:
            mse = np.mean((onnx_out[0].masks - pt_out[0].masks) ** 2)
        except:
            mse = 0
        writer.writerow([frame_idx, f"{pt_time:.6f}", f"{onnx_time:.6f}", f"{mse:.10f}"])

        # show inferences

        # Show frame
        pt_result = pt_out[0].plot()  # This plots the detections on the image
        onnx_result = onnx_out[0].plot()

        # Convert BGR to RGB (OpenCV uses BGR by default)
        # pt_rgb = cv2.cvtColor(pt_result, cv2.COLOR_BGR2RGB)
        # onnx_rgb = cv2.cvtColor(onnx_result, cv2.COLOR_BGR2RGB)


        cv2.imshow("Press any key to advance, 'q' to quit ONNX", onnx_result)
        cv2.imshow("Press any key to advance, 'q' to quit PT", pt_result)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()