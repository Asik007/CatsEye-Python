# Import numpy and OpenCV
import numpy as np
import cv2
 
import os

# test code to only stabilize the mask video output from the sclera_ML.py script, using the same optical flow method as in sclera.py


# cap = cv2.VideoCapture(video_source)
video_source = r"uploads\IMG_1759.MOV"  # Change this to your video file path or camera index

testing_out = os.path.join("output", "jupyter_test")
os.makedirs(testing_out, exist_ok=True)
print(f"Testing output directory: {testing_out}")

outlined_path = os.path.join(testing_out, "sclera_outline.mp4")
mask_path = os.path.join(testing_out, "sclera_mask.mp4")

# Read input video
cap = cv2.VideoCapture(mask_path)
 
# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')


# Set up output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = os.path.join(testing_out, "mask_stabilized_test.mp4")
out_writer = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
print(f"writer set up at {out_path} with resolution {(w, h)} and FPS {cap.get(cv2.CAP_PROP_FPS)}")

# Read first frame
_, prev = cap.read()
 
# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# show features detected in the first frame
# make a mask of all the current data and then erode it shrink it
mask = cv2.threshold(prev_gray, 1, 255, cv2.THRESH_BINARY)[1]
# somehow shrink the mask

# kernel = np.ones((5,5), np.uint8)
# mask = cv2.erode(mask, kernel, iterations=5)

dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) # calculate distance from edge for each pixel
mask = (dist > 50).astype(np.uint8) * 255  # keep only pixels >50px from any edge

# show(mask)
features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3, mask=mask)
features = np.intp(features)

featured_prev = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
for feature in features:
    x, y = feature.ravel()
    cv2.circle(featured_prev, (x, y), 5, (0, 255, 0), -1)

# show(featured_prev)
# calculate the optical flow (i.e. track feature points)
for _ in range(1, n_frames):
    # Read next frame
    success, img = cap.read()
    if not success:
        print("No more frames to read or error reading frame.")
        break
 
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    mask = cv2.threshold(prev_gray, 1, 255, cv2.THRESH_BINARY)[1]

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) # calculate distance from edge for each pixel
    mask = (dist > 50).astype(np.uint8) * 255  # keep only pixels >50px from any edge

    # show(mask)
    features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3, mask=mask)
    features = np.intp(features)

    featured_prev = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
    for feature in features:
        x, y = feature.ravel()
        cv2.circle(featured_prev, (x, y), 5, (0, 255, 0), -1)

    # show(featured_prev)
    # Ensure features are in the correct format (float32, shape (n, 1, 2))
    if features is not None and len(features) > 0:
        features = features.astype(np.float32)# Calculate optical flow
    
    old_features = features.copy()
    features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, features, None)

    good_new = features[status == 1]
    good_old = old_features[status == 1]  # ← now actually the old positions
 
    # Estimate transformation matrix using RANSAC
    if len(good_new) >= 4:  # Need at least 4 points to compute homography
        matrix, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
        if matrix is not None:
            # Warp the current frame to align with the previous frame
            stabilized_frame = cv2.warpPerspective(img, matrix, (w, h))
            out_writer.write(stabilized_frame)
        else:
            print("Homography could not be computed for this frame.")
            out_writer.write(img)  # Write original frame if homography fails
    else:
        print("Not enough good points to compute homography for this frame.")
        out_writer.write(img)  # Write original frame if not enough points
 
    # Update previous frame and previous points
    prev_gray = gray.copy()
    features = good_new.reshape(-1, 1, 2)

# Release video objects
cap.release()
out_writer.release()
print(f"Stabilized video saved to {out_path}")


