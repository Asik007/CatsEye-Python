from typing import Any, cast

import numpy as np
import cv2
from cv2 import Mat
from numpy import ndarray


def bin_complement(binary_image: np.ndarray) -> np.ndarray:
    return ~binary_image

def bin_pipeline(binary_image: np.ndarray) -> np.ndarray:
    OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    complement = bin_complement(binary_image)
    # cv2.imshow("complement", complement)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, OPEN_KERNEL)
    # cv2.imshow("open_image", binary_image)
    # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, CLOSE_KERNEL)
    # cv2.imshow("close_image", binary_image)
    fin_bin = complement - binary_image
    print("pixels changed:", np.count_nonzero(fin_bin != opened))

    # cv2.imshow("pre_bin", fin_bin)
    # 3. Remove small disconnected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ~fin_bin,
        connectivity=8
    )

    MIN_AREA = 75

    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= MIN_AREA
    keep[0] = False  # discard background

    result = (keep[labels] * 255).astype(np.uint8)

    return result




def extract_seg(orig_bin_image: np.ndarray, seg_bin_image) -> tuple[Mat | ndarray, list[Any]]:
    skeleton = cv2.ximgproc.thinning(seg_bin_image)
    # select regions where only the skeleton is

    # Skeleton locations become a mask
    skeleton_mask = skeleton > 0

    # Connected components of original segmentation
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        orig_bin_image,
        connectivity=8
    )

    fin_seg = np.zeros_like(orig_bin_image)

    segments = []

    # Keep a component if its region contains skeleton pixels
    for label in range(1, num_labels):
        component = labels == label

        if np.any(component & skeleton_mask):
            fin_seg[component] = min(label, 255) # purely to visualize (1-179 = H from HSV)


            ys, xs = np.where(component)

            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1

            # Crop just this segment
            segment = component[y1:y2, x1:x2].astype(np.bool)

            diameter_along_seg = cv2.distanceTransform(segment.astype(np.uint8), cv2.DIST_L2, 3)
            mean_diameter = np.mean(diameter_along_seg)
            max_diameter = np.max(diameter_along_seg)
            min_diameter = np.min(diameter_along_seg)

            print(f"Mean diameter: {mean_diameter}")
            print(f"Max diameter: {max_diameter}")
            print(f"Min diameter: {min_diameter}")

            segments.append(segment)

    # Math to get info about segs

    colored = cv2.applyColorMap(fin_seg, cv2.COLORMAP_HSV)
    colored[fin_seg == 0] = 0

    return colored, segments

def vessel_math(skeletons: ndarray):
    # take in a binary image of a segment and just get the diameter at all points along the skeleton

    return

def data_bin(input_frame, output_path):
    cleaned = bin_pipeline(input_frame)
    # skeletonize and extract original segments/regions from the first image
    color_seg, skeletons = extract_seg(input_frame, cleaned)
    cv2.imwrite(output_path / "colored_segments.jpg", color_seg)
    print(f"skeletons: {len(skeletons)} example shape: {skeletons[0].shape}")
    skeleton_path = output_path/ "skeletons"
    skeleton_path.mkdir(parents=True, exist_ok=True)
    for i, skeleton in enumerate(skeletons):
        cv2.imwrite( skeleton_path / f"{i:02}.jpg", skeleton * 255)
    return cleaned

if __name__ == "__main__":
    bin_img_path = r"F:\Code\CatsEye-Python\output\testing\binary_mask_DCA1.png"
    test_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("test_img", test_img)
    cleaned = bin_pipeline(test_img)
    cv2.imshow("fin_img", cleaned)
    # skeletonize and extract original segments/regions from the first image
    color_seg, skeletons = extract_seg(test_img, cleaned)
    cv2.imshow("regions_img", color_seg)
    print(f"skeletons: {len(skeletons)} example shape: {skeletons[0].shape}")
    for i, skeleton in enumerate(skeletons):
        cv2.imwrite(f"F:/Code/CatsEye-Python/output/testing/skeletons/{i:02}.jpg", skeleton * 255)

    # get some stats from it

    cv2.waitKey(0)