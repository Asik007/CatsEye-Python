"""
Author: Zubin Bhuyan
Date: June 21, 2023

MIT License

"""

import json
import os
from pathlib import Path

def convert_coco_to_yolo_segmentation(json_file, output_folder, image_folder):
    # Load the JSON file
    with open(json_file, 'r') as file:
        coco_data = json.load(file)

    # Create a "labels" folder to store YOLO segmentation annotations
    os.makedirs(output_folder, exist_ok=True)

    # Extract annotations from the COCO JSON data
    annotations = coco_data['annotations']
    for annotation in annotations:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']
        bbox = annotation['bbox']

        # Find the image filename from the COCO data
        for image in coco_data['images']:
            if image['id'] == image_id:
                image_filename = os.path.basename(image['file_name'])
                image_filename = os.path.splitext(image_filename)[0] # Removing the extension. (In our case, it is the .jpg or .png part.)
                image_width = image['width']
                image_height = image['height']
                break

        # Calculate the normalized center coordinates and width/height
        x_center = (bbox[0] + bbox[2] / 2) / image_width
        y_center = (bbox[1] + bbox[3] / 2) / image_height
        bbox_width = bbox[2] / image_width
        bbox_height = bbox[3] / image_height

        # Convert COCO segmentation to YOLO segmentation format
        yolo_segmentation = [f"{(x) / image_width:.5f} {(y) / image_height:.5f}" for x, y in zip(segmentation[0][::2], segmentation[0][1::2])]
        #yolo_segmentation.append(f"{(segmentation[0][0]) / image_width:.5f} {(segmentation[0][1]) / image_height:.5f}")
        yolo_segmentation = ' '.join(yolo_segmentation)

        # Generate the YOLO segmentation annotation line
        yolo_annotation = f"{category_id} {yolo_segmentation}"

        # Save the YOLO segmentation annotation in a file
        output_filename = os.path.join(output_folder, f"{image_filename}.txt")
        output_image_path = os.path.join(image_folder, f"{image_filename}.jpg") # Assuming the images are in .jpg format
        print(f"Saving annotation to: C:-Users\dragon\Code\CatsEye-Python\{output_filename}")
        # copy the imaages to the images folder
        # get all files other than the json file in the same folder as the json file
        
        with open(output_filename, 'a+') as file:
            file.write(yolo_annotation + '\n')
        with open(output_image_path, 'wb') as img_file:
            with open(os.path.join(os.path.dirname(json_file), image['file_name']), 'rb') as source_img:
                img_file.write(source_img.read())
    print("Conversion completed. YOLO segmentation annotations saved in 'labels' folder.")


def make_data_yaml(output_folder, classes=["Eye"]):
    folder_keys = {"train": "train", "val": "valid", "test": "test"}
    
    path_lines = ""
    for key, folder_name in folder_keys.items():
        folder_path = os.path.join(output_folder, folder_name, "images")
        if os.path.exists(folder_path):
            path_lines += f"\n    {key}: \"{folder_name}/images\""

    data_yaml_content = f"""{path_lines}
    nc: {len(classes)}
    names:
    # 0: "{classes[0]}" """

    data_yaml_path = os.path.join(output_folder, "data.yaml")
    with open(data_yaml_path, 'w') as file:
        file.write(data_yaml_content)

    print(f"data.yaml file created at: C:-Users\dragon\Code\CatsEye-Python\{data_yaml_path}")
    



# Example usage
input_folder = Path("ML_stuff\\coco_data") #Input folder root with your subfolders
path_names = [x for x in input_folder.iterdir() if x.is_dir()] #List of folders in the input folder
# stems = [x.name for x in path_names]
print(path_names)
# names = ["valid", "train", "test"] #List of folders
output_folder = Path("ML_stuff\\Yolo_dataset") #Output folder name
for pname in path_names:
    anno_file = f"{pname}/_annotations.coco.json"
    new_folder = os.path.join(output_folder, pname.name)
    new_folder_images = os.path.join(new_folder, "images")
    new_folder_labels = os.path.join(new_folder, "labels")
    os.makedirs(new_folder, exist_ok=True)
    os.makedirs(new_folder_images, exist_ok=True)
    os.makedirs(new_folder_labels, exist_ok=True)
    convert_coco_to_yolo_segmentation(anno_file, new_folder_labels, new_folder_images)
make_data_yaml(output_folder)

# yolo training command:
