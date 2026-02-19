import cv2
import numpy as np
import os
import json

def generate_masks(json_folder_path, output_mask_dir, allowed_sets=("train",)):
    """
    Generate binary masks from polygon and polyline annotations in multiple JSON files in a folder.

    Args:
        json_folder_path (str): Path to the folder containing JSON files with annotations.
        output_mask_dir (str): Directory to save the generated masks.
        allowed_sets (tuple): Tuple of allowed set names (default: ("train",)).
    """
    # Ensure output directory exists
    os.makedirs(output_mask_dir, exist_ok=True)

    # Loop through all JSON files in the folder
    for json_file in os.listdir(json_folder_path):
        if json_file.endswith(".json"):  # Process only JSON files
            json_file_path = os.path.join(json_folder_path, json_file)
            print(f"Processing JSON file: {json_file_path}")

            # Load JSON annotations
            with open(json_file_path, 'r') as f:
                jsonData = json.load(f)

            # Process each label in the JSON file
            for label in jsonData['data']:
                # Extract relevant information
                imageFileName = label['fileName']
                annotations = label['regionLabel']
                category = label['set']
                width = label['width']
                height = label['height']

                # Skip images not in allowed sets
                if category not in allowed_sets:
                    continue

                # Skip images with no annotations
                if len(annotations) == 0:
                    print(f"Skipping image {imageFileName} (empty annotations)")
                    continue

                # Initialize blank masks for different regions
                mask = np.zeros((height, width), dtype=np.uint8)

                # Process regions for polygons and polylines
                for region in annotations:
                    polygon_points = []
                    for point in region['points']:
                        point_x = point[0]
                        point_y = point[1]
                        polygon_points.append([point_x, point_y])

                    pts = np.array(polygon_points, np.int32)

                    # Default stroke width to 1 if not specified for PolyLine
                    stroke_width = region.get('strokeWidth', 1)

                    if region['className'] == 'weld':
                        if region['type'] == 'PolyGon':
                            cv2.fillPoly(mask, [pts], 255)
                        elif region['type'] == 'PolyLine':
                            cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=stroke_width)

                # Save the mask
                image_filename = os.path.splitext(imageFileName)[0]
                mask_output_path = os.path.join(output_mask_dir, f"{image_filename}_mask.png")
                cv2.imwrite(mask_output_path, mask)
                print(f"Mask saved at {mask_output_path}")

# Paths
json_file_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Deep Images_20241210/dataset/clamp_deep_images_weld/dataset_original/labels"
output_mask_dir = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Deep Images_20241210/dataset/clamp_deep_images_weld/dataset_original/train_masks"

# Generate masks
generate_masks(json_file_path, output_mask_dir)
