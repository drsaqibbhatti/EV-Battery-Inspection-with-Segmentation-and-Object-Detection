import cv2
import numpy as np
import os
import json

def generate_stacked_images(json_folder_path, image_folder_path, output_stack_dir, allowed_sets=("train",)):
    """
    Generate vertically stacked images with:
    - Original image with contours (pinhole annotations).
    - Binary filled mask.

    Args:
        json_folder_path (str): Path to the folder containing JSON files with annotations.
        image_folder_path (str): Path to the folder containing the original images.
        output_stack_dir (str): Directory to save the stacked images.
        allowed_sets (tuple): Tuple of allowed set names (default: ("train",)).
    """
    # Ensure output directory exists
    os.makedirs(output_stack_dir, exist_ok=True)

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

                # Initialize a blank mask for pinhole
                mask_pinhole = np.zeros((height, width), dtype=np.uint8)

                # Loop through annotations and process pinhole polygons and polylines
                for region in annotations:
                    if region['className'] == 'pinhole':
                        polygon_points = []
                        for point in region['points']:
                            point_x = point[0]
                            point_y = point[1]
                            polygon_points.append([point_x, point_y])

                        pts = np.array(polygon_points, np.int32)
                        
                        # Get the strokeWidth for PolyLine (default to 5 if not specified)
                        stroke_width = region.get('strokeWidth', 5)

                        if region['type'] == 'PolyGon':
                            cv2.fillPoly(mask_pinhole, [pts], (255))  # Filled polygon
                        elif region['type'] == 'PolyLine':
                            cv2.polylines(mask_pinhole, [pts], isClosed=True, color=(255), thickness=stroke_width)

                # Read the original image
                image_path = os.path.join(image_folder_path, imageFileName)
                if not os.path.exists(image_path):
                    print(f"Original image {image_path} not found. Skipping.")
                    continue

                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"Failed to read image {image_path}. Skipping.")
                    continue

                # Resize the mask to match the original image dimensions (if needed)
                if (mask_pinhole.shape[1] != original_image.shape[1]) or (mask_pinhole.shape[0] != original_image.shape[0]):
                    mask_pinhole = cv2.resize(mask_pinhole, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Create an overlay by drawing borders on the original image
                overlay = original_image.copy()
                contours, _ = cv2.findContours(mask_pinhole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)  # Red border for pinhole

                # Convert the mask to 3 channels for stacking
                mask_pinhole_colored = cv2.cvtColor(mask_pinhole, cv2.COLOR_GRAY2BGR)

                # Vertically stack the original image with overlay and the mask
                stacked_image = cv2.vconcat([overlay, mask_pinhole_colored])

                # Save the stacked image
                stacked_output_path = os.path.join(output_stack_dir, f"{os.path.splitext(imageFileName)[0]}_stacked.png")
                cv2.imwrite(stacked_output_path, stacked_image)
                print(f"Stacked image saved at {stacked_output_path}")



# Paths
json_folder_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/SV_busbar_v2/Busbar_Al/images_inspection/dataset/labels"
image_folder_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/SV_busbar_v2/Busbar_Al/images_inspection/dataset/images"
output_stack_dir = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/SV_busbar_v2/Busbar_Al/images_inspection/train_masks"

# Generate masks and stacked images
generate_stacked_images(json_folder_path, image_folder_path, output_stack_dir)
