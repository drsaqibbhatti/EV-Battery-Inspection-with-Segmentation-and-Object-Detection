import os
import json
import csv

def process_json_folder(json_folder_path, output_csv_folder):
    """
    Process all JSON files in a folder and create CSV files with the same name.

    Args:
        json_folder_path (str): Path to the folder containing JSON files.
        output_csv_folder (str): Path to the folder where CSV files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_csv_folder, exist_ok=True)

    # Loop through all JSON files in the folder
    for json_file in os.listdir(json_folder_path):
        if json_file.endswith('.json'):  # Process only JSON files
            json_file_path = os.path.join(json_folder_path, json_file)
            csv_file_name = os.path.splitext(json_file)[0] + ".csv"
            csv_file_path = os.path.join(output_csv_folder, csv_file_name)
            
            print(f"Processing {json_file_path} -> {csv_file_path}")

            # Read JSON and create CSV
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            rows = []
            for item in data.get('data', []):
                file_name = item.get('fileName', '')
                file_set = item.get('set', '')
                width = item.get('width', '')
                height = item.get('height', '')

                for region in item.get('regionLabel', []):
                    class_name = region.get('className', '')
                    region_type = region.get('type', '')

                    rows.append({
                        'fileName': file_name,
                        'set': file_set,
                        'className': class_name,
                        'type': region_type,
                        'width': width,
                        'height': height
                    })
            
            # Write rows to CSV
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['fileName', 'set', 'className', 'type', 'width', 'height'])
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"CSV file created: {csv_file_path}")

# Example usage
json_folder_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Labeling/Clamp Labeling/labels"  # Replace with the path to your folder containing JSON files
output_csv_folder = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Labeling/Clamp Labeling/labels"  # Replace with the path to save CSV files


#process_json_folder(json_folder_path, output_csv_folder)



def match_images_and_json(image_folder, json_folder, output_file):
    # Get the list of image filenames
    image_filenames = {f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))}

    # Prepare the output data
    matched_data = []

    # Loop through JSON files
    for json_file in os.listdir(json_folder):
        json_path = os.path.join(json_folder, json_file)
        if os.path.isfile(json_path) and json_file.endswith(".json"):
            # Load JSON data
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            # Extract matching entries from the `data` field
            filtered_data = [
                item for item in json_data.get("data", [])
                if item.get("fileName") in image_filenames
            ]

            # If any matching entries are found, include them
            if filtered_data:
                json_data["data"] = filtered_data  # Replace `data` with filtered entries
                matched_data.append(json_data)

    # Write the matched data to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(matched_data, f, indent=4)

    print(f"Matched data written to {output_file}")




# Replace these with your actual folder paths
image_folder = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Labeling/Clamp Labeling/Clamp Labeling/Origin image"
output_file = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Labeling/Clamp Labeling/labels/matched_data.json"

# # Call the function
# match_images_and_json(image_folder, json_folder_path, output_file)

import os
import json
import csv

def generate_missing_images_csv(image_folder, json_folder, output_csv):
    # Get the list of image filenames in the folder
    image_filenames = {f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))}

    # Prepare a list to hold missing files data
    missing_images = []

    # Loop through JSON files
    for json_file in os.listdir(json_folder):
        json_path = os.path.join(json_folder, json_file)
        if os.path.isfile(json_path) and json_file.endswith(".json"):
            # Load JSON data
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            # Check each `fileName` in the `data` field
            for item in json_data.get("data", []):
                file_name = item.get("fileName")
                set_name = item.get("set")
                if file_name and file_name not in image_filenames:
                    # Add missing image details to the list
                    missing_images.append({"fileName": file_name, "set": set_name})

    # Write missing images to a CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["fileName", "set"])
        writer.writeheader()  # Write header row
        writer.writerows(missing_images)  # Write rows

    print(f"CSV with missing images generated: {output_csv}")

# Replace with the paths to your folders and output CSV file
image_folder = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Labeling/dataset/images"
json_folder = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Labeling/dataset/labels"
output_csv = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Labeling/dataset/labels/missing_images.csv"

# Call the function
generate_missing_images_csv(image_folder, json_folder, output_csv)


