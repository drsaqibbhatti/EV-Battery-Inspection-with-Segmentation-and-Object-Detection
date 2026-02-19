import os
import numpy as np
import torch
import onnxruntime as ort
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from util.ObjDatasetClamp import ObjDatasetClamp
from util.helper import non_max_suppression

# Output directory for cropped images
crop_output_dir = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/ClampImages of One Object/Origin/CWAR1A02/output_boxes"
os.makedirs(crop_output_dir, exist_ok=True)

# Parameters
imageHeight, imageWidth = 256, 256  # Input dimensions for model inference
transform = transforms.Compose([transforms.ToTensor()])

# Load ONNX model
onnx_model_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/SebangLabelMaker/Onnx_trained_model_for_cropping/CropModel.onnx"  # Update with your ONNX model path
ort_session = ort.InferenceSession(onnx_model_path)

# Dataset and DataLoader
validDataset = ObjDatasetClamp(
    root="/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/ClampImages of One Object/Origin/CWAR1A02",
    transform=transform,
    img_size=(imageHeight, imageWidth)
)

validLoader = DataLoader(
    validDataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=lambda x: tuple(zip(*x))
)

def preprocess_image(image):
    """
    Preprocess the image for ONNX model inference.
    """
    # Ensure the input is a 4D tensor (batch_size, channels, height, width)
    return image.numpy()

# Perform inference
for i, (images_val, orig_paths, orig_sizes) in enumerate(validLoader):
    # Prepare input image
    images_val = torch.stack([img for img in images_val], dim=0)  # Shape: (batch_size, 3, height, width)
    preprocessed_image = preprocess_image(images_val)

    # Run ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: preprocessed_image}
    ort_outputs = ort_session.run(None, ort_inputs)
    predictions = ort_outputs[0]  # Assuming the first output contains predictions

    # Original image
    orig_image_path = orig_paths[0]
    orig_image = Image.open(orig_image_path).convert("RGB")
    orig_width, orig_height = orig_sizes[0]

    # Apply Non-Maximum Suppression (NMS)
    outputs = non_max_suppression(torch.tensor(predictions), conf_threshold=0.5, iou_threshold=0.5)

    # Save cropped boxes
    for j, output in enumerate(outputs):
        if output is None or output.size(0) == 0:  # Skip if no predictions
            continue

        for k, (box, score, cls) in enumerate(zip(output[:, :4], output[:, 4], output[:, 5])):
            x_min, y_min, x_max, y_max = map(int, box.tolist())

            # Rescale to original image size
            scale_x = orig_width / imageWidth
            scale_y = orig_height / imageHeight
            x_min = int(x_min * scale_x)
            y_min = int(y_min * scale_y)
            x_max = int(x_max * scale_x)
            y_max = int(y_max * scale_y)

            # Crop the box area
            cropped_img = orig_image.crop((x_min, y_min, x_max, y_max))

            # Save the cropped image
            original_filename = os.path.basename(orig_image_path)
            cropped_filename = os.path.join(
                crop_output_dir, f"{os.path.splitext(original_filename)[0]}.jpg"
            )

            cropped_img.save(cropped_filename)
            print(f"Saved cropped box: {cropped_filename}")
