import cv2
import torch
import numpy as np
import os
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image

def evaluate_model(model_path, data_path, save_dir):
    # Check if CUDA is available
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print("The device is:", device)

    # Load the model
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Transformations
    imageWidth = 640
    imageHeight = 640

    transNormalProcess = transforms.Compose([
        transforms.Resize((imageHeight, imageWidth)),
        transforms.ToTensor()
    ])

    # Directory to save predicted masks
    os.makedirs(save_dir, exist_ok=True)

    # Get all image files in the dataset directory
    image_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Use tqdm to create a progress bar
    with torch.no_grad():
        for i, image_path in enumerate(tqdm(image_files, desc="Evaluating", total=len(image_files))):
            # Load and preprocess the image using PIL
            image = Image.open(image_path).convert("RGB")
            torch_image = transNormalProcess(image)  # Apply transformations

            # Swap channels to BGR
            torch_image = torch_image[[2, 1, 0], :, :].unsqueeze(0).to(device)  # Add batch dimension

            # Perform inference
            output = model(torch_image)

            # Convert predictions to binary format
            preds = output.cpu().numpy().round().astype(int)
            predicted_mask = preds[0][0] * 255  # Convert mask to [0, 255]
            predicted_mask = predicted_mask.astype(np.uint8)

            resized_image = cv2.resize(np.array(image)[:, :, ::-1], (imageWidth, imageHeight))  # Convert RGB to BGR

            # Create an overlay of the resized image and the predicted mask
            overlay = resized_image.copy()
            contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # Draw red contours for prediction
    

            # Save the overlay image with the original file name
            base_filename = os.path.basename(image_path)
            overlay_filename = os.path.join(save_dir, f"overlay_{base_filename}")
            cv2.imwrite(overlay_filename, overlay)

    print(f"Overlays saved to {save_dir}")

if __name__ == "__main__":
    model_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/trainedModel/Phase_3/clamp/width/run_clamp_3/clamp_width_SegNextV2_2024-12-23_accuracy_train_0.98324_loss_0.00846_epoch_211_lr_0.003_time_1233.00M_alpha_0.7_beta_0.3_gamma_2_best.pth"
    data_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/ClampImages of One Object/Origin/CWAR1A02/output_boxes"
    save_dir = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/ClampImages of One Object/Origin/CWAR1A02/predicted_masks"  # Update with your desired folder to save predicted masks

    os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn't exist

    evaluate_model(model_path, data_path, save_dir)
