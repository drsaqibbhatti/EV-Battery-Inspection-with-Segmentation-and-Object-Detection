import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

imageWidth=1280
imageHeight=320
onnx_model_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/trainedModel/Phase_3/busbar_al/pinhole/run_5_img_1280_320_SELECTED/model_busbar_al_pinhole_SegN.onnx"
session = ort.InferenceSession(onnx_model_path)

image_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/SV_busbar_v2/Busbar_Al/images_inspection/Pinhole_오인식_Pass/Pinhole_오인식/crop_20250331_225718778.jpg"


# Load and preprocess image in BGR format (as it was loaded originally)
image_bgr = cv2.imread(image_path)  
image_bgr = cv2.resize(image_bgr, (imageWidth, imageHeight)) 
image_bgr = image_bgr.astype(np.float32) / 255.0  
image_bgr = np.transpose(image_bgr, (2, 0, 1))  
image_bgr = np.expand_dims(image_bgr, axis=0) 

# Load and preprocess image in RGB format
image_rgb = cv2.imread(image_path)  
image_rgb = cv2.resize(image_rgb, (imageWidth, imageHeight))
image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)  # Convert to RGB
image_rgb = image_rgb.astype(np.float32) / 255.0  
image_rgb = np.transpose(image_rgb, (2, 0, 1))  
image_rgb = np.expand_dims(image_rgb, axis=0) 

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference on BGR input
predicted_mask_bgr = session.run([output_name], {input_name: image_bgr})[0]
predicted_mask_bgr = predicted_mask_bgr.squeeze()
binary_mask_bgr = (predicted_mask_bgr > 0.5).astype(np.uint8) * 255

# Run inference on RGB input
predicted_mask_rgb = session.run([output_name], {input_name: image_rgb})[0]
predicted_mask_rgb = predicted_mask_rgb.squeeze()
binary_mask_rgb = (predicted_mask_rgb > 0.5).astype(np.uint8) * 255

print("Raw predicted mask shape:", predicted_mask_bgr.shape)

# # Display both masks for comparison
# cv2.imshow("Predicted Mask (BGR Input)", binary_mask_bgr)
# cv2.imshow("Predicted Mask (RGB Input)", binary_mask_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Optional: Calculate difference between RGB and BGR masks
# difference = cv2.absdiff(binary_mask_bgr, binary_mask_rgb)
# cv2.imshow("Difference between BGR and RGB Masks", difference)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Optional: Calculate difference between RGB and BGR masks
# Optional: Calculate difference between RGB and BGR masks
# difference = cv2.absdiff(binary_mask_bgr, binary_mask_rgb)

# # Plot and save using matplotlib (WSL-safe)
# plt.figure(figsize=(12.8, 3.2))


# # plt.title("Predicted Mask (BGR Input)")
# plt.imshow(binary_mask_bgr, cmap='gray', interpolation='none')
# plt.axis('off')

# # plt.subplot(1, 3, 2)
# # plt.title("Predicted Mask (RGB Input)")
# # plt.imshow(binary_mask_rgb, cmap='gray')
# # plt.axis('off')

# # plt.subplot(1, 3, 3)
# # plt.title("Difference between BGR and RGB Masks")
# # plt.imshow(difference, cmap='hot')
# # plt.axis('off')

# plt.tight_layout(pad=0)

# # ✅ Save to file instead of showing (no GUI in WSL)
# plt.savefig("/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/SV_busbar_v2/Busbar_Al/images_inspection/Pinhole_오인식_Pass/Pinhole_오인식/crop_20250331_222801263_mask.png")
# print("Saved: onnx_mask_comparison.png")

original_image = cv2.imread(image_path)
original_image = cv2.resize(original_image, (imageWidth, imageHeight))
original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Find contours from the predicted RGB binary mask
contours, _ = cv2.findContours(binary_mask_bgr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw red contours on a copy of the original image
overlay_image = original_rgb.copy()
cv2.drawContours(overlay_image, contours, -1, (255, 0, 0), 2)  # Red in RGB

# # Plot and save
# plt.figure(figsize=(12.8, 3.2))
# plt.imshow(overlay_image)
# # plt.title("Original Image with Predicted Contours (Red)")
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.savefig("/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/SV_busbar_v2/Busbar_Al/images_inspection/Pinhole_오인식_Pass/Pinhole_오인식/crop_20250331_222801263_overlay.png")
# print("Saved: model_test_overlay.png")



# Create a vertical subplot: 2 rows, 1 column
fig, axs = plt.subplots(2, 1, figsize=(12.8, 6.4))  # height doubled for 2 images

# Top: binary mask
axs[0].imshow(binary_mask_bgr, cmap='gray', interpolation='none')
axs[0].axis('off')

# Bottom: overlay with contours
axs[1].imshow(overlay_image)
axs[1].axis('off')

# Remove padding and save
plt.tight_layout(pad=0)
save_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/SV_busbar_v2/Busbar_Al/images_inspection/Pinhole_오인식_Pass/Pinhole_오인식/crop_20250331_225718778_mask.png"
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
print("Saved:", save_path)
