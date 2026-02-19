import torch
import numpy as np
import cv2


model_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/trainedModel/Phase_3/clamp/width/run_clamp_4/clamp_width_SegNextV2_2024-12-20_accuracy_train_0.99102_loss_0.00461_epoch_23_lr_0.003_time_1170.80M_alpha_0.7_beta_0.3_gamma_2_best.pth"
image_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Deep Images_20241210/dataset/clamp_deep_images_weld/dataset/boxes/train/images/2p4s_FT_Clamp.jpg"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()


image = cv2.imread(image_path)
image = cv2.resize(image, (640, 640))  
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0  
image = np.transpose(image, (2, 0, 1))  
image = np.expand_dims(image, axis=0) 
image = torch.from_numpy(image).to(device)


with torch.no_grad():
    output = model(image)


predicted_mask = output.squeeze().cpu().numpy()
binary_mask = (predicted_mask > 0.5).astype(np.uint8) 
visual_mask = binary_mask * 255 


cv2.imshow("Mask", visual_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
