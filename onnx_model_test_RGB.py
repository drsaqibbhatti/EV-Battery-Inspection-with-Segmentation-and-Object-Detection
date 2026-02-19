import onnxruntime as ort
import numpy as np
import cv2

imageHeight = 640
imageWidth = 640

onnx_model_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/trainedModel/Phase_3/clamp/width/Final_model/model_clamp_weld_SegN.onnx"
session = ort.InferenceSession(onnx_model_path)


image_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/Sebang_Latest_Dec24/Clamp Deep Images_20241210/dataset/clamp_deep_images_weld_segregated/images/2p4s_FT_Clamp (1).jpg"
image = cv2.imread(image_path)  
image = cv2.resize(image, (imageWidth, imageHeight)) 
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
image = image.astype(np.float32) / 255.0  
image = np.transpose(image, (2, 0, 1)) 
image = np.expand_dims(image, axis=0)  


input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
predicted_mask = session.run([output_name], {input_name: image})[0]


predicted_mask = predicted_mask.squeeze() 
binary_mask = (predicted_mask > 0.5).astype(np.uint8)  
visual_mask = binary_mask * 255  

cv2.imshow("Mask", visual_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

