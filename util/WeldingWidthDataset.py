
import torch
import numpy as np
import os
import json
import random
import cv2


from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageOps
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

USE_CUDA = torch.cuda.is_available() # Returns True if the GPU is available, False otherwise
device = torch.device("cuda" if USE_CUDA else "cpu") # Use GPU if available, otherwise use CPU


class WeldingWidthDataset(Dataset):
    def __init__(self,  path="", transform=None, category="train", useHFlip=True, useVFlip=True, num_images=None):
        self.path = path
        self.transform = transform
        self.category = category
        self.useHFlip = useHFlip
        self.useVFlip = useVFlip
        self.num_images=num_images
        

        labelRootPath = os.path.join(self.path, "labels")
        imageRootPath = os.path.join(self.path, "images")
        self.labelInfo = []
        for file in os.listdir(labelRootPath):
            if file.endswith(".json"):
                jsonFilePath = os.path.join(labelRootPath, file)
                with open(jsonFilePath, 'r') as jsonFile:
                    jsonData = json.load(jsonFile)
                for label in jsonData['data']:
                    imageFileName = label['fileName']
                    annotations = label['regionLabel']
                    category = label['set']

                    if self.category != category: 
                        continue

                    if len(annotations) == 0:
                        continue

                    polygons = []
                    for region in annotations:
                        if region['className'] == 'weld' and region['type'] == 'PolyGon':
                            polygons.append(region['points'])

                    imageFilePath = os.path.join(imageRootPath, imageFileName)
                    self.labelInfo.append([imageFilePath, polygons])
                    
        if self.num_images is not None:
            self.labelInfo = self.labelInfo[:self.num_images]  # Limit the number of images if specified
        self.labelCount = len(self.labelInfo)
                

        
    def __len__(self):
        return self.labelCount
    
    def __getitem__(self, index):


        imagePath = self.labelInfo[index][0]
        polygons = self.labelInfo[index][1]

        hFlip = False
        vFlip = False

        if self.useVFlip == True and random.random() > 0.5:
            vFlip = True

        if self.useHFlip == True and random.random() > 0.5:
            hFlip = True


         
        image = Image.open(imagePath).convert('RGB')
        # image= image[:, :, ::-1]
        original_width, original_height = image.size

        if hFlip == True:
            image = ImageOps.mirror(image)

        if vFlip == True:
            image = ImageOps.flip(image)



        torch_image = self.transform(image)
        ############################################
        torch_image = torch_image[[2, 1, 0], :, :]
        #############################################
        resize_height = torch_image.shape[1]
        resize_width = torch_image.shape[2]


        torch_label = torch.zeros([1, resize_height, resize_width], dtype=torch.float32, device=device).detach()
        cv_label_image = np.zeros([resize_height, resize_width], dtype=np.uint8)
        for polygon in polygons:

            polygon_points = []
            for point in polygon:
                point_x = point[0] / original_width * resize_width
                point_y = point[1] / original_height * resize_height
                polygon_points.append([point_x, point_y])
            pts = np.array(polygon_points, np.int32)
            cv2.fillPoly(cv_label_image, [pts], (255))


        if vFlip == True:
            cv_label_image = cv2.flip(cv_label_image, 0) 

        if hFlip == True:
            cv_label_image = cv2.flip(cv_label_image, 1) 


        torch_label_image = torch.tensor(cv_label_image, dtype=torch.float32, device=device) / 255
        torch_label[0] = torch_label_image


        return torch_image, torch_label
