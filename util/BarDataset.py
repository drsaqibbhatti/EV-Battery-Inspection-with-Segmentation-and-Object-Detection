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

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class BarDataset(Dataset):
    def __init__(self, path="", transform=None, category="train", useHFlip=True, useVFlip=True, num_images=None):
        self.path = path
        self.transform = transform
        self.category = category
        self.useHFlip = useHFlip
        self.useVFlip = useVFlip
        self.num_images = num_images  # New argument to limit the number of images
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

                    regions = []
                    for region in annotations:

                        if region['className'] in ['bar'] and region['type'] in ['PolyGon', 'PolyLine']:
                            regions.append(region)
                        
                        
                    if len(regions) == 0:
                        continue
                        

                    imageFilePath = os.path.join(imageRootPath, imageFileName)
                    self.labelInfo.append([imageFilePath, regions])
                    
        if self.num_images is not None:
            self.labelInfo = self.labelInfo[:self.num_images]  # Limit the number of images if specified
        self.labelCount = len(self.labelInfo)
        
    def __len__(self):
        return self.labelCount
    
    def __getitem__(self, index):

        imagePath = self.labelInfo[index][0]
        regions = self.labelInfo[index][1]

        hFlip = False
        vFlip = False

        if self.useVFlip and random.random() > 0.5:
            vFlip = True

        if self.useHFlip and random.random() > 0.5:
            hFlip = True

        image = Image.open(imagePath).convert('RGB')
        # image= image[:, :, ::-1]
        original_width, original_height = image.size

        if hFlip:
            image = ImageOps.mirror(image)
        if vFlip:
            image = ImageOps.flip(image)



        
        cv_label_image_bar = np.zeros([original_height, original_width], dtype=np.uint8)


        for region in regions:
            polygon_points = []
            for point in region['points']:
                point_x = point[0] 
                point_y = point[1]
                polygon_points.append([point_x, point_y])
            
            pts = np.array(polygon_points, np.int32)
            
            # Get the strokeWidth for PolyLine (default to 1 if not specified)
            stroke_width = region.get('strokeWidth', 5)
            

            if region['className'] == 'bar':
                if region['type'] == 'PolyGon':
                    cv2.fillPoly(cv_label_image_bar, [pts], (255))
                elif region['type'] == 'PolyLine':
                    cv2.polylines(cv_label_image_bar, [pts], isClosed=True, color=(255), thickness=stroke_width)

        if vFlip:
            cv_label_image_bar = cv2.flip(cv_label_image_bar, 0)
        if hFlip:
            cv_label_image_bar = cv2.flip(cv_label_image_bar, 1)

  
        resize_height, resize_width = self.transform.transforms[0].size  # Extract resize dimensions directly

        # Now resize both the image and the masks
        image = image.resize((resize_width, resize_height), Image.BILINEAR)
        

        # Resize masks
        cv_label_image_bar = cv2.resize(cv_label_image_bar, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)
        


        # Convert the image to tensor
        torch_image = self.transform(image)
    

        ################ BGR #########################
        torch_image = torch_image[[2, 1, 0], :, :]
        ##############################################
        
        torch_label_image_bar = torch.tensor(cv_label_image_bar, dtype=torch.float32, device=device) / 255


        torch_label = torch.zeros([1, resize_height, resize_width], dtype=torch.float32, device=device).detach()
        torch_label[0] = torch_label_image_bar
        

        
        return torch_image, torch_label
