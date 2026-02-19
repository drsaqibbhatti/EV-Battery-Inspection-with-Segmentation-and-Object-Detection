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

class ClampWidthDataset(Dataset):
    def __init__(self, path="", transform=None, useHFlip=True, useVFlip=True, num_images=None):

        self.path = path
        self.transform = transform
        self.useHFlip = useHFlip
        self.useVFlip = useVFlip
        self.num_images = num_images

        imageRootPath = os.path.join(self.path, "images")
        labelRootPath = os.path.join(self.path, "labels")

        self.labelInfo = []
        for image_file in os.listdir(imageRootPath):
            if not image_file.endswith((".jpg")):
                continue

            json_file = os.path.join(labelRootPath, os.path.splitext(image_file)[0] + ".json")
            if not os.path.exists(json_file):
                continue

            with open(json_file, 'r') as jsonFile:
                jsonData = json.load(jsonFile)

            # Collect polygons from the JSON file
            polygons = [entry['Points'] for entry in jsonData if entry['Name'] == "weld"]

            if not polygons:
                continue

            imageFilePath = os.path.join(imageRootPath, image_file)
            self.labelInfo.append([imageFilePath, polygons])

        # Limit the number of images if specified
        if self.num_images is not None:
            self.labelInfo = self.labelInfo[:self.num_images]

        self.labelCount = len(self.labelInfo)

    def __len__(self):
        return self.labelCount

    def __getitem__(self, index):
        imagePath, polygons = self.labelInfo[index]

        hFlip = False
        vFlip = False

        if self.useVFlip and random.random() > 0.5:
            vFlip = True

        if self.useHFlip and random.random() > 0.5:
            hFlip = True

        # Load and transform image
        image = Image.open(imagePath).convert('RGB')
        original_width, original_height = image.size

        if hFlip:
            image = ImageOps.mirror(image)

        if vFlip:
            image = ImageOps.flip(image)

        torch_image = self.transform(image)
        torch_image = torch_image[[2, 1, 0], :, :]  # RGB to BGR (if needed)

        resize_height, resize_width = torch_image.shape[1], torch_image.shape[2]

        # Create label mask
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

        if vFlip:
            cv_label_image = cv2.flip(cv_label_image, 0)

        if hFlip:
            cv_label_image = cv2.flip(cv_label_image, 1)

        torch_label_image = torch.tensor(cv_label_image, dtype=torch.float32, device=device) / 255
        torch_label[0] = torch_label_image

        return torch_image, torch_label
