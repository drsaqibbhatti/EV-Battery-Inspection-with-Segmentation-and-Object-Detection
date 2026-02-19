import os
import random
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

class ObjDatasetClamp(Dataset):
    def __init__(self, root, transform=None, img_size=(512, 512)):
        """
        Args:
            root (str): Path to the images directory.
            transform (callable, optional): Transformations to apply to images.
            useHFlip (bool): Enable horizontal flip augmentation.
            useVFlip (bool): Enable vertical flip augmentation.
            img_size (tuple): Target image size (height, width) for resizing.
        """
        self.image_files = [os.path.join(root, file) for file in os.listdir(root) if file.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform
        self.img_size = img_size  # Expected as (height, width)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        orig_width, orig_height = img.size

        # Resize image
        img_resized = img.resize(self.img_size, Image.NEAREST)

        # Apply transformations
        img_resized = self.transform(img_resized)
        img_resized = img_resized[[2, 1, 0], :, :]  # RGB to BGR
                
        # Return resized image and original image size
        return img_resized, img_path, (orig_width, orig_height)
