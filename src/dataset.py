import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class TraversablePathDataset(Dataset):
    def __init__(self, data_path, split, transform=None):
        """
        Args:
            data_path (str): Root dataset directory.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Transformations for input images.
        """
        self.data_dir = os.path.join(data_path, split)
        self.transform = transform

        self.image_paths = []
        self.mask_paths = []

        for city in os.listdir(self.data_dir):
            city_path = os.path.join(self.data_dir, city)
            if os.path.isdir(city_path):
                for image_name in os.listdir(city_path):
                    image_folder = os.path.join(city_path, image_name)
                    if os.path.isdir(image_folder):
                        img_path = os.path.join(image_folder, f"{image_name}_leftImg8bit.png")
                        mask_path = os.path.join(image_folder, "binary_mask.png")
                        
                        if os.path.exists(img_path) and os.path.exists(mask_path):
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Ensure it's grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.tensor(np.array(mask, dtype=np.uint8), dtype=torch.long)

        return image, mask
