import os
import numpy as np
import cv2
from pathlib import Path
from torchvision import transforms
from PIL import Image

class ProcessData():
    def __init__(self, data_path):
        '''
        Process the data
        '''
        self.data_path = Path(data_path)
        self.gt_path = self.data_path / "gtFine"
        self.img_path = self.data_path / "leftImg8bit"
        self.processed_path = self.data_path / "processed"
        self.processed_path.mkdir(exist_ok=True)
    
    def retrieve_data(self):
        '''
        Retrieves data and stores it in a structured folder structure.
        '''
        cities = [d for d in self.img_path.iterdir() if d.is_dir()]
        for city in cities:
            (self.processed_path / city.name).mkdir(exist_ok=True)
            for img_file in (city.glob("*.png")):
                gt_file = self.gt_path / city.name / img_file.name.replace("leftImg8bit", "gtFine_labelIds")
                if gt_file.exists():
                    print(f"Found: {img_file.name} and {gt_file.name}")
                else:
                    print(f"Missing ground truth for {img_file.name}")
    
    def create_binary_ground_truth(self):
        '''
        Convert the multi-labeled pixel ground truth to binary (traversable/non-traversable).
        Stores in the processed folder structure.
        '''
        for city in self.gt_path.iterdir():
            if not city.is_dir():
                continue
            
            save_dir = self.processed_path / city.name
            save_dir.mkdir(exist_ok=True)
            
            for gt_file in city.glob("*_labelIds.png"):
                gt = cv2.imread(str(gt_file), cv2.IMREAD_UNCHANGED)
                
                # Define traversable class IDs (e.g., roads, sidewalks)
                traversable_classes = {7, 8, 11}  # Adjust based on Cityscapes class IDs
                
                # Convert to binary mask (1 for traversable, 0 for non-traversable)
                binary_gt = np.isin(gt, list(traversable_classes)).astype(np.uint8) * 255
                
                save_path = save_dir / gt_file.name.replace("labelIds", "binary")
                cv2.imwrite(str(save_path), binary_gt)
                print(f"Saved binary ground truth: {save_path}")
    
    def augment_data(self):
        '''
        Create data augmentation functions.
        '''
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
        
        for city in self.processed_path.iterdir():
            if not city.is_dir():
                continue
            
            for img_file in city.glob("*_leftImg8bit.png"):
                image = Image.open(img_file)
                aug_image = augmentation(image)
                
                save_path = img_file.with_name(img_file.stem + "_aug.png")
                transforms.ToPILImage()(aug_image).save(save_path)
                print(f"Saved augmented image: {save_path}")

if __name__ == "__main__":
    data_processor = ProcessData("/path/to/cityscapes")  # Update with actual dataset path
    data_processor.retrieve_data()
    data_processor.create_binary_ground_truth()
    data_processor.augment_data()
