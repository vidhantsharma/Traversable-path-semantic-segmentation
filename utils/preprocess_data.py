import os
import numpy as np
import cv2
from pathlib import Path
from torchvision import transforms
from PIL import Image

from utils.create_folder_structure import CreateFolderStructure
from utils.create_binary_gt import CreateBinaryGT

class ProcessData():
    def __init__(self, data_path, processed_path = "processed_path"):
        '''
        Process the data
        '''
        self.raw_data_path = data_path
        self.processed_path = processed_path
        os.makedirs(self.processed_path, exist_ok=True)
    
    def retrieve_data(self):
        '''
        Retrieves data and stores it in a structured folder structure.
        '''
        create_folder_structure = CreateFolderStructure(raw_dataset_path=self.raw_data_path, processed_dataset_path=self.processed_path)
        create_folder_structure.create_folders_and_move_files()

    def create_binary_ground_truth(self):
        '''
        Convert the multi-labeled pixel ground truth to binary (traversable/non-traversable).
        Stores in the processed folder structure.
        '''
        create_binary_mask = CreateBinaryGT(gt_path=self.processed_path, output_path=self.processed_path)  

        create_binary_mask.process_cityscapes_gt()  
        
    def augment_data(self):
        '''
        Create data augmentation functions.
        '''
        # TODO : varshitha
        pass
        
    def run(self,retrieve_data = True, create_binary_mask = True, augument_data=False):
        if retrieve_data:
            self.retrieve_data()
        try:
            if create_binary_mask:
                self.create_binary_ground_truth()
        except Exception as ex:
            raise Exception(f"Error while creating binary ground truth: {str(ex)}")
        if augument_data:
            self.augment_data()

# if __name__ == "__main__":
#     data_path = "raw_dataset"
#     processed_path = "processed_dataset"
#     data_processor = ProcessData(data_path=data_path, processed_path=processed_path)
#     data_processor.run(retrieve_data = True, create_binary_mask = True, augument_data = False)

