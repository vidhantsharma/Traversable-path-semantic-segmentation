import os
import shutil

class CreateFolderStructure():
    def __init__(self, raw_dataset_path, processed_dataset_path):
        self.raw_dataset_path = raw_dataset_path
        self.processed_dataset_path = processed_dataset_path
    def create_folders_and_move_files(self):
        for split in ['train', 'val', 'test']:
            for dataset_type in ['leftImg8bit', 'gtFine']:
                split_path = os.path.join(self.raw_dataset_path, dataset_type, split)
                if not os.path.exists(split_path):
                    continue
                
                for city in os.listdir(split_path):
                    city_path = os.path.join(split_path, city)
                    if not os.path.isdir(city_path):
                        continue
                    
                    for file in os.listdir(city_path):
                        if file.endswith('.png'):
                            file_name = os.path.splitext(file)[0]
                            file_name_parts = file_name.split('_')
                            file_name = '_'.join(file_name_parts[:3])
                            dest_path = os.path.join(self.processed_dataset_path, split, city, file_name)
                            os.makedirs(dest_path, exist_ok=True)
                            
                            shutil.copy(os.path.join(city_path, file), os.path.join(dest_path, file))
                            print(f"Moved: {file} -> {dest_path}")

if __name__ == "__main__":
    source_directory = r"raw_dataset"
    destination_directory = r"processed_dataset"
    
    create_foldeer_structure = CreateFolderStructure(source_directory, destination_directory)
    create_foldeer_structure.create_folders_and_move_files()
