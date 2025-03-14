from torch.utils.data import DataLoader
from src.dataset import TraversablePathDataset
from utils.preprocess_data import ProcessData
from pathlib import Path
import torch

class TraversablePathDataloader:
    def __init__(self, raw_data_path, processed_data_path, batch_size=32, preprocess_data=False, shuffle=True, transform=None, num_workers=4):
        """
        Initializes the dataloader with the specified parameters.

        Args:
            raw_data_path (str): Path to raw data.
            processed_data_path (str): Path to processed data.
            batch_size (int): Number of samples per batch.
            preprocess_data (bool): Whether to preprocess data before loading.
            shuffle (bool): Whether to shuffle the training dataset.
            transform (callable, optional): Transformations for input images.
            num_workers (int): Number of subprocesses for data loading.
        """
        self.data_path = raw_data_path
        self.processed_path = processed_data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.num_workers = num_workers

        # Preprocess data if required
        if preprocess_data:
            self._run_preprocessing()

        # Ensure the processed path exists
        if not Path(self.processed_path).exists():
            raise FileNotFoundError(f"Processed data path does not exist: {self.processed_path}")

    def _run_preprocessing(self):
        """Runs data preprocessing pipeline."""
        preprocess_data = ProcessData(data_path=self.data_path, processed_path=self.processed_path)
        preprocess_data.run()

    def _create_dataloader(self, split, shuffle_override=None):
        """
        Creates a DataLoader for the given dataset split.

        Args:
            split (str): One of 'train', 'val', or 'test'.
            shuffle_override (bool, optional): Overrides default shuffle behavior.

        Returns:
            DataLoader: PyTorch DataLoader instance.
        """
        dataset = TraversablePathDataset(self.processed_path, split, self.transform)
        shuffle = self.shuffle if shuffle_override is None else shuffle_override
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def get_train_dataloader(self):
        return self._create_dataloader('train', shuffle_override=True)

    def get_validation_dataloader(self):
        return self._create_dataloader('val', shuffle_override=False)

    def get_test_dataloader(self):
        return self._create_dataloader('test', shuffle_override=False)
