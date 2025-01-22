import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, List
from .logger import SushiLogger

class SushiDataset(Dataset):
    def __init__(self, data_dir: str):
        self.logger = SushiLogger()
        self.data_dir = data_dir
        self.data_files = []
        
        try:
            self._load_data_files()
        except Exception as e:
            self.logger.log_error(e, "dataset_initialization")
            raise
    
    def _load_data_files(self):
        """Load data files from the data directory"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.logger.log_model_state("data_loader", {"status": "created_data_directory"})
            
        # Load all .pt files from the data directory
        self.data_files = [
            f for f in os.listdir(self.data_dir) 
            if f.endswith('.pt')
        ]
        
        self.logger.log_model_state("data_loader", {
            "num_files": len(self.data_files),
            "data_dir": self.data_dir
        })
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            file_path = os.path.join(self.data_dir, self.data_files[idx])
            data = torch.load(file_path)
            
            # Assuming each file contains input and target tensors
            # Modify this according to your data format
            input_tensor = data['input']
            target_tensor = data['target']
            
            return input_tensor, target_tensor
            
        except Exception as e:
            self.logger.log_error(e, f"data_loading_{idx}")
            raise

def get_data_loader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader instance with the SushiDataset"""
    dataset = SushiDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )