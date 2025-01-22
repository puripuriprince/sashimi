import torch
import os
from typing import Tuple

def generate_sample_data(
    num_samples: int = 100,
    seq_length: int = 1024,
    dim: int = 384,
    data_dir: str = "data/training"
) -> None:
    """Generate sample training data"""
    os.makedirs(data_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Generate random input sequence
        input_tensor = torch.randn(seq_length, dim)
        
        # Generate random target (modify this according to your needs)
        target_tensor = torch.randn(seq_length, dim)
        
        # Save the tensors
        torch.save({
            'input': input_tensor,
            'target': target_tensor
        }, os.path.join(data_dir, f'sample_{i}.pt'))

if __name__ == "__main__":
    generate_sample_data()