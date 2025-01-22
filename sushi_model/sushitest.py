import torch
from utils.transformer_squared import TransformerSquared
import torch
from titans_pytorch import NeuralMemory
# from t2 import Transformer2
# from Papers.blt.bytelatent.tokenizers.byte_tokenizer import ByteTokenizer
from config.model_config import ModelConfig
from utils.logger import SushiLogger
from typing import Optional
from utils.data_loader import get_data_loader
from utils.generate_sample_data import generate_sample_data
import os

class SushiModel:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.logger = SushiLogger()
        self.config = config or ModelConfig()
        
        try:
            self._initialize_components()
        except Exception as e:
            self.logger.log_error(e, "model_initialization")
            raise
    
    def _initialize_components(self):
        # Initialize neural memory
        self.mem = NeuralMemory(
            dim=self.config.dim,
            chunk_size=self.config.chunk_size,
            pre_rmsnorm=self.config.pre_rmsnorm
        ).cuda()
        
        # Initialize Transformer² model
        self.model = TransformerSquared(
            dim=self.config.dim,
            num_layers=self.config.depth,
            num_heads=self.config.heads,
            num_trainable_experts=self.config.num_experts,
            expert_paths=self.config.expert_paths,
        ).cuda()
        
        self.logger.log_model_state("initialization", {
            "dim": self.config.dim,
            "num_experts": self.config.num_experts,
            "device": "cuda"
        })
    
    def process_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        try:
            # Process input through memory
            retrieved = self.mem(seq)
            
            # Pass through Transformer² with memory-enhanced features
            output = self.model(retrieved)
            
            self.logger.log_model_state("process_sequence", {
                "input_shape": seq.shape,
                "output_shape": output.shape
            })
            
            return output
            
        except Exception as e:
            self.logger.log_error(e, "process_sequence")
            raise

def main():
    try:
        # Generate sample data if it doesn't exist
        data_dir = "data/training"
        if not os.path.exists(data_dir):
            generate_sample_data(
                num_samples=100,
                seq_length=1024,
                dim=384,
                data_dir=data_dir
            )
        
        # Get data loader
        train_loader = get_data_loader(
            data_dir=data_dir,
            batch_size=32,
            shuffle=True
        )
        
        # Initialize model
        model = SushiModel()
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters())
        
        # Train the model
        model.train_experts(train_loader, optimizer)
        
        # Test with a single batch
        for batch, (data, target) in enumerate(train_loader):
            output = model.process_sequence(data.cuda())
            assert output.shape == data.shape, "Output shape should match input shape"
            break
            
    except Exception as e:
        logger = SushiLogger()
        logger.log_error(e, "main")
        raise

if __name__ == "__main__":
    main()

# Initialize with both trainable and pre-trained experts




