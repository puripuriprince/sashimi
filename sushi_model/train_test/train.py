import torch
import torch.nn as nn
from utils.logger import SushiLogger
from utils.data_loader import get_data_loader
from utils.transformer_squared import TransformerLayerSVD as TransformerLayer
from config.model_config import ModelConfig

class SushiModel(nn.Module):
    def __init__(self, config):
        super(SushiModel, self).__init__()
        
        # Initialize logger
        self.logger = SushiLogger()
        
        # Model configuration
        self.config = config
        
        try:
            self._initialize_components()
        except Exception as e:
            self.logger.log_error("model_initialization", e)
            raise
            
    def _initialize_components(self):
        # Define layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.config) for _ in range(self.config.num_layers)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
        self.logger.log_model_state("initialization", {
            "num_layers": self.config.num_layers,
            "hidden_size": self.config.hidden_size
        })
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, x):
        try:
            # Forward pass implementation
            for layer in self.transformer_layers:
                x = layer(x)
            return x
        except Exception as e:
            self.logger.log_error("forward_pass", e)
            raise

def train(config, dataloader, criterion, num_epochs):
    logger = SushiLogger()
    
    try:
        # Initialize model and optimizer
        model = SushiModel(config)
        optimizer = torch.optim.AdamW(model.parameters())
        
        # Training loop with metadata logging
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                output = model(batch.input)
                loss = criterion(output, batch.target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Log epoch metrics
            logger.log_model_state(f"epoch_{epoch}", {
                "avg_loss": epoch_loss / num_batches,
                "num_batches": num_batches
            })
            
        return model
            
    except Exception as e:
        logger.log_error("training_loop", e)
        raise

def main():
    logger = SushiLogger()  # Initialize logger
    
    try:
        # Initialize config
        config = ModelConfig(
            dim=1024,
            depth=24,
            heads=8,
            num_experts=16
        )
        
        # Get data loader
        dataloader = get_data_loader(
            data_dir="data/training",
            batch_size=32,
            shuffle=True
        )
        
        # Initialize training components
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        model = train(config, dataloader, criterion, config.num_epochs)
        
    except Exception as e:
        logger.log_error("main", e)
        raise

if __name__ == "__main__":
    main()
