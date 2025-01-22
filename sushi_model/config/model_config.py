from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ModelConfig:
    dim: int = 1024
    depth: int = 24
    heads: int = 8
    num_experts: int = 16
    expert_dim: int = 1024
    chunk_size: int = 128
    pre_rmsnorm: bool = True
    expert_paths: Optional[Dict[str, str]] = None