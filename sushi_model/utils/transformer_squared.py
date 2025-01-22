import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

import math
import logging
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from config.model_config import ModelConfig
from torch.utils.data import DataLoader
import os

logger = logging.getLogger(__name__)

@dataclass
class TransformerStats:
    """Track transformer performance metrics"""
    layer_times: List[float] = None
    attention_patterns: List[torch.Tensor] = None
    memory_per_layer: List[int] = None
    gradient_norms: List[float] = None
    peak_memory: float = 0.0
    task_classification: str = None  # Added for task classification
    expert_selection: Dict[str, float] = None  # Added for expert weights

class ExpertModule(nn.Module):
    """Individual expert module that can be trained"""
    def __init__(self, dim: int, rank: int = 4):
        super().__init__()
        self.dim = dim
        self.rank = rank
        # Initialize SVD adapter for this expert
        self.svd_adapter = SVDAdapter(
            dim=dim,
            rank=rank
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply expert's SVD adaptation
        return self.svd_adapter(x)

class ExpertGate(nn.Module):
    """Gate network to select experts"""
    def __init__(self, 
                 input_dim: int,
                 num_experts: int,
                 top_k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get gate logits
        gate_logits = self.gate(x)
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        gate_weights = F.softmax(top_k_logits, dim=-1)
        
        return gate_weights, top_k_indices

class TransformerSquared(nn.Module):
    """
    Implementation of Transformer² (Transformer Squared) from the paper:
    'TRANSFORMER2: SELF-ADAPTIVE LLMS'
    
    Optimized version with:
    - Flash attention support
    - Memory efficient attention
    - Gradient checkpointing
    - Performance monitoring
    - SVD-based adaptation
    """
    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 8192,
        num_trainable_experts: int = 4,  # Number of trainable experts
        expert_paths: Optional[Dict[str, str]] = None,  # Paths to pre-trained experts
        top_k: int = 2,
        use_gradient_checkpointing: bool = True,
        use_flash_attention: bool = True,
        memory_efficient: bool = True,
    ):
        super().__init__()
        logger.info(f"Initializing TransformerSquared with dim={dim}, layers={num_layers}")
        
        # Extract dim value if ModelConfig is passed
        dim_value = dim.dim if isinstance(dim, ModelConfig) else dim
        
        # Initialize trainable experts
        self.trainable_experts = nn.ModuleList([
            ExpertModule(dim=dim_value) 
            for _ in range(num_trainable_experts)
        ])
        
        # Load pre-trained experts
        self.pretrained_experts = self._load_experts(expert_paths) if expert_paths else {}
        
        # Total number of experts (trainable + pre-trained)
        total_experts = num_trainable_experts + len(self.pretrained_experts)
        logger.info(f"Total experts: {total_experts} ({num_trainable_experts} trainable, {len(self.pretrained_experts)} pre-trained)")
        
        # Initialize expert gate
        self.expert_gate = ExpertGate(
            input_dim=dim_value,
            num_experts=total_experts,
            top_k=top_k
        )
        
        # Initialize SVD components for each layer
        self.layers = nn.ModuleList([
            TransformerLayerSVD(
                dim=dim_value,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_flash_attention = use_flash_attention
        self.memory_efficient = memory_efficient
        self.norm = nn.LayerNorm(dim_value)
        
        # Performance tracking
        self.stats = TransformerStats()
        self._initialize_stats()
        
        # Expert training metrics
        self.expert_usage = torch.zeros(total_experts)
        self.expert_load = torch.zeros(total_experts)
        
        # Map expert indices to names for logging
        self.expert_index_to_name = {}
        for i in range(num_trainable_experts):
            self.expert_index_to_name[i] = f"trainable_expert_{i}"
        for i, name in enumerate(self.pretrained_experts.keys()):
            self.expert_index_to_name[num_trainable_experts + i] = name

    def _load_experts(self, expert_paths: Dict[str, str]) -> Dict[str, ExpertModule]:
        """Load pre-trained expert modules"""
        experts = {}
        for name, path in expert_paths.items():
            try:
                state_dict = torch.load(path)
                expert = ExpertModule(dim=self.dim)
                expert.load_state_dict(state_dict)
                expert.requires_grad_(False)  # Freeze pre-trained experts
                experts[name] = expert
                logger.info(f"Loaded pre-trained expert: {name}")
            except Exception as e:
                logger.error(f"Failed to load expert {name}: {str(e)}")
        return experts

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with expert selection and application"""
        batch_size = x.shape[0]
        
        try:
            # Get expert weights and indices from gate
            gate_weights, expert_indices = self.expert_gate(x[:, 0])  # Use [CLS] token
            
            # Track expert usage
            self._update_expert_stats(expert_indices)
            
            # Apply selected experts
            expert_outputs = []
            for i in range(self.expert_gate.top_k):
                weights = gate_weights[:, i].unsqueeze(-1)
                indices = expert_indices[:, i]
                
                expert_out = torch.zeros_like(x)
                for b in range(batch_size):
                    expert_idx = indices[b].item()
                    # Route to appropriate expert (trainable or pre-trained)
                    if expert_idx < len(self.trainable_experts):
                        expert_out[b] = self.trainable_experts[expert_idx](x[b])
                    else:
                        pretrained_idx = expert_idx - len(self.trainable_experts)
                        expert_name = list(self.pretrained_experts.keys())[pretrained_idx]
                        expert_out[b] = self.pretrained_experts[expert_name](x[b])
                
                expert_outputs.append(weights * expert_out)
            
            # Combine expert outputs
            combined_output = sum(expert_outputs)
            
            # Regular transformer forward pass with expert-adapted input
            for layer in self.layers:
                combined_output = layer(combined_output)[0]
            
            combined_output = self.norm(combined_output)
            
            return {
                'hidden_states': combined_output,
                'expert_weights': gate_weights,
                'expert_indices': expert_indices
            }
            
        except Exception as e:
            logger.error(f"Error in transformer forward pass: {str(e)}")
            raise

    def _update_expert_stats(self, expert_indices: torch.Tensor):
        """Track expert usage statistics"""
        # Update usage count
        for indices in expert_indices:
            self.expert_usage[indices] += 1
            
        # Update load balancing stats
        total_calls = self.expert_usage.sum()
        if total_calls > 0:
            self.expert_load = self.expert_usage / total_calls

    def train_experts(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10
    ) -> None:
        """Train the experts using the provided data loader"""
        self.train()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                
                optimizer.zero_grad()
                output = self(data)
                loss = nn.functional.mse_loss(output, target)
                
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0:
                    self.logger.log_model_state("training", {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss": loss.item()
                    })

    def _compute_expert_loss(self, 
                           outputs: Dict[str, torch.Tensor],
                           labels: torch.Tensor,
                           expert_load: torch.Tensor,
                           load_balance_coef: float = 0.01) -> torch.Tensor:
        """Compute loss with expert load balancing"""
        # Task-specific loss
        task_loss = F.cross_entropy(outputs['hidden_states'], labels)
        
        # Expert load balancing loss
        desired_load = 1.0 / len(self.experts)
        load_balance_loss = ((expert_load - desired_load) ** 2).mean()
        
        # Combined loss
        total_loss = task_loss + load_balance_coef * load_balance_loss
        
        return total_loss

    def _log_expert_stats(self):
        """Log expert usage statistics with names"""
        for i, (usage, load) in enumerate(zip(self.expert_usage, self.expert_load)):
            expert_name = self.expert_index_to_name[i]
            logger.info(f"Expert {expert_name}: Usage = {usage}, Load = {load:.4f}")

    def _initialize_stats(self):
        """Initialize performance tracking stats"""
        self.stats.layer_times = []
        self.stats.attention_patterns = []
        self.stats.memory_per_layer = []
        self.stats.gradient_norms = []

    def _track_gradient_norms(self):
        """Track gradient norms during training"""
        with torch.no_grad():
            grad_norms = []
            for layer in self.layers:
                layer_norm = 0
                for p in layer.parameters():
                    if p.grad is not None:
                        layer_norm += p.grad.norm().item()
                grad_norms.append(layer_norm)
            self.stats.gradient_norms = grad_norms

    def save_expert(self, expert_idx: int, path: str):
        """Save a trained expert"""
        if expert_idx >= len(self.trainable_experts):
            raise ValueError(f"Invalid expert index {expert_idx}")
        
        try:
            torch.save(
                self.trainable_experts[expert_idx].state_dict(),
                path
            )
            logger.info(f"Saved expert {expert_idx} to {path}")
        except Exception as e:
            logger.error(f"Failed to save expert {expert_idx}: {str(e)}")

class SVDAdapter(nn.Module):
    """
    Maintains SVD-based scaling for a weight matrix W = U S V^T.
    Freezes U and V, but learns z separately. Then scaled_sigma = sigma * z.
    """
    def __init__(self, W: torch.Tensor, rank: int, init_std: float = 0.01):
        super().__init__()
        # Perform offline SVD once
        U_, S_, Vt_ = torch.linalg.svd(W, full_matrices=False)
        # Keep top `rank` components
        r = min(rank, S_.numel())
        self.U = nn.Parameter(U_[:, :r], requires_grad=False)
        self.Vt = nn.Parameter(Vt_[:r, :], requires_grad=False)
        self.sigma_base = nn.Parameter(S_[:r], requires_grad=False)
        
        # Learnable vector z of shape [r], init near 1
        self.z = nn.Parameter(torch.ones(r) + init_std * torch.randn(r))

    def forward(self):
        """Recompute adapted weight = U diag(sigma_base * z) V^T"""
        scaled_sigma = self.sigma_base * self.z
        # [out_dim, r] x diag(r) x [r, in_dim]
        W_approx = (self.U * scaled_sigma.unsqueeze(0)) @ self.Vt
        return W_approx

class TransformerLayerSVD(nn.Module):
    """Transformer layer with SVD-based adaptation"""
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 rank: int = 4):
        super().__init__()
        self.rank = rank
        
        # Initialize weight matrices
        self.qkv_weight = nn.Parameter(torch.randn(dim, 3 * dim) / math.sqrt(dim))
        
        # SVD adapter for attention weights
        self.svd_adapter = SVDAdapter(self.qkv_weight, rank=rank)
        
        # Attention with SVD adaptation
        self.attention = SVDAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            rank=rank
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, 
                x: torch.Tensor,
                z_vector: Optional[torch.Tensor] = None) -> tuple:
        # Store original dimensions
        batch_size, seq_len, _ = x.shape
        
        # Attention with SVD adaptation
        attn_out, attn_maps = self.attention(self.norm1(x), z_vector=z_vector)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        # Ensure output maintains original dimensions
        if x.size(0) != batch_size or x.size(1) != seq_len:
            x = x.reshape(batch_size, seq_len, -1)
            
        # Final shape verification
        assert x.shape[0] == batch_size, f"Expected batch size {batch_size}, got {x.shape[0]}"
        assert x.shape[1] == seq_len, f"Expected seq length {seq_len}, got {x.shape[1]}"
        
        return x, attn_maps
        
    def apply_singular_scaling(self, z_vector: Optional[torch.Tensor]):
        """Apply SVD-based scaling from z_vector"""
        if z_vector is not None:
            # Scale singular values with z_vector
            scaled_sigma = self.sigma * z_vector
            # Update weight matrix W = U * diag(scaled_sigma) * V
            W = torch.mm(self.U * scaled_sigma.unsqueeze(0), self.V)
            # Apply scaled weights to attention
            self.attention.update_weights(W)
        # Attention with SVD adaptation
        attn_out, attn_maps = self.attention(self.norm1(x), z_vector=z_vector)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_maps

class SVDAttention(nn.Module):
    def update_weights(self, W: torch.Tensor):
        """Update weights with SVD-adapted matrix"""
        # Update projection weights with new SVD-scaled matrix
        with torch.no_grad():
            # Split W for differential attention
            W1, W2 = W.chunk(2, dim=-1)
            self.to_qkv1.weight.copy_(W1)
            self.to_qkv2.weight.copy_(W2)
    """SVD-Enhanced Differential Attention"""
    def __init__(self, dim, num_heads=8, rank=4, sigma=0.1, dropout=0.1):
        super().__init__()
        # Extract dim value if ModelConfig is passed
        dim_value = dim.dim if isinstance(dim, ModelConfig) else dim
        self.dim = dim_value
        self.num_heads = num_heads
        assert dim_value % num_heads == 0, f"dim {dim_value} must be divisible by num_heads {num_heads}"
        self.d_head = dim_value // num_heads
        self.rank = rank
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.scale = 1 / math.sqrt(self.d_head)
        
        # SVD adaptation parameters
        self.U = nn.Parameter(torch.randn(dim_value, rank))
        self.V = nn.Parameter(torch.randn(rank, dim_value))
        
        # Projection layers for differential attention
        self.to_qkv1 = nn.Linear(dim_value, dim_value * 3, bias=False)
        self.to_qkv2 = nn.Linear(dim_value, dim_value * 3, bias=False)
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.to_qkv1.weight)
        nn.init.xavier_uniform_(self.to_qkv2.weight)
        
        # Output projection with LayerNorm instead of GroupNorm
        # This avoids issues with group size divisibility
        self.to_out = nn.Sequential(
            nn.Linear(dim_value, dim_value),
            nn.LayerNorm(dim_value)
        )
        
        # Initialize projection layers
        self.proj1 = nn.Linear(dim_value // 2, dim_value)
        self.proj2 = nn.Linear(dim_value // 2, dim_value)
        
        # Initialize projection weights
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.proj2.weight)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, z_vector=None):
        """Forward pass with SVD adaptation and differential attention"""
        # Store original dimensions
        batch_size, seq_len, _ = x.shape
        
        # SVD adaptation
        svd_adapt = self.sigma * (self.U @ self.V)  # Shape: [dim, rank] @ [rank, dim] -> [dim, dim]
        if z_vector is not None:
            # Ensure z_vector matches rank dimension for SVD adaptation
            z_vector = z_vector[:self.rank] if z_vector.size(0) > self.rank else z_vector
            z_vector = z_vector.reshape(1, -1)  # [1, rank] for correct broadcasting
            scaled_sigma = self.sigma * z_vector  # [rank] * [1, rank] -> [1, rank]
            svd_adapt = self.U @ (scaled_sigma.unsqueeze(-1) * self.V)  # Proper matrix multiplication
            
        # Project to input dimension
        svd_adapt = svd_adapt[:self.dim].sum(dim=-1)  # Shape: [dim]
        
        # Reshape svd_adapt to match input shape [batch_size, seq_len, dim]
        svd_adapt = svd_adapt.reshape(1, 1, self.dim)
        svd_adapt = svd_adapt.expand(batch_size, seq_len, self.dim)
            
        # Add SVD adaptation
        adapted_x = x + svd_adapt
        
        # Split input for differential attention
        x1, x2 = torch.chunk(adapted_x, 2, dim=-1)  # Each has shape [batch_size, seq_len, dim//2]
        
        # Project split inputs to full dimension
        x1 = self.proj1(x1)  # Shape: [batch_size, seq_len, dim]
        x2 = self.proj2(x2)  # Shape: [batch_size, seq_len, dim]
        
        # Get QKV pairs
        qkv1 = self.to_qkv1(x1)  # Shape: [batch_size, seq_len, 3*dim]
        qkv2 = self.to_qkv2(x2)  # Shape: [batch_size, seq_len, 3*dim]
        
        # Split into Q, K, V
        q1, k1, v1 = qkv1.chunk(3, dim=-1)
        q2, k2, v2 = qkv2.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q1 = q1.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k1 = k1.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v1 = v1.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        q2 = q2.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k2 = k2.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v2 = v2.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # Differential attention
        out = self.differential_attention(q1, k1, v1, q2, k2, v2)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Output projection with LayerNorm
        out = self.to_out(out)
        
        # Ensure output shape matches input shape
        if out.size(0) != batch_size or out.size(1) != seq_len:
            out = out.view(batch_size, seq_len, -1)
            
        return out, None
        
    def differential_attention(self, q1, k1, v1, q2, k2, v2):
        """
        Compute differential attention as described in the paper.
        Subtracts second attention map from first.
        """
        # Store original batch size
        batch_size = q1.size(0)
        
        # Compute attention scores
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)
        
        # Apply dropout
        attn1 = self.dropout(attn1)
        attn2 = self.dropout(attn2)
        
        # Compute differential attention
        diff_attn = attn1 - attn2
        
        # Apply attention to values
        out1 = torch.matmul(diff_attn, v1)
        out2 = torch.matmul(diff_attn, v2)
        
        # Average the outputs and ensure batch dimension is preserved
        out = (out1 + out2) / 2
        if out.size(0) != batch_size:
            out = out.view(batch_size, -1, out.size(-1))
            
        return out


# dont read this just ignore it
#TransformerSquared(
#    dim=768,
#    num_trainable_experts=4,
#    expert_paths={
#        'math_expert': 'path/to/math_expert.pt',
#        'code_expert': 'path/to/code_expert.pt'
#    }
#)

# Train only the trainable experts
# optimizer = torch.optim.AdamW(model.trainable_experts.parameters())
# model.train_experts(train_loader, optimizer)

# Save a trained expert
#model.save_expert(expert_idx=0, path='path/to/save/new_expert.pt')

# Forward pass will use both trainable and pre-trained experts