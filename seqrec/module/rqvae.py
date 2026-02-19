import torch
import torch.nn as nn
from typing import List, Optional

from dataclasses import dataclass

from dataclasses import dataclass
from transformers.utils import ModelOutput

from .utils import kmeans_init_
from .mlp import MLP

@dataclass
class QuantizerOutput:
    quantized: torch.Tensor  # (Batch, EmbeddingDim)
    indices: torch.LongTensor  # (Batch,)
    loss: torch.Tensor

class Quantizer(nn.Module):
    def __init__(self, code_size: int, embedding_dim: int, beta: float):
        super().__init__()
        self.codes = nn.Parameter(torch.zeros(1, code_size, embedding_dim))
        self.alpha = 0.1
        self.beta = beta
        self.register_buffer('initialized', torch.tensor(0))

    def init_codebooks(self, data: torch.Tensor):
        """Initialize codebook entries using k-means on the provided data."""
        if self.initialized:
            raise RuntimeError("CodeBook has already been initialized.")
        if data.dim() != 2 or data.size(1) != self.codes.size(2):
            raise ValueError(f"Data must be of shape (Batch, EmbeddingDim={self.codes.size(2)}), but got {data.shape}")
        
        with torch.no_grad():
            kmeans_init_(self.codes.data.squeeze(0), data)
            self.initialized.fill_(1)

    def forward(self, x: torch.Tensor, temperature: float):
        """
        Args:
            x: (Batch, EmbeddingDim)
        Returns:
            quantized: (Batch, EmbeddingDim)
            indices: (Batch,) - Codebook indices
        """

        if self.training and not self.initialized:
            # Explicitly initialize codebook with data samples during the first forward pass in training mode
            raise RuntimeError("CodeBook must be initialized with data before training. Call codebook.initialize(data) first.")

        # Compute distances to codebook entries
        # x: (B, D), codes: (1, C, D) -> distances: (B, C)
        distances = torch.norm(x.unsqueeze(1) - self.codes, dim=-1) / (x.size(1) ** 0.5) / self.alpha # (B, C)

        if self.training:
            # Gummbel Softmax for differentiable codebook selection
            weights = torch.nn.functional.gumbel_softmax(-distances, tau=temperature, hard=True, dim=-1)  # (B, C)
            quantized = torch.einsum('bc,bcd->bd', weights, self.codes)  # (B, D)
            indices = None
        else:
            # Get nearest codebook entry
            indices = torch.argmin(distances, dim=-1)  # (B,)
            quantized = self.codes[0, indices, :]  # (B, D)

        # Loss = ||sg(x) - quantized||^2 + beta * ||x - sg(quantized)||^2, where sg() is stop-gradient
        loss = torch.mean((x - quantized.detach()) ** 2) + self.beta * torch.mean((quantized - x.detach()) ** 2)

        return QuantizerOutput(quantized=quantized, indices=indices, loss=loss)


@dataclass
class ResidualQuantizerOutput:
    quantized: torch.Tensor  # (Batch, EmbeddingDim)
    indices: torch.LongTensor  # (Batch, NumCodebooks)
    loss: torch.Tensor
    layer_losses: List[torch.Tensor]  # List of losses from each codebook layer


class ResidualQuantizer(nn.Module):
    def __init__(self, code_sizes: List[int], embedding_dim: int, beta=0.25):
        super().__init__()
        self.codebooks = nn.ModuleList([Quantizer(code_size, embedding_dim, beta) for code_size in code_sizes])

    @torch.no_grad()
    def init_codebooks(self, data: torch.Tensor):
        """Initialize all codebooks with the provided data."""
        residual = data.clone()
        for i, codebook in enumerate(self.codebooks):
            codebook.init_codebooks(residual)
            codebook.eval()
            output = codebook(residual, temperature=0.001)
            codebook.train()
            residual = residual - output.quantized

    def forward(self, x: torch.Tensor, temperature: float=0.001) -> ResidualQuantizerOutput:
        """
        Args:
            x: (Batch, EmbeddingDim)
        Returns:
            quantized: (Batch, EmbeddingDim)
            indices: (Batch, NumCodebooks)
            loss: Scalar
        """
        all_indices = []
        z = torch.zeros_like(x)
        total_loss = 0.0
        layer_losses = []

        for codebook in self.codebooks:
            output = codebook(x, temperature)
            x -= output.quantized
            z += output.quantized
            if not self.training:
                all_indices.append(output.indices)
            total_loss += output.loss
            layer_losses.append(output.loss)
        
        indices = None if self.training else torch.stack(all_indices, dim=1) # (B, NumCodebooks)

        return ResidualQuantizerOutput(quantized=z, indices=indices, loss=total_loss, layer_losses=layer_losses)



@dataclass
class RQVAEOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reconstructed: Optional[torch.FloatTensor] = None
    recon_loss: Optional[torch.FloatTensor] = None
    layer_losses: Optional[List[torch.FloatTensor]] = None
    # quantized: Optional[torch.FloatTensor] = None
    # indices: Optional[List[torch.LongTensor]] = None

class RQVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], code_sizes: List[int], beta=0.25):
        super().__init__()
        self.encoder = MLP(input_dim=input_dim, hidden_dims=hidden_dims, out_dim=hidden_dims[-1])
        # ※ ResidualQuantizer は前のステップで実装したものを想定
        self.quantizer = ResidualQuantizer(code_sizes, hidden_dims[-1], beta)
        self.decoder = MLP(input_dim=hidden_dims[-1], hidden_dims=list(reversed(hidden_dims)), out_dim=input_dim)

    def forward(
        self, 
        features: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        temperature: float = 0.001,
        **kwargs
    ) -> RQVAEOutput:
        
        encoded = self.encoder(features)
        quantizer_output = self.quantizer(encoded, temperature=temperature)
        reconstructed = self.decoder(quantizer_output.quantized)
        
        recon_loss = torch.mean((features - reconstructed) ** 2)
        total_loss = quantizer_output.loss + recon_loss

        return RQVAEOutput(
            loss=total_loss, # HF Trainer はこの `loss` を見て逆伝播します
            reconstructed=reconstructed,
            recon_loss=recon_loss,
            layer_losses=quantizer_output.layer_losses
        )

    @torch.no_grad()    
    def init_codebooks(self, data: torch.Tensor):
        batch_size = 1024
        features = []
        for i in range(0, data.size(0), batch_size):
            batch_data = data[i:i+batch_size]
            encoded = self.encoder(batch_data)
            features.append(encoded)
        features = torch.cat(features, dim=0)
        self.quantizer.init_codebooks(features)