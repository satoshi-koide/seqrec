import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass
from transformers.utils import ModelOutput

# utils.py と mlp.py のインポートは環境に合わせて維持してください
from .utils import kmeans_init_
from .mlp import MLP

@dataclass
class QuantizerOutput:
    quantized: torch.Tensor
    indices: torch.LongTensor
    loss: torch.Tensor

def get_rotation_matrix(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    ベクトル v (固定) を ベクトル u (動的) に向けるバッチ処理対応の回転行列 R を計算する
    u: (B, D) - 正規化済みであること
    v: (D) - 正規化済みであること
    """
    B, D = u.shape
    v_b = v.unsqueeze(0).expand(B, D)
    
    # A = u v^T - v u^T
    uvT = torch.bmm(u.unsqueeze(2), v_b.unsqueeze(1))  # (B, D, D)
    vuT = torch.bmm(v_b.unsqueeze(2), u.unsqueeze(1))  # (B, D, D)
    A = uvT - vuT
    
    A2 = torch.bmm(A, A)
    dot = torch.sum(u * v_b, dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
    
    I = torch.eye(D, device=u.device, dtype=u.dtype).unsqueeze(0).expand(B, D, D)
    
    # ロドリゲスの回転公式（行列版）
    R = I + A + A2 / (1 + dot + eps)
    return R

class Quantizer(nn.Module):
    """Layer 1 用の通常の Quantizer (回転なし)"""
    def __init__(self, code_size: int, embedding_dim: int, beta: float):
        super().__init__()
        self.codes = nn.Parameter(torch.zeros(1, code_size, embedding_dim))
        self.alpha = 0.1
        self.beta = beta
        self.register_buffer('initialized', torch.tensor(0))

    def init_codebooks(self, data: torch.Tensor, prev_q: Optional[torch.Tensor] = None):
        if self.initialized: return
        with torch.no_grad():
            kmeans_init_(self.codes.data.squeeze(0), data)
            self.initialized.fill_(1)

    def forward(self, x: torch.Tensor, prev_q: Optional[torch.Tensor] = None, temperature: float = 0.001):
        distances = torch.norm(x.unsqueeze(1) - self.codes, dim=-1) / (x.size(1) ** 0.5) / self.alpha

        if self.training:
            weights = F.gumbel_softmax(-distances, tau=temperature, hard=True, dim=-1)
            quantized = torch.einsum('bc,bcd->bd', weights, self.codes)
            indices = torch.argmax(weights, dim=-1)
        else:
            indices = torch.argmin(distances, dim=-1)
            quantized = self.codes[0, indices, :]

        #loss = torch.mean((x - quantized.detach()) ** 2) + self.beta * torch.mean((quantized - x.detach()) ** 2)
        loss_commit = torch.mean(torch.sum((x - quantized.detach()) ** 2, dim=-1))
        loss_codebook = torch.mean(torch.sum((quantized - x.detach()) ** 2, dim=-1))
        loss = loss_commit + self.beta * loss_codebook
        return QuantizerOutput(quantized=quantized, indices=indices, loss=loss)


class RotationalQuantizer(nn.Module):
    """Layer 2 以降用の 回転ベース Quantizer"""
    def __init__(self, code_size: int, embedding_dim: int, beta: float):
        super().__init__()
        self.codes = nn.Parameter(torch.zeros(1, code_size, embedding_dim))
        self.alpha = 0.1
        self.beta = beta
        self.register_buffer('initialized', torch.tensor(0))
        
        v = torch.ones(embedding_dim) / (embedding_dim ** 0.5)
        self.register_buffer('v', v)

    def _get_R_and_R_inv(self, prev_q: torch.Tensor):
        u = F.normalize(prev_q, dim=-1, eps=1e-6)
        R = get_rotation_matrix(u, self.v)
        R_inv = R.transpose(1, 2)
        return R, R_inv

    def init_codebooks(self, residual: torch.Tensor, prev_q: torch.Tensor):
        if self.initialized: return
        with torch.no_grad():
            _, R_inv = self._get_R_and_R_inv(prev_q)
            canonical_residual = torch.bmm(R_inv, residual.unsqueeze(2)).squeeze(2)
            kmeans_init_(self.codes.data.squeeze(0), canonical_residual)
            self.initialized.fill_(1)

    def forward(self, x: torch.Tensor, prev_q: torch.Tensor, temperature: float = 0.001):
        R, R_inv = self._get_R_and_R_inv(prev_q)
        x_canonical = torch.bmm(R_inv, x.unsqueeze(2)).squeeze(2)
        distances = torch.norm(x_canonical.unsqueeze(1) - self.codes, dim=-1) / (x.size(1) ** 0.5) / self.alpha

        if self.training:
            weights = F.gumbel_softmax(-distances, tau=temperature, hard=True, dim=-1)
            quantized_canonical = torch.einsum('bc,bcd->bd', weights, self.codes)
            indices = torch.argmax(weights, dim=-1)
        else:
            indices = torch.argmin(distances, dim=-1)
            quantized_canonical = self.codes[0, indices, :]

        quantized = torch.bmm(R, quantized_canonical.unsqueeze(2)).squeeze(2)
        # loss = torch.mean((x - quantized.detach()) ** 2) + self.beta * torch.mean((quantized - x.detach()) ** 2)
        loss_commit = torch.mean(torch.sum((x - quantized.detach()) ** 2, dim=-1))
        loss_codebook = torch.mean(torch.sum((quantized - x.detach()) ** 2, dim=-1))
        loss = loss_commit + self.beta * loss_codebook
        return QuantizerOutput(quantized=quantized, indices=indices, loss=loss)

@dataclass
class ResidualQuantizerOutput:
    quantized: torch.Tensor
    indices: torch.LongTensor
    loss: torch.Tensor
    layer_losses: List[torch.Tensor]

class ResidualQuantizer(nn.Module):
    def __init__(self, code_sizes: List[int], embedding_dim: int, beta=0.25):
        super().__init__()
        self.codebooks = nn.ModuleList()
        for i, code_size in enumerate(code_sizes):
            if i == 0:
                self.codebooks.append(Quantizer(code_size, embedding_dim, beta))
            else:
                self.codebooks.append(RotationalQuantizer(code_size, embedding_dim, beta))

    @torch.no_grad()
    def init_codebooks(self, data: torch.Tensor):
        residual = data.clone()
        prev_q = torch.zeros_like(data)
        
        for i, codebook in enumerate(self.codebooks):
            # [修正] Layer 1 と Layer 2 以降で呼び出し方を明確に分岐
            if i == 0:
                codebook.init_codebooks(residual)
                codebook.eval()
                output = codebook(residual, temperature=0.001)
            else:
                codebook.init_codebooks(residual, prev_q=prev_q)
                codebook.eval()
                output = codebook(residual, prev_q=prev_q, temperature=0.001)
            
            codebook.train()
            residual = residual - output.quantized
            prev_q = prev_q + output.quantized

    def forward(self, x: torch.Tensor, temperature: float=0.001) -> ResidualQuantizerOutput:
        all_indices = []
        layer_losses = []
        z = torch.zeros_like(x)
        total_loss = 0.0
        residual = x.clone()

        for i, codebook in enumerate(self.codebooks):
            # [修正] Layer 1 と Layer 2 以降で呼び出し方を明確に分岐
            if i == 0:
                output = codebook(residual, temperature=temperature)
            else:
                output = codebook(residual, prev_q=z, temperature=temperature)
            
            residual = residual - output.quantized
            z = z + output.quantized
            
            all_indices.append(output.indices)
            total_loss += output.loss
            layer_losses.append(output.loss)
        
        indices = torch.stack(all_indices, dim=1)
        return ResidualQuantizerOutput(quantized=z, indices=indices, loss=total_loss, layer_losses=layer_losses)


@dataclass
class RQVAEOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reconstructed: Optional[torch.FloatTensor] = None
    recon_loss: Optional[torch.FloatTensor] = None
    layer_losses: Optional[List[torch.FloatTensor]] = None
    indices: Optional[torch.LongTensor] = None
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
        
        recon_loss = torch.mean(torch.sum((features - reconstructed) ** 2, dim=-1))
        total_loss = quantizer_output.loss + recon_loss

        return RQVAEOutput(
            loss=total_loss, # HF Trainer はこの `loss` を見て逆伝播します
            reconstructed=reconstructed,
            recon_loss=recon_loss,
            layer_losses=quantizer_output.layer_losses,
            indices=quantizer_output.indices
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