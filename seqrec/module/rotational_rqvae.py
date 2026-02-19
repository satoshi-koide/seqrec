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
    
    uvT = torch.bmm(u.unsqueeze(2), v_b.unsqueeze(1))  # (B, D, D)
    vuT = torch.bmm(v_b.unsqueeze(2), u.unsqueeze(1))  # (B, D, D)
    A = uvT - vuT
    
    A2 = torch.bmm(A, A)
    dot = torch.sum(u * v_b, dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
    
    I = torch.eye(D, device=u.device, dtype=u.dtype).unsqueeze(0).expand(B, D, D)
    return I + A + A2 / (1 + dot + eps)

def compute_distance(x: torch.Tensor, codes: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    """
    距離計算を統合したヘルパー関数。コード選択(B, C)とLoss計算(B, 1)の両方に対応。
    x: (B, D)
    codes: (1, C, D) または (B, 1, D)
    """
    # 基本の L2 距離の二乗 (MSE)
    mse_distance = torch.sum((x.unsqueeze(1) - codes) ** 2, dim=-1)
    
    if metric == "l2":
        return mse_distance
    elif metric == "asymmetric":
        # 飛び越え & 方向違いを検知する幾何学ペナルティ: ReLU(||c||^2 - x・c)
        c_sq_norm = torch.sum(codes ** 2, dim=-1)
        dot_product = torch.sum(x.unsqueeze(1) * codes, dim=-1)
        penalty = F.relu(c_sq_norm - dot_product)
        return mse_distance + penalty
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


class Quantizer(nn.Module):
    """通常の Quantizer (回転なし)"""
    def __init__(self, code_size: int, embedding_dim: int, beta: float, distance_metric: str = "l2"):
        super().__init__()
        self.codes = nn.Parameter(torch.zeros(1, code_size, embedding_dim))
        self.alpha = 0.1
        self.beta = beta
        self.distance_metric = distance_metric
        self.register_buffer('initialized', torch.tensor(0))

    def init_codebooks(self, data: torch.Tensor, prev_q: Optional[torch.Tensor] = None):
        if self.initialized: return
        with torch.no_grad():
            kmeans_init_(self.codes.data.squeeze(0), data)
            self.initialized.fill_(1)

    def forward(self, x: torch.Tensor, prev_q: Optional[torch.Tensor] = None, temperature: float = 0.001):
        # 指定された Metric で距離を計算し、Gumbel用にスケールを調整
        raw_distances = compute_distance(x, self.codes, metric=self.distance_metric)
        distances = raw_distances / x.size(1) / self.alpha

        if self.training:
            weights = F.gumbel_softmax(-distances, tau=temperature, hard=True, dim=-1)
            quantized = torch.einsum('bc,bcd->bd', weights, self.codes)
            indices = torch.argmax(weights, dim=-1)
        else:
            indices = torch.argmin(distances, dim=-1)
            quantized = self.codes[0, indices, :]

        # Loss 計算 (unsqueeze(1) を使うことで codes=(B,1,D) として compute_distance に流し込む)
        loss_commit = compute_distance(x, quantized.unsqueeze(1).detach(), metric=self.distance_metric).mean()
        loss_codebook = compute_distance(x.detach(), quantized.unsqueeze(1), metric=self.distance_metric).mean()
        loss = loss_commit + self.beta * loss_codebook
        
        return QuantizerOutput(quantized=quantized, indices=indices, loss=loss)


class RotationalQuantizer(nn.Module):
    """回転ベースの Quantizer"""
    def __init__(self, code_size: int, embedding_dim: int, beta: float, distance_metric: str = "l2"):
        super().__init__()
        self.codes = nn.Parameter(torch.zeros(1, code_size, embedding_dim))
        self.alpha = 0.1
        self.beta = beta
        self.distance_metric = distance_metric
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
        
        # 正準空間での距離計算
        raw_distances = compute_distance(x_canonical, self.codes, metric=self.distance_metric)
        distances = raw_distances / x.size(1) / self.alpha

        if self.training:
            weights = F.gumbel_softmax(-distances, tau=temperature, hard=True, dim=-1)
            quantized_canonical = torch.einsum('bc,bcd->bd', weights, self.codes)
            indices = torch.argmax(weights, dim=-1)
        else:
            indices = torch.argmin(distances, dim=-1)
            quantized_canonical = self.codes[0, indices, :]

        quantized = torch.bmm(R, quantized_canonical.unsqueeze(2)).squeeze(2)
        
        # Loss 計算 (元の空間で計算。※幾何学距離は回転不変なため元空間でも結果は同じ)
        loss_commit = compute_distance(x, quantized.unsqueeze(1).detach(), metric=self.distance_metric).mean()
        loss_codebook = compute_distance(x.detach(), quantized.unsqueeze(1), metric=self.distance_metric).mean()
        loss = loss_commit + self.beta * loss_codebook
        
        return QuantizerOutput(quantized=quantized, indices=indices, loss=loss)


@dataclass
class ResidualQuantizerOutput:
    quantized: torch.Tensor
    indices: torch.LongTensor
    loss: torch.Tensor
    layer_losses: List[torch.Tensor]

class ResidualQuantizer(nn.Module):
    def __init__(
        self, 
        code_sizes: List[int], 
        embedding_dim: int, 
        beta: float = 0.25,
        use_rotation: bool = True,
        distance_metric: str = "l2"
    ):
        super().__init__()
        self.codebooks = nn.ModuleList()
        self.use_rotation = use_rotation
        
        for i, code_size in enumerate(code_sizes):
            # i == 0 の場合、または回転を使わない設定の場合は通常の Quantizer を使用
            if i == 0 or not self.use_rotation:
                self.codebooks.append(Quantizer(code_size, embedding_dim, beta, distance_metric))
            else:
                self.codebooks.append(RotationalQuantizer(code_size, embedding_dim, beta, distance_metric))

    @torch.no_grad()
    def init_codebooks(self, data: torch.Tensor):
        residual = data.clone()
        prev_q = torch.zeros_like(data)
        
        for i, codebook in enumerate(self.codebooks):
            if i == 0 or not self.use_rotation:
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
            if i == 0 or not self.use_rotation:
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

class RQVAE(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        code_sizes: List[int], 
        beta: float = 0.25,
        use_rotation: bool = True,           # トップレベルから切り替え可能に
        distance_metric: str = "asymmetric"  # "l2" or "asymmetric"
    ):
        '''
        RQVAE のコンストラクタ。回転の有無や距離計算の種類もここで指定できるようにする。
        use_rotation = False & distance_metric = "l2" の組み合わせが、従来の Residual Quantizer と同等
        '''
        super().__init__()
        self.encoder = MLP(input_dim=input_dim, hidden_dims=hidden_dims, out_dim=hidden_dims[-1])
        
        # フラグをそのまま下層へパススルー
        self.quantizer = ResidualQuantizer(
            code_sizes=code_sizes, 
            embedding_dim=hidden_dims[-1], 
            beta=beta,
            use_rotation=use_rotation,
            distance_metric=distance_metric
        )
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
            loss=total_loss, 
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