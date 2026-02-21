import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from dataclasses import dataclass
from transformers.utils import ModelOutput

# utils.py と mlp.py のインポートは環境に合わせて維持してください
from .utils import kmeans_init_
from .mlp import MLP

import math
from transformers import TrainerCallback

class GumbelTemperatureCallback(TrainerCallback):
    def __init__(self, tau_init=1.0, tau_min=0.1, decay_ratio=0.7):
        """
        decay_ratio: 全学習ステップの何%で tau_min に到達させるか（デフォルト70%）
        """
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.decay_ratio = decay_ratio
        self.gamma = None

    def on_train_begin(self, args, state, control, **kwargs):
        # 学習開始時に、Trainerが計算済みの総ステップ数を取得
        total_steps = state.max_steps
        target_step = int(total_steps * self.decay_ratio)
        
        if target_step > 0:
            # tau_min = tau_init * (gamma ^ target_step) を gamma について解く
            self.gamma = math.exp(math.log(self.tau_min / self.tau_init) / target_step)
        else:
            self.gamma = 1.0
            
        print(f"[Callback] Gumbel Softmax scheduler initialized. Target step: {target_step}, Gamma: {self.gamma:.6f}")

    def on_step_begin(self, args, state, control, model, **kwargs):
        # 毎ステップの最初に温度を計算
        if self.gamma is None:
            return
            
        current_tau = self.tau_init * (self.gamma ** state.global_step)
        current_tau = max(self.tau_min, current_tau)
        
        # モデルの属性を直接書き換える
        # (DataParallelやDDP環境でラップされている場合を考慮して model.module をチェック)
        unwrapped_model = model.module if hasattr(model, "module") else model
        unwrapped_model.tau = current_tau

    # (オプション) Wandb等を使っている場合、ログ記録のタイミングで温度も出力すると分析に便利です
    def on_log(self, args, state, control, model, logs=None, **kwargs):
        if logs is not None:
            unwrapped_model = model.module if hasattr(model, "module") else model
            logs["gumbel_tau"] = unwrapped_model.tau

class BetaSchedulerCallback(TrainerCallback):
    def __init__(self, beta_init=0.0, beta_max=0.25, start=0.3):
        """
        decay_ratio: 全学習ステップの何%で beta_max に到達させるか（デフォルト70%）
        """
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.start = start

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps
        self.target_step = int(total_steps * self.start)
        
        if self.target_step > 0:
            # beta_max = beta_init * (diff ^ target_step) を diff について解く
            self.diff = (self.beta_max - self.beta_init) / self.target_step
        else:
            self.diff = 0.0
            
        print(f"[Callback] Gumbel Softmax scheduler initialized. Target step: {self.target_step}, Diff: {self.diff:.6f}")
    def on_step_begin(self, args, state, control, model, **kwargs):
        if self.diff is None:
            return
            
        if state.global_step < self.target_step:
            current_beta = self.beta_init + self.diff * state.global_step
            current_beta = min(self.beta_max, current_beta)
            
            # モデルの属性を直接書き換える
            # (DataParallelやDDP環境でラップされている場合を考慮して model.module をチェック)
            unwrapped_model = model.module if hasattr(model, "module") else model
            unwrapped_model.set_beta(current_beta)

    # (オプション) Wandb等を使っている場合、ログ記録のタイミングで温度も出力すると分析に便利です
    def on_log(self, args, state, control, model, logs=None, **kwargs):
        if logs is not None:
            unwrapped_model = model.module if hasattr(model, "module") else model
            logs["commitment/beta"] = unwrapped_model.beta

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
    def __init__(self, code_size: int, embedding_dim: int, beta: float, spherical_norm: bool, forward_mode: str = "STE", distance_metric: str = "l2"):
        super().__init__()
        self.codes = nn.Parameter(torch.zeros(1, code_size, embedding_dim))
        self.alpha = 0.1
        self.beta = beta
        self.forward_mode = forward_mode
        self.distance_metric = distance_metric
        self.spherical_norm = spherical_norm
        self.register_buffer('initialized', torch.tensor(0))
        # 🚀 連続未使用カウント用のバッファを追加
        self.register_buffer('unused_count', torch.zeros(code_size))
        self.dead_threshold = 50 # 50バッチ連続で使われなかったら復活させる

    def init_codebooks(self, data: torch.Tensor, prev_q: Optional[torch.Tensor] = None):
        if self.initialized: return
        with torch.no_grad():
            kmeans_init_(self.codes.data.squeeze(0), data)
            self.initialized.fill_(1)

    def forward(self, x: torch.Tensor, prev_q: Optional[torch.Tensor] = None, temperature: float = 1.0) -> QuantizerOutput:
        if self.spherical_norm:
            x = F.normalize(x, p=2, dim=-1)
            active_codes = F.normalize(self.codes, p=2, dim=-1)
        else:
            active_codes = self.codes

        # 指定された Metric で距離を計算
        distances = compute_distance(x, active_codes, metric=self.distance_metric)

        if self.training:
            if self.forward_mode == 'gumbel':
                weights = F.gumbel_softmax(-distances, tau=temperature, hard=True, dim=-1)
                quantized = torch.einsum('bc,bcd->bd', weights, active_codes)
                quantized_out = quantized
                indices = torch.argmax(weights, dim=-1)
            else:  # STE
                indices = torch.argmin(distances, dim=-1)

                # ==== 🚀 死んだコードの復活戦略 (Patience付き) ====
                usage = torch.bincount(indices, minlength=self.codes.size(1))
                
                # 使われたコードはカウントリセット、使われなかったコードはカウントアップ
                self.unused_count[usage > 0] = 0
                self.unused_count[usage == 0] += 1
                
                # 閾値を超えた「真の死んだコード」だけを取得
                dead_indices = (self.unused_count >= self.dead_threshold).nonzero(as_tuple=True)[0]

                if len(dead_indices) > 0:
                    rand_idx = torch.randint(0, x.size(0), (len(dead_indices),), device=x.device)
                    self.codes.data[0, dead_indices, :] = x[rand_idx].detach()
                    
                    # 復活させたコードのカウンターをリセット
                    self.unused_count[dead_indices] = 0

                    if self.spherical_norm:
                        active_codes = F.normalize(self.codes, p=2, dim=-1)
                    else:
                        active_codes = self.codes
                        
                    distances = compute_distance(x, active_codes, metric=self.distance_metric)
                    indices = torch.argmin(distances, dim=-1)
                # ========================================================

                quantized = active_codes[0, indices, :]
                quantized_out = x + (quantized - x).detach() # STE trick
        else:
            indices = torch.argmin(distances, dim=-1)
            quantized = active_codes[0, indices, :]
            quantized_out = quantized

        # Loss 計算 (unsqueeze(1) を使うことで codes=(B,1,D) として compute_distance に流し込む)
        loss_commit = compute_distance(x, quantized.detach().unsqueeze(1), metric=self.distance_metric).mean()
        loss_codebook = compute_distance(x.detach(), quantized.unsqueeze(1), metric=self.distance_metric).mean()
        loss = loss_commit + self.beta * loss_codebook
        
        return QuantizerOutput(quantized=quantized_out, indices=indices, loss=loss)


class RotationalQuantizer(nn.Module):
    """回転ベースの Quantizer"""
    def __init__(self, code_size: int, embedding_dim: int, beta: float, forward_mode: str = "STE", distance_metric: str = "l2"):
        super().__init__()
        self.codes = nn.Parameter(torch.zeros(1, code_size, embedding_dim))
        self.alpha = 0.1
        self.beta = beta
        self.distance_metric = distance_metric
        self.forward_mode = forward_mode
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

    def forward(self, x: torch.Tensor, prev_q: torch.Tensor, temperature: float = 1.0):
        R, R_inv = self._get_R_and_R_inv(prev_q)
        x_canonical = torch.bmm(R_inv, x.unsqueeze(2)).squeeze(2)
        
        # 正準空間での距離計算
        raw_distances = compute_distance(x_canonical, self.codes, metric=self.distance_metric)
        # distances = raw_distances / x.size(1) / self.alpha # これはダメ。かなり一様ランダムになってしまう。
        distances = raw_distances

        if self.training:
            if self.forward_mode == 'gumbel':
                weights = F.gumbel_softmax(-distances, tau=temperature, hard=True, dim=-1)
                quantized_canonical = torch.einsum('bc,bcd->bd', weights, self.codes)
                quantized_canonical_out = quantized_canonical
                indices = torch.argmax(weights, dim=-1)
            else:  # STE
                indices = torch.argmin(distances, dim=-1)
                quantized_canonical = self.codes[0, indices, :]
                quantized_canonical_out = x_canonical + (quantized_canonical - x_canonical).detach()
        else:
            indices = torch.argmin(distances, dim=-1)
            quantized_canonical = self.codes[0, indices, :]
            quantized_canonical_out = quantized_canonical

        quantized = torch.bmm(R, quantized_canonical_out.unsqueeze(2)).squeeze(2)
        
        # Loss 計算 (元の空間で計算。※幾何学距離は回転不変なため元空間でも結果は同じ)
        loss_commit = compute_distance(x, quantized_canonical.detach().unsqueeze(1), metric=self.distance_metric).mean()
        loss_codebook = compute_distance(x.detach(), quantized_canonical.unsqueeze(1), metric=self.distance_metric).mean()
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
        spherical_norm: bool = True,
        forward_mode: str = "STE",
        distance_metric: str = "l2",
        use_rotation: bool = False,
    ):
        super().__init__()
        self.codebooks = nn.ModuleList()
        self.use_rotation = use_rotation
        self.forward_mode = forward_mode
        
        for i, code_size in enumerate(code_sizes):
            # i == 0 の場合、または回転を使わない設定の場合は通常の Quantizer を使用
            if i == 0 or not self.use_rotation:
                spherical_norm = (i == 0) and spherical_norm # 最初の層だけ球面正規化する例
                self.codebooks.append(Quantizer(code_size, embedding_dim, beta, spherical_norm=spherical_norm, forward_mode=self.forward_mode, distance_metric=distance_metric))
            else:
                self.codebooks.append(RotationalQuantizer(code_size, embedding_dim, beta, forward_mode=self.forward_mode, distance_metric=distance_metric))

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

    def forward(self, x: torch.Tensor, temperature: float) -> ResidualQuantizerOutput:
        all_indices = []
        layer_losses = []
        z = torch.zeros_like(x)
        total_loss = 0.0
        residual = x.clone()
        debug_metrics = {}

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
    debug_metrics: Optional[Dict[str, float]] = None

class RQVAE(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        code_sizes: List[int], 
        beta: float = 0.25,
        spherical_norm: bool = True,
        forward_mode: str = "STE",
        use_rotation: bool = True,           # トップレベルから切り替え可能に
        distance_metric: str = "l2"  # "l2" or "asymmetric"
    ):
        '''
        RQVAE のコンストラクタ。回転の有無や距離計算の種類もここで指定できるようにする。
        use_rotation = False & distance_metric = "l2" の組み合わせが、従来の Residual Quantizer と同等
        '''
        super().__init__()
        
        self.tau = 0.1
        self.beta = beta
        self.is_warmup = False
        self.forward_mode = forward_mode

        # フラグをそのまま下層へパススルー
        self.quantizer = ResidualQuantizer(
            code_sizes=code_sizes, 
            embedding_dim=hidden_dims[-1], 
            beta=beta,
            forward_mode=forward_mode,
            use_rotation=use_rotation,
            distance_metric=distance_metric
        )

        self.encoder = MLP(input_dim=input_dim, hidden_dims=hidden_dims, out_dim=hidden_dims[-1])
        self.decoder = MLP(input_dim=hidden_dims[-1], hidden_dims=list(reversed(hidden_dims)), out_dim=input_dim)

    def set_warmup_mode(self, mode: bool):
        """Warm-up（純粋なAE）モードのON/OFFを切り替える"""
        self.is_warmup = mode
        if mode:
            print("[Mode] Switched to Autoencoder Warm-up mode. Quantization is bypassed.")
        else:
            print("[Mode] Switched to Full RQ-VAE mode. Quantization is active.")

    def set_beta(self, beta: float):
        self.beta = beta
        for codebook in self.quantizer.codebooks:
            codebook.beta = beta

    def forward(
        self, 
        features: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        normalize = True
        encoded = self.encoder(features)
        #encoded = F.normalize(encoded, p=2, dim=-1) # ここで正規化してもいいですが、Quantizer 内で距離計算前に正規化する方が柔軟性が高いのでそちらに移動しました。

        if normalize:
            features = F.normalize(features, p=2, dim=-1) # この正規化は怪しい。再構成ベクトルも正規化することになるが、それは向きだけを揃えることを意味する。

        # === 追加: Warm-up モード時は量子化をバイパス ===
        if getattr(self, "is_warmup", False):
            reconstructed = self.decoder(encoded)
            if normalize:
                reconstructed = F.normalize(reconstructed, p=2, dim=-1) # ここも正規化するなら、features と同様に正規化する必要があります。
            recon_loss = torch.mean(torch.sum((features - reconstructed) ** 2, dim=-1))

            #with torch.no_grad():
                #features_norm = torc#h.norm(features, dim=-1).mean().item()
                #recon_norm = torch.norm(reconstructed, dim=-1).mean().item()
                #print(f"[Debug] Warm-up mode: recon_loss={recon_loss.item():.4f}, features_norm={features_norm:.4f}, recon_norm={recon_norm:.4f}")
            
            return RQVAEOutput(
                loss=recon_loss, 
                reconstructed=reconstructed,
                recon_loss=recon_loss,
                layer_losses=None,   # Warm-up中はCommitment Lossなし
                indices=None,        # Warm-up中はIDなし
                debug_metrics=None   # Warm-up中はデバッグメトリクスなし
            )

        # 通常の RQ-VAE モード
        quantizer_output = self.quantizer(encoded, temperature=self.tau)
        reconstructed = self.decoder(quantizer_output.quantized)
        if normalize:
            reconstructed = F.normalize(reconstructed, p=2, dim=-1) # ここも正規化するなら、features と同様に正規化する必要があります。

        recon_loss = torch.mean(torch.sum((features - reconstructed) ** 2, dim=-1))
        total_loss = quantizer_output.loss + recon_loss

        debug_metrics = self.compute_debug_metrics(encoded, quantizer_output)

        return RQVAEOutput(
            loss=total_loss, 
            reconstructed=reconstructed,
            recon_loss=recon_loss,
            layer_losses=quantizer_output.layer_losses,
            indices=quantizer_output.indices,
            debug_metrics=debug_metrics
        )

    @torch.no_grad()
    def compute_debug_metrics(self, encoded: torch.Tensor, quantizer_output: QuantizerOutput) -> Dict[str, float]:
        z_var = torch.var(encoded, dim=0).mean()

        encoded_norm = torch.norm(encoded, dim=-1)

        q_norm = torch.nn.functional.normalize(quantizer_output.quantized, p=2, dim=-1)
        zq_cos_sim = torch.mean(torch.sum(F.normalize(encoded, p=2, dim=-1) * q_norm, dim=-1))

        # === 🚀 修正: 生の重みを取得し、必ず F.normalize をかける！ ===
        cb_weights_raw = self.quantizer.codebooks[0].codes if hasattr(self.quantizer.codebooks[0], 'codes') else self.quantizer.codebooks[0].weight
        cb_weights_norm = F.normalize(cb_weights_raw, p=2, dim=-1)

        # 1. 球面上の分散（多様性）
        cb_var_L1 = torch.var(cb_weights_norm).mean() # 正規化後の分散

        # 2. Gumbel Match Ratio (STEの正答率)
        metric = self.quantizer.codebooks[0].distance_metric
        # 🚀 正規化済みの cb_weights_norm を使って距離を計算する
        distances = compute_distance(F.normalize(encoded, p=2, dim=-1), cb_weights_norm, metric=metric).squeeze(1)
        hard_indices = torch.argmin(distances, dim=-1)
        
        if quantizer_output.indices.dim() == 2:
            sampled_indices_L1 = quantizer_output.indices[:, 0]
        else:
            sampled_indices_L1 = quantizer_output.indices
            
        match_ratio_L1 = (hard_indices == sampled_indices_L1).float().mean()

        debug_metrics = {
            "debug/encoded_norm": encoded_norm.mean(),
            "debug/z_variance": z_var,
            "debug/zq_cos_sim": zq_cos_sim,
            "debug/cb_var_L1": cb_var_L1,
            "debug/gumbel_match_ratio_L1": match_ratio_L1
        }
        
        return debug_metrics

    @torch.no_grad()    
    def init_codebooks(self, data: torch.Tensor):
        batch_size = 1024
        features = []
        for i in range(0, data.size(0), batch_size):
            batch_data = data[i:i+batch_size]
            encoded = self.encoder(batch_data)
            encoded = torch.nn.functional.normalize(encoded, p=2, dim=-1)
            features.append(encoded)
        features = torch.cat(features, dim=0)
        self.quantizer.init_codebooks(features)