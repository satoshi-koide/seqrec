from typing import List
from torch import nn
from torch import Tensor
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x) -> Tensor:
        return l2norm(x, dim=self.dim, eps=self.eps)

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.0,
        apply_input_norm: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout = dropout

        dims = [self.input_dim] + self.hidden_dims + [self.out_dim]
        
        self.mlp = nn.Sequential()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            #if 0 < i or (i == 0 and apply_input_norm):
            #    self.mlp.append(nn.LayerNorm(in_d)) # Maybe important.
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))
            if i != len(dims)-2:
                self.mlp.append(nn.LayerNorm(out_d)) # Post Norm
                self.mlp.append(nn.SiLU())
                if dropout != 0:
                    self.mlp.append(nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.input_dim, f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        return self.mlp(x)