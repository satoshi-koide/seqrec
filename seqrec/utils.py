import torch
from transformers import EvalPrediction
from transformers.utils import is_flash_attn_2_available

def get_optimal_attention_config():
    """
    現在の環境で利用可能な最速の Attention 実装とデータ型を返します。
    優先順位: Flash Attention 2 > SDPA (PyTorch Native) > Eager (Default)
    """
    
    # 1. Flash Attention 2 が使えるかチェック
    # (ライブラリがインストールされており、かつGPUがAmpere以上であること)
    if is_flash_attn_2_available() and torch.cuda.is_available():
        print("🚀 Using Flash Attention 2")
        return {
            "attn_implementation": "flash_attention_2",
            # FA2は fp16 か bf16 が必須。
            # GPUが bf16 対応なら bf16 (数値安定性が高い)、そうでなければ fp16
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        }
    
    # 2. PyTorch 2.0 以降なら SDPA (Scaled Dot Product Attention) を使う
    # これはT4やV100でも動作し、そこそこ速い
    elif hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("⚡ Using PyTorch SDPA (Scaled Dot Product Attention)")
        return {
            "attn_implementation": "sdpa",
            "torch_dtype": torch.float16, # SDPAもfp16推奨
        }
    
    # 3. それ以外 (古いPyTorchなど)
    else:
        print("🐢 Using Default Attention (Eager)")
        return {
            "attn_implementation": "eager",
            "torch_dtype": torch.float32,
        }