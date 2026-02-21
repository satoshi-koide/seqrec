import ast
import numpy as np
import torch
import gzip
from typing import Dict
import os

from transformers import Trainer, TrainingArguments
from seqrec.dataset import Item, ItemDataset, ItemDatasetCollator
#from seqrec.module.rqvae import RQVAE
from seqrec.module.rotational_rqvae import RQVAE, GumbelTemperatureCallback, BetaSchedulerCallback
from seqrec.module.feature_extractor import CachedItemFeatureStore

def load_data(dataset_path: str, data_size=20000):
    for item_id, line in enumerate(gzip.open(dataset_path + '/meta.json.gz', 'rt')):
        if data_size and item_id >= data_size:
            break
        item_info = ast.literal_eval(line)
        categories = item_info.get('categories', None)
        categories = categories[0] if categories else None
        yield Item(
            item_id=item_id,
            asin=item_info['asin'],
            title=item_info.get('title', None),
            description=item_info.get('description', None),
            price=item_info.get('price', None),
            brand=item_info.get('brand', None),
            categories=categories,
            image_path=item_info.get('image_path', None)
        )

import torch
import math
from typing import Dict
from transformers import Trainer

class RQVAETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # カスタム指標を一時的に溜めておくバッファ
        self._custom_logs_buffer = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        # ロギングのタイミングでバッファに保存する
        if self.state.global_step % self.args.logging_steps == 0:
            self._custom_logs_buffer.clear()
            
            # 再構成ロス
            if outputs.recon_loss is not None:
                self._custom_logs_buffer["loss/recon"] = outputs.recon_loss.item()
            
            # 各層の Commitment Loss
            if outputs.layer_losses is not None:
                for i, l_loss in enumerate(outputs.layer_losses):
                    self._custom_logs_buffer[f"loss/L{i+1}"] = l_loss.item()
            
            # 変更点: Active Codes の代わりに Perplexity を計算
            if outputs.indices is not None:
                B, num_layers = outputs.indices.shape
                for i in range(num_layers):
                    # 該当レイヤーの全インデックスを1次元化
                    idx = outputs.indices[:, i].flatten()
                    
                    # 各インデックスの出現回数をカウント
                    _, counts = torch.unique(idx, return_counts=True)
                    
                    # 確率分布 p(x) を計算
                    probs = counts.float() / counts.sum()
                    
                    # エントロピー H = -Σ p(x) log p(x)
                    # (log(0) を防ぐために微小な値 1e-10 を足す)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    
                    # パープレキシティ = exp(H)
                    perplexity = torch.exp(entropy).item()
                    
                    self._custom_logs_buffer[f"usage/L{i+1}_perp"] = perplexity

            if hasattr(outputs, "debug_metrics") and outputs.debug_metrics:
                for k, v in outputs.debug_metrics.items():
                    self._custom_logs_buffer[k] = v.item()

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        # 1. 親クラスの training_step を呼ぶ（ここで順伝播と loss.backward() が実行される）
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch, **kwargs)

        # 2. Backward 直後のタイミングで、Encoder の勾配ノルムを計算・記録
        if self.state.global_step % self.args.logging_steps == 0:
            # DataParallel や DDP でラップされている場合を考慮
            unwrapped_model = model.module if hasattr(model, "module") else model
            
            # ※注意: 実際のモデルの Encoder の属性名に合わせて変更してください
            # (例: self.encoder, self.feature_extractor など)
            encoder = getattr(unwrapped_model, "encoder", None) 
            
            if encoder is not None:
                total_norm = 0.0
                for p in encoder.parameters():
                    if p.grad is not None:
                        # 勾配の L2 ノルムの二乗を足し合わせる
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                
                # 全体の平方根をとる
                total_norm = total_norm ** 0.5
                self._custom_logs_buffer["grad_norm/encoder"] = total_norm

            quantizer = getattr(unwrapped_model, "quantizer", None)
            if quantizer is not None:
                total_norm = 0.0
                for p in quantizer.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                
                self._custom_logs_buffer["grad_norm/codebook"] = total_norm ** 0.5
        return loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        # Trainer が出力しようとしている標準の logs 辞書に、バッファの中身を合体させる
        if self._custom_logs_buffer:
            logs.update(self._custom_logs_buffer)
        
        # 受け取った追加引数 (start_time など) もそのまま親クラスに渡す
        super().log(logs, *args, **kwargs)


def inspect_codebook_scales(model):
    print("=== RQ-VAE Codebook Scales by Layer ===")
    
    for i, q in enumerate(model.quantizer.codebooks):
        # コードブックのテンソルを取得 (shape: [codebook_size, dim] を想定)
        codes = q.codes
        
        # 勾配計算から切り離して純粋な値として扱う
        if isinstance(codes, torch.nn.Parameter):
            codes = codes.detach()
            
        # 1. 全体の分散 (Variance) と標準偏差 (Std)
        variance = torch.var(codes).item()
        std_dev = torch.std(codes).item()
        
        # 2. 各ベクトルの L2ノルム（原点からの距離）の平均と最大値
        norms = torch.norm(codes, p=2, dim=-1)
        mean_norm = norms.mean().item()
        max_norm = norms.max().item()
        
        print(f"Layer {i+1}:")
        print(f"  Variance : {variance:.6f}")
        print(f"  Std Dev  : {std_dev:.6f}")
        print(f"  Mean Norm: {mean_norm:.6f} (Max: {max_norm:.6f})")
        print("-" * 40)

# def create_optimizer(model, lr: float, weight_decay: float):
#     # パラメータをグループ分け
#     encoder_params = [p for n, p in model.named_parameters() if "encoder" in n]
#     decoder_params = [p for n, p in model.named_parameters() if "decoder" in n]
#     quantizer_params = [p for n, p in model.named_parameters() if "quantizer" in n]

#     # 学習率の設定
#     optim_groups = [
#         {"params": quantizer_params, "lr": lr, "weight_decay": weight_decay},   # Quantizerだけ学習
#         {"params": encoder_params, "lr": 0.0},      # 凍結したい層は 0
#         {"params": decoder_params, "lr": 0.0},      # 凍結したい層は 0
#     ]

#     optimizer = torch.optim.AdamW(optim_groups)
#     return optimizer

def main(dataset_path: str):
    os.environ["WANDB_PROJECT"] = "RQ-VAE"

    # Settings
    device = 'cuda'
    model_names = {
        'text': "BAAI/bge-base-en-v1.5",
        'image': "google/siglip-base-patch16-224"
    }

    cached_feature_paths = { f'{k}_features': f'{dataset_path}/feature_cache_{k}_{model_names[k].replace("/", "_")}.npz' for k in model_names }
    feature_extractor = CachedItemFeatureStore(cached_feature_paths, normalize=True)
    input_dim = sum(feature_extractor.feature_dims().values())

    print("Example feature vector (first item):")
    for i in range(5):
        example_features = feature_extractor([i])  # IDだけ渡して特徴量を取得
        for k, v in example_features.items():
            print(f"  {k} features shape: {v.shape}, norm: {torch.norm(v[0]).item():.4f}")

    forward_mode = 'STE' # or 'gumbel'
    #forward_mode = 'gumbel'

    item_dataset = ItemDataset({item.item_id: item for item in load_data(dataset_path, data_size=None)})
    print(f"Loaded {len(item_dataset)} items.")

    collate_fn = ItemDatasetCollator(feature_extractor)

    rqvae = RQVAE(input_dim=input_dim, hidden_dims=[1024, 512, 256], code_sizes=[256, 256, 256], beta=0.25, spherical_norm=True, forward_mode=forward_mode, use_rotation=False).to(device)

    # ==========================================
    # Stage 1: Autoencoder Warm-up (事前学習)
    # ==========================================
    rqvae.set_warmup_mode(True)

    # Warm-up用のTrainer設定（エポック数は5〜10程度で十分です）
    warmup_batch_size = 2048
    warmup_args = TrainingArguments(
        learning_rate=1e-4 * warmup_batch_size / 128, 
        weight_decay=0.0,
        num_train_epochs=20, 
        per_device_train_batch_size=warmup_batch_size,
        logging_steps=100,
        output_dir="./output_rqvae_warmup", 
        save_strategy="no")
    warmup_trainer = RQVAETrainer(model=rqvae, args=warmup_args, train_dataset=item_dataset, data_collator=collate_fn)
    warmup_trainer.train()

    # ==========================================
    # Stage 2: Hierarchical K-means Initialization
    # ==========================================
    print("Initializing codebooks...")
    item_ids = torch.tensor(list(range(10000)), dtype=torch.long, device=device)
    print("Extracting features for codebook initialization...")
    item_features = torch.cat([feat for feat in feature_extractor(item_ids).values()], dim=-1).to(device)  # (NumItems, input_dim)
    print("Initializing codebooks with item features...")
    rqvae.init_codebooks(item_features)

    inspect_codebook_scales(rqvae)


    # ==========================================
    # Stage 3: RQ-VAE Fine-tuning (本学習)
    # ==========================================
    rqvae.set_warmup_mode(False)

    # 2. Encoder と Decoder の勾配計算をオフ（凍結）
    for param in rqvae.parameters():
        param.requires_grad = False
    for param in rqvae.quantizer.parameters():
        param.requires_grad = True

    batch_size = 2048
    # lr = 1e-5 * batch_size / 128
    # weight_decay=0.01
    # optimizer = create_optimizer(rqvae, lr=lr, weight_decay=weight_decay) # Encoder / Decoder は lr=0 で凍結、Quantizer のみ学習

    training_args = TrainingArguments(
        output_dir="./output_rqvae",
        
        # 学習設定
        num_train_epochs=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=128, # 注意: メモリ圧迫する場合は下げる
        learning_rate=1e-5 * batch_size / 128, # バッチサイズに応じた学習率スケーリング
        weight_decay=0.0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        
        # 高速化・ログ
        #fp16=torch.cuda.is_available(), # GPUがあればFP16有効化
        bf16=torch.cuda.is_available(), # GPUがあればBF16有効化
        logging_dir='./logs',
        logging_steps=100,
        dataloader_num_workers=0, # データローダーの並列数
    )

    if forward_mode == 'gumbel':
        print('[Info] Using Gumbel-Softmax relaxation for quantization during training.')
        callbacks = [GumbelTemperatureCallback(tau_init=0.1, tau_min=0.001, decay_ratio=0.7)]
    elif forward_mode == 'STE':
        print('[Info] Using Straight-Through Estimator (STE) for quantization during training.')
        callbacks = []

    # Commitment loss の重みを徐々に増やすスケジューラーも追加
    callbacks.append(BetaSchedulerCallback(beta_init=0.0, beta_max=0.25, start=0.3))

    trainer = RQVAETrainer(
        model=rqvae,
        args=training_args,
        train_dataset=item_dataset,
        data_collator=collate_fn,
        #optimizers = (optimizer, None), # カスタムオプティマイザを直接渡す
        callbacks=callbacks,
    )
    trainer.train()


    # 3. 全体の微調整フェーズ（Encoder / Decoder の凍結を解除して全体を微調整）
    for param in rqvae.parameters():
        param.requires_grad = True

    if forward_mode == 'gumbel':
        print('[Info] Using Gumbel-Softmax relaxation for quantization during training.')
        callbacks = [GumbelTemperatureCallback(tau_init=0.1, tau_min=0.001, decay_ratio=0.7)]
    elif forward_mode == 'STE':
        print('[Info] Using Straight-Through Estimator (STE) for quantization during training.')
        callbacks = []

    # Commitment loss の重みを徐々に増やすスケジューラーも追加
    callbacks.append(BetaSchedulerCallback(beta_init=0.0, beta_max=0.25, start=0.3))

    batch_size = 2048
    training_args = TrainingArguments(
        output_dir="./output_rqvae_phase2",
        
        # 学習設定
        num_train_epochs=2000,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=128, # 注意: メモリ圧迫する場合は下げる
        learning_rate=1e-5 * batch_size / 128, # バッチサイズに応じた学習率スケーリング
        weight_decay=0.0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        
        # 高速化・ログ
        #fp16=torch.cuda.is_available(), # GPUがあればFP16有効化
        bf16=torch.cuda.is_available(), # GPUがあればBF16有効化
        logging_dir='./logs',
        logging_steps=100,
        dataloader_num_workers=0, # データローダーの並列数
        report_to="wandb",
    )

    trainer = RQVAETrainer(
        model=rqvae,
        args=training_args,
        train_dataset=item_dataset,
        data_collator=collate_fn,
        #optimizers = (optimizer, None), # カスタムオプティマイザを直接渡す
        callbacks=callbacks,
    )
    trainer.train()

    # Save model
    category = dataset_path.split("/")[-1]
    torch.save(rqvae.state_dict(), f"./rqvae_final_{category}_{forward_mode}.pth")

     # 学習後のコードブックのスケールを再度確認


if __name__ == "__main__":
    main("dataset/toys")

