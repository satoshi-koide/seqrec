import ast
import numpy as np
import torch
import gzip
from typing import Dict

from transformers import Trainer, TrainingArguments
from seqrec.dataset import Item, ItemDataset, ItemDatasetCollator
#from seqrec.module.rqvae import RQVAE
from seqrec.module.rotational_rqvae import RQVAE
from seqrec.module.feature_extractor import ItemFeatureStore

def load_data(dataset_path: str, data_size=20000):
    for item_id, line in enumerate(gzip.open(dataset_path + '/meta.json.gz', 'rt')):
        if item_id >= data_size:
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

class RQVAETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # カスタム指標を一時的に溜めておくバッファ
        self._custom_logs_buffer = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        # ロギングのタイミングでバッファに保存する（self.log はここでは呼ばない）
        if self.state.global_step % self.args.logging_steps == 0:
            self._custom_logs_buffer.clear()
            
            # 再構成ロス
            if outputs.recon_loss is not None:
                self._custom_logs_buffer["loss/recon"] = outputs.recon_loss.item()
            
            # 各層の Commitment Loss
            if outputs.layer_losses is not None:
                for i, l_loss in enumerate(outputs.layer_losses):
                    self._custom_logs_buffer[f"loss/L{i+1}"] = l_loss.item()
            
            # ※ Perplexity や Active Codes を計算している場合はここに追加
            if outputs.indices is not None:
                B, num_layers = outputs.indices.shape
                for i in range(num_layers):
                    active = torch.unique(outputs.indices[:, i]).numel()
                    self._custom_logs_buffer[f"usage/L{i+1}_active"] = active

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        # Trainer が出力しようとしている標準の logs 辞書に、バッファの中身を合体させる
        if self._custom_logs_buffer:
            logs.update(self._custom_logs_buffer)
        
        # 受け取った追加引数 (start_time など) もそのまま親クラスに渡す
        super().log(logs, *args, **kwargs)


def main(dataset_path: str):
    device = 'cuda'

    item_dataset = ItemDataset({item.item_id: item for item in load_data(dataset_path, data_size=100000)})
    print(f"Loaded {len(item_dataset)} items.")
    feature_extractor = ItemFeatureStore(item_dataset, image_model_name=None).to(device)
    feature_extractor.build_cache(batch_size=512, verbose=True)


    collate_fn = ItemDatasetCollator(feature_extractor)

    rqvae = RQVAE(input_dim=768, hidden_dims=[256, 128, 32], code_sizes=[256, 256, 256], beta=0.25).to(device)

    # init codebooks with item features
    print("Initializing codebooks...")
    item_ids = torch.tensor(list(range(10000)), dtype=torch.long, device=device)
    print("Extracting features for codebook initialization...")
    item_features = feature_extractor(item_ids.to(device))['text_features']  # (NumItems, FeatureDim)
    print("Initializing codebooks with item features...")
    rqvae.init_codebooks(item_features)


    batch_size = 1024
    training_args = TrainingArguments(
        output_dir="./output_rqvae",
        
        # 学習設定
        num_train_epochs=1000,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=128, # 注意: メモリ圧迫する場合は下げる
        learning_rate=1e-4 * batch_size / 128, # バッチサイズに応じた学習率スケーリング
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        # 高速化・ログ
        #fp16=torch.cuda.is_available(), # GPUがあればFP16有効化
        bf16=torch.cuda.is_available(), # GPUがあればBF16有効化
        logging_dir='./logs',
        logging_steps=10,
        dataloader_num_workers=0, # データローダーの並列数
    )

    trainer = RQVAETrainer(
        model=rqvae,
        args=training_args,
        train_dataset=item_dataset,
        data_collator=collate_fn,
    )
    trainer.train()

if __name__ == "__main__":
    main("dataset/toys")

