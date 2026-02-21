import torch
from seqrec.module.mlp import MLP
from dataclasses import dataclass
from seqrec.module.feature_extractor import CachedItemFeatureStore
from transformers import Trainer, TrainingArguments
from transformers.utils import ModelOutput
from seqrec.dataset import Item, ItemDataset, ItemDatasetCollator
import gzip
import ast
from typing import Optional, Dict

@dataclass
class AEOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    recon_x: Optional[torch.Tensor] = None

class AE(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.0):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dims, hidden_dims[-1], dropout)
        self.decoder = MLP(hidden_dims[-1], hidden_dims[::-1], input_dim, dropout)

    def forward(self, features):
        z = self.encoder(features)
        z = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-8)  # L2正規化
        #print(features.mean(dim=0).detach().cpu().numpy())
        recon_x = self.decoder(z)
        loss = torch.mean(torch.sum((features - recon_x) ** 2, dim=-1))
        return AEOutput(loss=loss, recon_x=recon_x)

def load_data(dataset_path: str, data_size=100000):
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

def main(dataset_path: str):
    import os
    os.environ["WANDB_PROJECT"] = "RQ-VAE"
    # Settings
    device = 'cuda'
    model_names = {
        'text': "BAAI/bge-base-en-v1.5",
        'image': "google/siglip-base-patch16-224"
    }

    cached_feature_paths = { f'{k}_features': f'{dataset_path}/feature_cache_{k}_{model_names[k].replace("/", "_")}.npz' for k in model_names }
    feature_extractor = CachedItemFeatureStore(cached_feature_paths, normalize=True)
    collate_fn = ItemDatasetCollator(feature_extractor)

    model = AE(input_dim=feature_extractor.feature_dims()['text_features'] + feature_extractor.feature_dims()['image_features'], hidden_dims=[1024, 512, 256], dropout=0.0).to(device)

    item_dataset = ItemDataset({item.item_id: item for item in load_data(dataset_path, data_size=110000)})
    # split into train and eval
    train_size = int(0.9 * len(item_dataset))
    train_dataset = torch.utils.data.Subset(item_dataset, list(range(train_size)))
    eval_dataset = torch.utils.data.Subset(item_dataset, list(range(train_size, len(item_dataset))))
    print(f"Loaded {len(item_dataset)} items.")

    collate_fn = ItemDatasetCollator(feature_extractor)

    warmup_args = TrainingArguments(
        learning_rate=1e-4 * 1024 / 128, 
        weight_decay=0.0,
        num_train_epochs=20, 
        per_device_train_batch_size=1024,
        logging_steps=10, 
        eval_strategy="steps",
        eval_steps=50,
        label_names=["features"],
        output_dir="./output_rqvae_warmup", 
        save_strategy="no",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=warmup_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn
    )
    trainer.train()


if __name__ == "__main__":
    main("dataset/toys")

