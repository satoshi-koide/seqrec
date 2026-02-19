import os
import ast
import gzip
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, ClassVar
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

class DataMap:
    def __init__(self, data_path):
        with open(f'{data_path}/datamaps.json', 'r') as f:
            self.datamaps = json.load(f)
        
        print(f"DataMap keys: {list(self.datamaps.keys())}")
        
        self.item2id = {k: int(v) for k, v in self.datamaps['item2id'].items()}
        self.id2item = {int(v): k for k, v in self.item2id.items()}

class SequentialRecDataset(Dataset):
    pad_token: int = 0
    ignore_index: int = -100

    def __init__(self, 
                 user_seqs, 
                 targets=None, 
                 max_len=50):
        self.user_seqs = user_seqs
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, index):
        seq = self.user_seqs[index]
        
        # -------------------------------------------------------
        # Input と Labels の構築 (パディングなし)
        # -------------------------------------------------------
        
        if self.targets is None:
            # === Train Mode ===
            if len(seq) < 2:
                # ガード: 無効な場合は長さ1のダミーを入れる
                input_seq = [self.pad_token]
                target_seq = [self.ignore_index]
            else:
                input_seq = seq[:-1]
                target_seq = seq[1:]
            
            # max_len に合わせて「切り出し」だけ行う (パディングはしない)
            # 例: [1, ..., 100] -> [51, ..., 100]
            input_seq = input_seq[-self.max_len:]
            target_seq = target_seq[-self.max_len:]
            
            # Tensor化 (長さはバラバラのまま)
            input_ids = torch.tensor(input_seq, dtype=torch.long)
            labels = torch.tensor(target_seq, dtype=torch.long)

        else:
            # === Eval/Test Mode ===
            target_item = self.targets[index]
            
            # 入力系列の切り出し
            input_seq = seq[-self.max_len:]
            input_ids = torch.tensor(input_seq, dtype=torch.long)
            
            # ラベルの作成: 全て -100 で初期化
            # 例: [-100, -100, ..., -100]
            labels = torch.full((len(input_seq),), -100, dtype=torch.long)
            
            # 最後の位置だけ正解を入れる
            # Left Padding (右寄せ) するため、有効な系列の末尾が予測位置になります
            labels[-1] = target_item

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    @classmethod
    def data_collator(cls, features):
        input_ids_list = [f['input_ids'] for f in features]
        labels_list = [f['labels'] for f in features]
        
        max_seq_len = max(len(ids) for ids in input_ids_list)
        batch_size = len(features)
        
        # input_ids は 0 で埋める
        padded_input_ids = torch.full(
            (batch_size, max_seq_len), cls.pad_token, dtype=torch.long
        )
        # labels は -100 で埋める
        padded_labels = torch.full(
            (batch_size, max_seq_len), cls.ignore_index, dtype=torch.long
        )
        
        for i, (inp, lbl) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = len(inp)
            # Left Padding (後ろから詰める)
            padded_input_ids[i, max_seq_len - seq_len:] = inp
            padded_labels[i, max_seq_len - seq_len:] = lbl

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels
        }

@dataclass
class Item:
    item_id: int
    asin: str
    title: str
    description: str
    price: float
    brand: str
    categories: str
    image_path: str     # Local path to the image file

    serialize_template: ClassVar[str] = "Item Title: {title}\nBrand: {brand}\nCategory: {categories}\nPrice: ${price}\nDescription: {description}"

    def serialize(self):
        return self.serialize_template.format(
            title=self.title,
            brand=self.brand,
            categories=', '.join(self.categories) if isinstance(self.categories, list) else self.categories,
            price=self.price,
            description=self.description
        )

class ItemDataset(Dataset):
    def __init__(self, items: Dict[Any, Item]):
        self.items = items
        self.asin_to_id = {item.asin: item.item_id for item in items.values()}
        self.idx_to_id = [item_id for idx, item_id in enumerate(self.items.keys())]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for ItemDataset of size {len(self)}")
        item_id = self.idx_to_id[idx]
        return self.items[item_id]
    
    def get_item(self, item_id: int) -> Optional[Item]:
        return self.items.get(item_id, None)

    def get_item_by_asin(self, asin: str) -> Optional[Item]:
        item_id = self.asin_to_id.get(asin)
        return self.get_item(item_id)

class ItemDatasetCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        # batch: List[Item]
        items = [item for item in batch if item is not None]
        item_ids = torch.tensor([item.item_id for item in items], dtype=torch.long)
        features = self.feature_extractor(item_ids, device='cpu')["text_features"]
        
        # ここで features は dict {"text_features": Tensor, "image_features": Tensor} の形で返ってくる想定
        return {"features": features}

def load_missing_ids(file_path, datamaps):
    missing_image_ids = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                asin = line.strip()
                if asin in datamaps.item2id:
                    missing_image_ids.add(datamaps.item2id[asin])
    return missing_image_ids

def load_item_meta(data_path, datamaps):
    meta_sub_path = os.path.join(data_path, 'meta_sub.jsonl')
    if not os.path.exists(meta_sub_path):
        print(f"Creating meta_sub.jsonl at {meta_sub_path}...")
        items = []
        for i, line in enumerate(gzip.open(os.path.join(data_path, 'meta.json.gz'), 'rt', encoding='utf-8')):
            item_info = ast.literal_eval(line.strip())
            if i == 0:
                print(f"Example item info: {item_info}")
            asin = item_info['asin']
            if asin in datamaps.item2id:
                item_id = datamaps.item2id[asin]
                items.append({
                    "item_id": item_id,
                    "asin": asin,
                    "title": item_info.get('title', None),
                    "description": item_info.get('description', None),
                    "price": item_info.get('price', None),
                    "brand": item_info.get('brand', None),
                    "categories": item_info.get('categories', None)
                })
        with open(meta_sub_path, 'w', encoding='utf-8') as out_f:
            for item in sorted(items, key=lambda x: x['item_id']):
                out_f.write(json.dumps(item) + "\n")
    
    items = {}
    for line in open(meta_sub_path, 'r', encoding='utf-8'):
        item_info = json.loads(line.strip())
        item_id = item_info['item_id']
        asin = item_info['asin']
        title = item_info.get('title', None)
        description = item_info.get('description', None)
        price = item_info.get('price', None)
        categories = item_info.get('categories', None)
        categories = categories[0] if categories else None  # カテゴリはリストの最初の要素を使用
        brand = item_info.get('brand', None)
        image_path = os.path.join(data_path, 'images', f"{asin}.jpg")  # 画像のローカルパス
        items[item_id] = Item(item_id=item_id, asin=asin, title=title, description=description, price=price, brand=brand, categories=categories, image_path=image_path)

    return ItemDataset(items)



def load_seq_data(data_path, missing_image_ids, max_len=50):
    user_ids = []
    sequences = []
    with open(os.path.join(data_path, 'sequential_data.txt'), 'r') as f:
        for line in f:
            line_ids = list(map(int, line.strip().split()))
            # line_ids[0]: UserID, line_ids[1:]: ItemIDs
            sequences.append(line_ids[1:])
            user_ids.append(line_ids[0])

    data_splits = {
        "train": {"seqs": [], "targets": None}, # For Train, targets=None (automatic Many-to-Many mode)
        "eval":  {"seqs": [], "targets": []},
        "test":  {"seqs": [], "targets": []}
    }
    
    # 欠損画像セット (テストデータのフィルタリング用)
    missing_set = set(missing_image_ids)

    for seq in sequences:
        if len(seq) < 3: continue

        # --- Test Set ---
        # 入力: [..., n-2] -> 正解: n-1 (最後)
        test_target = seq[-1]
        test_input = seq[:-1]
        
        if test_target not in missing_set:
            data_splits["test"]["seqs"].append(test_input)
            data_splits["test"]["targets"].append(test_target)

        # --- Eval Set ---
        # 入力: [..., n-3] -> 正解: n-2 (最後から2番目)
        eval_target = seq[-2]
        eval_input = seq[:-2]
        
        if eval_target not in missing_set:
            data_splits["eval"]["seqs"].append(eval_input)
            data_splits["eval"]["targets"].append(eval_target)

        # --- Train Set (Many-to-Many) ---
        # TestとEvalで使った部分を除外した系列全体を学習に使う
        # Datasetクラス内で [:-1] を入力、[1:] を正解に変換するので、
        # ここでは「学習に使いたい履歴全体」をそのまま渡す。
        train_seq = seq[:-2]
        
        if len(train_seq) >= 2:
            data_splits["train"]["seqs"].append(train_seq)
            # targets は append しない (Noneのまま)

    # Datasetインスタンス化
    datasets = {}
    
    # Train (targets=None)
    datasets["train"] = SequentialRecDataset(
        user_seqs=data_splits["train"]["seqs"],
        targets=None, # これが Many-to-Many モードのトリガー
        max_len=max_len,
    )
    
    # Eval (targetsあり)
    datasets["eval"] = SequentialRecDataset(
        user_seqs=data_splits["eval"]["seqs"],
        targets=data_splits["eval"]["targets"],
        max_len=max_len,
    )
    
    # Test (targetsあり)
    datasets["test"] = SequentialRecDataset(
        user_seqs=data_splits["test"]["seqs"],
        targets=data_splits["test"]["targets"],
        max_len=max_len,
    )
        
    return datasets

def create_datasets(data_path, max_len=50):
    """
    Many-to-Many 学習用にデータセットを作成するファクトリー関数
    """
    
    # 1. Load data maps
    datamaps = DataMap(data_path)
    print(f"Number of items: {len(datamaps.id2item)}")
    
    # [Optional] Load missing image IDs (This is used to filter out test/eval samples that have missing images, as per the original code's logic)
    missing_image_ids = load_missing_ids(os.path.join(data_path, 'missing_asins.txt'), datamaps)
    print(f"Loaded {len(missing_image_ids)} missing image IDs. {missing_image_ids}")

    # 2. Load item metadata (ASIN, text, image path)
    item_dataset = load_item_meta(data_path, datamaps)

    # 3. Load sequential_data.txt
    dataset = load_seq_data(data_path, missing_image_ids, max_len=max_len)

    return dataset, item_dataset


def test_dataset_creation():
    import time

    # Example usage
    categories = ["toys", "beauty", "sports"]

    datasets = {}
    item_datasets = {}

    data_collator = SequentialRecDataset.data_collator

    for category in categories:
        data_dir = f"dataset/{category}"
        max_len = 50
        
        # Optional: Load pre-extracted features (if available)
        # image_feat = torch.load("image_features.pt")  # (num_items + 1, dim)
        # text_feat = torch.load("text_features.pt")    # (num_items + 1, dim)
    
        datasets[category], item_datasets[category] = create_datasets(data_dir, max_len)
        
        print(f"Category: {category}")
        print(f"\tNumber of items: {len(item_datasets[category])}")
        avg_seq_len = np.mean([len(seq) for seq in datasets[category]['train'].user_seqs])
        print(f"\tTrain samples: {len(datasets[category]['train'])}")
        print(f"\tEval samples: {len(datasets[category]['eval'])}")
        print(f"\tTest samples: {len(datasets[category]['test'])}")
        print(f"\tAverage sequence length (train): {avg_seq_len:.2f}")
        print()
        print(f"\tExample item: {item_datasets[category][1]}")  # 最初のアイテムの例を表示
        image = Image.open(item_datasets[category][1].image_path)

        # 4. DataLoader
        train_loader = DataLoader(datasets[category]["train"], batch_size=128, shuffle=True, collate_fn=data_collator)
        test_loader = DataLoader(datasets[category]["test"], batch_size=128, shuffle=False, collate_fn=data_collator)

        print(f"\tNumber of batches (train): {len(train_loader)}")
        for batch in train_loader:
            print(batch["input_ids"].shape)  # (batch_size, max_len)
            print(batch["labels"].shape)   # (batch_size,)
            break

def test_datamap_loading():
    data_path = "dataset/toys"
    datamap = DataMap(data_path)
    print(f"Number of items: {len(datamap.id2item)}")
    print(f"Example item2id: {list(datamap.item2id.items())[:5]}")

if __name__ == "__main__":
    test_dataset_creation()
    # test_datamap_loading()