import os
import ast
import gzip
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
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
    def __init__(self, 
                 user_seqs, 
                 targets=None, 
                 max_len=50, 
                 pad_token=0):
        """
        Args:
            user_seqs (list): ユーザーごとのアイテムID履歴 [[1, 2, ...], ...]
            targets (list, optional): Eval/Test用の正解アイテムIDリスト。
                                      Noneの場合はTrainモード（Many-to-Many学習）として動作する。
            max_len (int): 系列最大長
            pad_token (int): パディングに使用するID (Loss計算時に無視されるIDと同じである必要がある)
        """
        self.user_seqs = user_seqs
        self.targets = targets
        self.max_len = max_len
        self.pad_token = pad_token

    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, index):
        # 元の系列全体を取得
        seq = self.user_seqs[index]
        
        # -------------------------------------------------------
        # Input と Labels の構築
        # -------------------------------------------------------
        
        if self.targets is None:
            # === Train Mode (Many-to-Many) ===
            # 入力: [x1, x2, ..., x_{N-1}]
            # 正解: [x2, x3, ..., x_N] (次のアイテム)
            # 例: seq=[1, 2, 3] -> input=[1, 2], label=[2, 3]
            
            # 系列が短すぎる場合のガード
            if len(seq) < 2:
                # 無効なデータとしてオール0などを返す（Dataset構築時に除外推奨）
                input_seq = [self.pad_token]
                target_seq = [self.pad_token]
            else:
                input_seq = seq[:-1]
                target_seq = seq[1:]
                
            # max_len に合わせて後ろから切り出す
            input_seq = input_seq[-self.max_len:]
            target_seq = target_seq[-self.max_len:]
            
            # パディングの準備
            seq_len = len(input_seq)
            input_ids = torch.full((self.max_len,), self.pad_token, dtype=torch.long)
            labels = torch.full((self.max_len,), self.pad_token, dtype=torch.long) # 0はignore_indexとする
            
            # 後ろ詰め (Right Padding)
            input_ids[self.max_len - seq_len:] = torch.tensor(input_seq, dtype=torch.long)
            labels[self.max_len - seq_len:] = torch.tensor(target_seq, dtype=torch.long)

        else:
            # === Eval/Test Mode (Many-to-One / Last Item Prediction) ===
            # 入力: 履歴すべて (max_lenまで)
            # 正解: 指定された target_id (最後のステップのみ有効、他は0埋め)
            
            target_item = self.targets[index]
            input_seq = seq[-self.max_len:]
            seq_len = len(input_seq)
            
            input_ids = torch.full((self.max_len,), self.pad_token, dtype=torch.long)
            labels = torch.full((self.max_len,), self.pad_token, dtype=torch.long) # 全て0(無視)で初期化
            
            # 入力を埋める
            input_ids[self.max_len - seq_len:] = torch.tensor(input_seq, dtype=torch.long)
            
            # ラベルは「最後のステップ」だけ正解を入れる
            # (入力系列の最後のアイテムを見た時点で、target_itemを予測してほしい)
            labels[-1] = torch.tensor(target_item, dtype=torch.long)

        # -------------------------------------------------------
        # 戻り値の作成
        # -------------------------------------------------------
        item = {
            "input_ids": input_ids,
            "labels": labels  # Trainerが自動的にLoss計算に使用
        }

        return item

def data_collator(features):
    """
    バッチ構築関数
    Datasetが既に整ったTensorを返しているので、stackするだけでOK
    """
    batch = {}
    batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
    batch['labels'] = torch.stack([f['labels'] for f in features])
    
    if 'seq_img_feat' in features[0]:
        batch['image_feats'] = torch.stack([f['seq_img_feat'] for f in features])
    
    if 'seq_text_feat' in features[0]:
        batch['text_feats'] = torch.stack([f['seq_text_feat'] for f in features])

    return batch


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

class ItemDataset(Dataset):
    def __init__(self, items: Dict[Any, Item]):
        self.items = items
        self.asin_to_id = {item.asin: item.item_id for item in items.values()}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def get_item_by_asin(self, asin: str) -> Optional[Item]:
        item_id = self.asin_to_id.get(asin)
        if item_id is not None:
            return self[item_id]
        return None

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



def load_seq_data(data_path, missing_image_ids, max_len=50, pad_token=0):
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
        pad_token=0
    )
    
    # Eval (targetsあり)
    datasets["eval"] = SequentialRecDataset(
        user_seqs=data_splits["eval"]["seqs"],
        targets=data_splits["eval"]["targets"],
        max_len=max_len,
        pad_token=0
    )
    
    # Test (targetsあり)
    datasets["test"] = SequentialRecDataset(
        user_seqs=data_splits["test"]["seqs"],
        targets=data_splits["test"]["targets"],
        max_len=max_len,
        pad_token=0
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
    dataset = load_seq_data(data_path, missing_image_ids, max_len=max_len, pad_token=0)

    return dataset, item_dataset


def test_dataset_creation():
    import time

    # Example usage
    categories = ["toys", "beauty", "sports"]

    datasets = {}
    item_datasets = {}

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