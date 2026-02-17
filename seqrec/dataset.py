import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SequentialRecDataset(Dataset):
    def __init__(self, 
                 user_seqs, 
                 targets=None, 
                 max_len=50, 
                 image_features=None, 
                 text_features=None, 
                 pad_token=0):
        """
        Args:
            user_seqs (list): ユーザーごとのアイテムID履歴 [[1, 2, ...], ...]
            targets (list, optional): Eval/Test用の正解アイテムIDリスト。
                                      Noneの場合はTrainモード（Many-to-Many学習）として動作する。
            max_len (int): 系列最大長
            image_features (torch.Tensor, optional): アイテムIDに対応する画像特徴量
            text_features (torch.Tensor, optional): アイテムIDに対応するテキスト特徴量
            pad_token (int): パディングに使用するID (Loss計算時に無視されるIDと同じである必要がある)
        """
        self.user_seqs = user_seqs
        self.targets = targets
        self.max_len = max_len
        self.pad_token = pad_token
        
        # 特徴量はメモリ効率のため参照として保持
        self.image_features = image_features
        self.text_features = text_features

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

        # 画像特徴量 (入力系列に対応するもの)
        if self.image_features is not None:
            # パディング(0)の部分は、特徴量テーブルの0番目(ゼロベクトル想定)が参照される
            item["seq_img_feat"] = self.image_features[input_ids]

        # テキスト特徴量
        if self.text_features is not None:
            item["seq_text_feat"] = self.text_features[input_ids]

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


def create_datasets(data_path, max_len=50, image_feat=None, text_feat=None):
    """
    Many-to-Many 学習用にデータセットを作成するファクトリー関数
    """
    import os
    
    # 1. データの読み込み
    # datamaps.json などが必要ならここで読み込む
    
    # 画像欠損IDの読み込み (オプション)
    missing_image_ids = set()
    missing_path = os.path.join(data_path, 'missing_asins.txt')
    if os.path.exists(missing_path):
        # datamaps.json をロードして変換するロジックが必要ならここに追加
        pass

    # sequential_data.txt の読み込み
    user_ids = []
    sequences = []
    with open(os.path.join(data_path, 'sequential_data.txt'), 'r') as f:
        for line in f:
            line_ids = list(map(int, line.strip().split()))
            # line_ids[0]: UserID, line_ids[1:]: ItemIDs
            sequences.append(line_ids[1:])
            user_ids.append(line_ids[0])

    data_splits = {
        "train": {"seqs": [], "targets": None}, # Trainはtargets=None (自動生成)
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
        image_features=image_feat,
        text_features=text_feat,
        pad_token=0
    )
    
    # Eval (targetsあり)
    datasets["eval"] = SequentialRecDataset(
        user_seqs=data_splits["eval"]["seqs"],
        targets=data_splits["eval"]["targets"],
        max_len=max_len,
        image_features=image_feat,
        text_features=text_feat,
        pad_token=0
    )
    
    # Test (targetsあり)
    datasets["test"] = SequentialRecDataset(
        user_seqs=data_splits["test"]["seqs"],
        targets=data_splits["test"]["targets"],
        max_len=max_len,
        image_features=image_feat,
        text_features=text_feat,
        pad_token=0
    )
        
    return datasets


if __name__ == "__main__":
    # Example usage
    categories = ["toys", "beauty", "sports"]

    datasets = {}

    for category in categories:
        data_dir = f"dataset/{category}"
        max_len = 50
        
        # Optional: Load pre-extracted features (if available)
        # image_feat = torch.load("image_features.pt")  # (num_items + 1, dim)
        # text_feat = torch.load("text_features.pt")    # (num_items + 1, dim)
    
        datasets[category] = create_datasets(data_dir, max_len)
        
        print(f"Category: {category}")
        avg_seq_len = np.mean([len(seq) for seq in datasets[category]['train'].user_seqs])
        print(f"\tAverage sequence length (train): {avg_seq_len:.2f}")
        print(f"\tTrain samples: {len(datasets[category]['train'])}")
        print(f"\tEval samples: {len(datasets[category]['eval'])}")
        print(f"\tTest samples: {len(datasets[category]['test'])}")

        # 4. DataLoader
        train_loader = DataLoader(datasets[category]["train"], batch_size=128, shuffle=True, collate_fn=data_collator)
        test_loader = DataLoader(datasets[category]["test"], batch_size=128, shuffle=False, collate_fn=data_collator)

        print(f"\tNumber of batches (train): {len(train_loader)}")
        for batch in train_loader:
            print(batch["input_ids"].shape)  # (batch_size, max_len)
            print(batch["labels"].shape)   # (batch_size,)
            break