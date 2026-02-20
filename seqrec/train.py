import json
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, EvalPrediction

from seqrec.dataset import create_datasets, SequentialRecDataset
from seqrec.module.sasrec import SASRec, SASRecConfig
from seqrec.module.feature_extractor import initialize_item_feature_store

def get_num_items(data_path):
    with open(f'{data_path}/datamaps.json', 'r') as f:
        datamaps = json.load(f)
    num_actual_items = len(datamaps['item2id'])
    return num_actual_items + 1

def preprocess_logits_for_metrics(logits, labels):
    """
    【変更点】
    Logits と Labels の両方をここで「最後のステップ」だけに絞り込みます。
    これにより、compute_metrics はパディングや系列長を気にする必要がなくなります。
    """
    if isinstance(logits, tuple):
        # モデルによっては (logits, hidden_states) などを返す場合があるため
        logits = logits[0]
    
    # --- Logits の処理 ---
    # [Batch, SeqLen, NumItems] -> [Batch, NumItems]
    # Left Padding なので、常に最後の要素 (-1) が予測すべき位置です
    last_item_logits = logits[:, -1, :]
    
    # --- Labels の処理 ---
    # [Batch, SeqLen] -> [Batch]
    # Labels も同様に最後の要素だけが必要です。
    # Dynamic Padding で他が -100 になっていても、予測対象は常に末尾です。
    last_item_labels = labels[:, -1]
    
    # ペアで返すと、Trainer はこれを numpy に変換して compute_metrics に渡してくれます
    return last_item_logits, last_item_labels

def compute_metrics(eval_pred: EvalPrediction):
    """
    preprocess_logits_for_metrics の戻り値が
    eval_pred.predictions と eval_pred.label_ids に入ってきます。
    """
    # preprocess で抽出済みなので、形状は以下の通り
    # logits: (Batch, NumItems)
    # labels: (Batch,) 
    logits, labels = eval_pred.predictions
    
    # --- 以降のロジックは以前と同じですが、ラベルのスライス処理が不要になります ---

    # ラベルのスライス処理 (labels[:, -1]) は削除！
    # preprocess で既に (Batch,) になっているためそのまま使えます。

    # --- Top-K 抽出 (ここからは元のロジックと同じ) ---
    k = 10
    batch_size = logits.shape[0]
    
    # 1. 上位K個のインデックスを取得 (順不同)
    # argpartition は全ソートしないので高速
    topk_ind = np.argpartition(logits, -k, axis=1)[:, -k:]
    
    # 2. 上位K個の値を取得してソート (NDCGのために順序が必要)
    rows = np.arange(batch_size)[:, None]
    topk_vals = logits[rows, topk_ind]
    
    # 値に基づいて降順ソートするためのインデックス
    sort_ind = np.argsort(topk_vals, axis=1)[:, ::-1]
    
    # 正しい順序に並び替えたアイテムID (Batch, K)
    topk_sorted = topk_ind[rows, sort_ind]

    # --- Recall@K ---
    # labels (Batch,) を (Batch, 1) にしてブロードキャスト比較
    # label が topk_sorted に含まれていれば True
    hit = (topk_sorted == labels[:, None]) # (Batch, K) bool
    
    # 各ユーザーについてHitしたか (1 or 0)
    recall = hit.sum() / batch_size

    # --- NDCG@K ---
    # Hitした位置 (rank) を探す
    # np.where は (row_indices, col_indices) を返す
    # hit は各行に最大1個しかTrueがない前提 (Next Item Prediction)
    hit_rows, hit_cols = np.where(hit)
    
    # rank = hit_cols (0-indexed: 0位, 1位...)
    # ndcg = 1 / log2(rank + 2)  (rank+1位なので log2(rank+1+1))
    ndcg_vals = np.zeros(batch_size)
    ndcg_vals[hit_rows] = 1.0 / np.log2(hit_cols + 2)
    ndcg = ndcg_vals.mean()

    return {
        "recall_at_10": recall,
        "ndcg_at_10": ndcg
    }

def main(dataset_path: str, feature_extractor: str = 'none', text_model_name: str = None, image_model_name: str = None):

    datasets, item_dataset = create_datasets(dataset_path)
    num_actual_items = get_num_items(dataset_path)
    data_collator = SequentialRecDataset.data_collator

    model_config = SASRecConfig(
        num_items=num_actual_items,
        max_len=50,
        hidden_units=64,
        dropout_rate=0.2, # The original paper uses 0.5 for sparse datasets
        use_rating=False,
        num_blocks=2,
        num_heads=1,   # The original paper recommends 1 head
        feature_extractor=feature_extractor,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
    )

    batch_size = 128 if feature_extractor != "trainable" else 20
    training_args = TrainingArguments(
        output_dir="./output_sasrec",
        
        # 学習設定
        num_train_epochs=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=128, # 注意: メモリ圧迫する場合は下げる
        learning_rate=1e-3 * batch_size / 128, # バッチサイズに応じた学習率スケーリング
        weight_decay=0.01,
        
        # 評価設定
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="ndcg_at_10",
        greater_is_better=True,
        
        # 高速化・ログ
        #fp16=torch.cuda.is_available(), # GPUがあればFP16有効化
        bf16=torch.cuda.is_available(), # GPUがあればBF16有効化
        logging_dir='./logs',
        logging_steps=50,
        dataloader_num_workers=4, # データローダーの並列数
    )    

    # 1. モデル初期化
    if model_config.text_model_name or model_config.image_model_name:
        is_trainable = feature_extractor == "trainable"
        feature_store = initialize_item_feature_store(
            item_dataset, 
            text_model_name=model_config.text_model_name,
            image_model_name=model_config.image_model_name,
            is_trainable=is_trainable,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not is_trainable:
            feature_store.build_cache(batch_size=128, verbose=True)  # キャッシュ構築
    else:
        feature_store = None

    model = SASRec(model_config, feature_store=feature_store)
    # 2. Trainerに直接渡す
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    # 3. ベストモデルでテスト評価
    test_metrics = trainer.evaluate(datasets["test"])
    print("Test Metrics:", test_metrics)
    
if __name__ == "__main__":
    categories = ['toys', 'sports', 'beauty']

    #text_model_name=None
    text_model_name="all-mpnet-base-v2"
    for category in categories:
        for image_model_name in ["google/vit-base-patch16-224-in21k", None]:
            print(f"=== Training on {category} category ===")
            dataset_path = f"dataset/{category}"

            #text_model_name="all_mpnet_base_v2",

            main(
                dataset_path,
                feature_extractor="frozen",
                text_model_name=text_model_name,
                image_model_name=image_model_name,
            )
        #main(dataset_path, feature_extractor="trainable")
