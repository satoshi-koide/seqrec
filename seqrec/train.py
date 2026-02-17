import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, EvalPrediction

from dataset import create_datasets, data_collator
from module.sasrec import SASRec, SASRecConfig

def get_num_items(data_path):
    with open(f'{data_path}/datamaps.json', 'r') as f:
        datamaps = json.load(f)
    num_actual_items = len(datamaps['item2id'])
    return num_actual_items + 1

def preprocess_logits_for_metrics(logits, labels):
    """
    【高速化の肝】
    評価時、GPU上で Logits (Batch, SeqLen, NumItems) から
    最後のステップ (Batch, NumItems) だけを抜き出して CPU に送る。
    これによりデータ転送量を 1/SeqLen に削減する。
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # [Batch, SeqLen, NumItems] -> [Batch, NumItems]
    # 最後のタイムステップ (Next Item Prediction) だけが必要
    last_item_logits = logits[:, -1, :]
    
    return last_item_logits

def compute_metrics(eval_pred: EvalPrediction):
    """
    バリデーション時の精度計算 (Recall@10, NDCG@10)
    preprocess_logits_for_metrics で軽量化された logits を受け取る
    """
    # logits: (Batch, NumItems)  <- preprocess済み
    # labels: (Batch, SeqLen)    <- Datasetからそのまま来る
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # ラベルも最後のステップだけを取り出す必要がある
    # labels が (Batch, SeqLen) の場合、最後の列を取得
    if labels.ndim > 1:
        # Many-to-Many Datasetでは、eval/test時は最後のステップ以外は0(pad)になっているはず
        # なので単純に -1 を取れば正解ラベルになる
        labels = labels[:, -1]

    # --- Top-K 抽出 (高速化のため argpartition 推奨) ---
    k = 10
    batch_size = logits.shape[0]
    
    # 上位K個のインデックスを取得 (順不同)
    # 軸1(アイテム次元)に対してトップ10を探す
    topk_ind = np.argpartition(logits, -k, axis=1)[:, -k:]
    
    # 上位K個の値を取得してソート (NDCGのために順序が必要)
    rows = np.arange(batch_size)[:, None]
    topk_vals = logits[rows, topk_ind]
    sort_ind = np.argsort(topk_vals, axis=1)[:, ::-1] # 降順ソート
    topk_sorted = topk_ind[rows, sort_ind] # (Batch, K) 正しい順序のアイテムID

    # --- Recall@K ---
    # labels (Batch,) を (Batch, 1) にしてブロードキャスト比較
    # label が topk_sorted に含まれていれば 1, なければ 0
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
    # ヒットしなかったユーザーは 0
    ndcg_vals = np.zeros(batch_size)
    ndcg_vals[hit_rows] = 1.0 / np.log2(hit_cols + 2)
    ndcg = ndcg_vals.mean()

    return {
        "recall_at_10": recall,
        "ndcg_at_10": ndcg
    }

def main(dataset_path: str = "dataset/toys"):

    datasets = create_datasets(dataset_path)
    num_actual_items = get_num_items(dataset_path)

    model_config = SASRecConfig(
        num_items=num_actual_items,
        max_len=50,
        hidden_units=64,
        dropout_rate=0.2,
        use_rating=False,
        image_feature_dim=0,
        text_feature_dim=0,
        num_blocks=3,
        num_heads=2,
    )

    training_args = TrainingArguments(
        output_dir="./output_sasrec",
        
        # 学習設定
        num_train_epochs=30,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128, # 注意: メモリ圧迫する場合は下げる
        learning_rate=1e-3,
        weight_decay=0.01,
        
        # 評価設定
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="ndcg_at_10",
        greater_is_better=True,
        
        # 高速化・ログ
        fp16=torch.cuda.is_available(), # GPUがあればFP16有効化
        logging_dir='./logs',
        logging_steps=50,
        dataloader_num_workers=4, # データローダーの並列数
    )    

    # 1. モデル初期化
    model = SASRec(model_config)

    # 2. Trainerに直接渡す
    trainer = Trainer(
        model=model,  # ラッパー不要！
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
    
if __name__ == "__main__":
    main()