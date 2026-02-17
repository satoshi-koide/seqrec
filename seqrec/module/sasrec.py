import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 補助モジュール (変更なし)
# ==========================================
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_units, hidden_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        output = self.dropout1(self.activation(self.linear1(inputs)))
        output = self.dropout2(self.linear2(output))
        return output + inputs

class SASRecBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(SASRecBlock, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_units)
        self.ln2 = nn.LayerNorm(hidden_units)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_units, 
            num_heads=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.mha_dropout = nn.Dropout(dropout_rate)
        self.ffn = PointWiseFeedForward(hidden_units, dropout_rate)

    def forward(self, inputs, attn_mask):
        attn_output, _ = self.mha(
            query=inputs, key=inputs, value=inputs, attn_mask=attn_mask
        )
        output = self.ln1(inputs + self.mha_dropout(attn_output))
        output = self.ln2(self.ffn(output))
        return output

# ==========================================
# 2. 本体クラス (Trainer対応版)
# ==========================================
class SASRec(nn.Module):
    def __init__(self, config):
        """
        config (dict): {
            "num_items": int,
            "max_len": int,
            "hidden_units": int,
            "num_blocks": int,
            "num_heads": int,
            "dropout_rate": float,
            "use_rating": bool,
            "image_feature_dim": int,
            "text_feature_dim": int
        }
        """
        super().__init__()
        self.config = config # Trainerが参照する場合があるため保持
        
        self.num_items = config['num_items']
        self.hidden_units = config['hidden_units']
        self.max_len = config['max_len']

        # --- Embedding Layers ---
        self.item_emb = nn.Embedding(self.num_items, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, self.hidden_units)
        self.emb_dropout = nn.Dropout(config['dropout_rate'])

        # Side Information
        if config.get('use_rating', False):
            self.rating_emb = nn.Embedding(6, self.hidden_units, padding_idx=0)
        
        if config.get('image_feature_dim', 0) > 0:
            self.img_proj = nn.Linear(config['image_feature_dim'], self.hidden_units)
        
        if config.get('text_feature_dim', 0) > 0:
            self.text_proj = nn.Linear(config['text_feature_dim'], self.hidden_units)

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList([
            SASRecBlock(
                self.hidden_units, 
                config['num_heads'], 
                config['dropout_rate']
            ) for _ in range(config['num_blocks'])
        ])

        # --- Loss Function ---
        # Trainer内でLoss計算を完結させるために保持
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 初期化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, labels=None, image_feats=None, text_feats=None, input_ratings=None, **kwargs):
        """
        Args:
            input_ids: (Batch, SeqLen)
            labels: (Batch) - Trainerから渡される正解ラベル (Optional)
            image_feats, text_feats, input_ratings: (Optional) Side Info
        Returns:
            dict: { "loss": loss_val, "logits": logits_tensor }
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # 1. Embedding
        seqs = self.item_emb(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        seqs += self.pos_emb(positions)

        # Side Information Integration
        if input_ratings is not None and hasattr(self, 'rating_emb'):
            seqs += self.rating_emb(input_ratings)
        if image_feats is not None and hasattr(self, 'img_proj'):
            seqs += self.img_proj(image_feats)
        if text_feats is not None and hasattr(self, 'text_proj'):
            seqs += self.text_proj(text_feats)

        seqs = self.emb_dropout(seqs)

        # 2. Mask Creation (Causal Mask)
        # 未来を見ないマスク
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)

        # 3. Transformer Forward
        for block in self.blocks:
            seqs = block(seqs, attn_mask)

        # 4. Prediction Head
        # 系列の最後のステップのみを使用 (Many-to-One)
        last_hidden = seqs[:, -1, :] # (Batch, Hidden)
        
        # アイテム埋め込み行列との内積で全アイテムのスコアを計算
        logits = torch.matmul(last_hidden, self.item_emb.weight.transpose(0, 1)) # (Batch, NumItems)

        # 5. Output Construction
        outputs = {"logits": logits}

        # labels が渡された場合は Loss を計算して辞書に含める
        if labels is not None:
            loss = self.criterion(logits, labels)
            outputs["loss"] = loss

        return outputs