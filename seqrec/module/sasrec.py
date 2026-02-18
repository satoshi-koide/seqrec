from typing import Literal
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from seqrec.module.feature_extractor import ItemFeatureStore

# ==========================================
# 1. Configuration Class
# ==========================================
class SASRecConfig(PretrainedConfig):
    model_type = "sasrec"

    def __init__(
        self,
        num_items=50000,
        max_len=50,
        hidden_units=64,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2,
        use_rating=False,
        pad_token_id=0,
        feature_extractor: Literal["none", "frozen", "trainable"] = None,
        **kwargs
    ):
        """
        Args:
            num_items (int): Vocabulary size (items + padding).
            max_len (int): Maximum sequence length.
            hidden_units (int): Embedding dimension.
            num_blocks (int): Number of Transformer blocks.
            num_heads (int): Number of Attention heads.
            dropout_rate (float): Dropout probability.
            use_rating (bool): Whether to use rating embeddings.
            pad_token_id (int): ID used for padding.
        """
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_rating = use_rating
        self.feature_extractor = feature_extractor
        super().__init__(pad_token_id=pad_token_id, **kwargs)


# ==========================================
# 2. Helper Modules (Layers)
# ==========================================
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(hidden_units, hidden_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU() # GELU is also an option
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        # inputs: (Batch, SeqLen, Hidden)
        output = self.dropout1(self.activation(self.linear1(inputs)))
        output = self.dropout2(self.linear2(output))
        return output + inputs # Residual Connection


class SASRecLayer(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_units)
        self.ln2 = nn.LayerNorm(hidden_units)
        
        # batch_first=True corresponds to (Batch, Seq, Feature)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_units, 
            num_heads=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.mha_dropout = nn.Dropout(dropout_rate)
        self.ffn = PointWiseFeedForward(hidden_units, dropout_rate)

    def forward(self, inputs, attn_mask):
        # 1. LayerNorm before Attention (Pre-Norm style is more stable, 
        #    but Original SASRec uses Post-Norm. Here we use Post-Norm structure)
        
        # Self-Attention
        # query=inputs, key=inputs, value=inputs
        attn_output, _ = self.mha(
            query=inputs, 
            key=inputs, 
            value=inputs, 
            attn_mask=attn_mask
        )
        # Residual + Norm
        output = self.ln1(inputs + self.mha_dropout(attn_output))
        
        # Feed Forward + Residual + Norm
        output = self.ln2(self.ffn(output))
        
        return output


# ==========================================
# 3. Main Model (PretrainedModel)
# ==========================================
class SASRec(PreTrainedModel):
    config_class = SASRecConfig

    def __init__(self, config, feature_store: ItemFeatureStore=None):
        super().__init__(config)

        if config.feature_extractor != "none" and feature_store is None:
            raise ValueError("feature_store must be provided if feature_extractor is enabled in config.")
        if config.feature_extractor == "frozen" and feature_store.is_trainable:
            raise ValueError("feature_store must be frozen if feature_extractor is set to 'frozen'.")
        if config.feature_extractor == "trainable" and not feature_store.is_trainable:
            raise ValueError("feature_store must be trainable if feature_extractor is set to 'trainable'.")

        self.num_items = config.num_items
        self.hidden_units = config.hidden_units
        self.max_len = config.max_len

        # --- Embeddings ---
        self.item_emb = nn.Embedding(self.num_items, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, self.hidden_units)
        self.emb_dropout = nn.Dropout(config.dropout_rate)

        # --- Side Information ---
        if config.use_rating:
            # 1-5 stars + 0 padding = 6
            self.rating_emb = nn.Embedding(6, self.hidden_units, padding_idx=0)
        
        self.feature_store = feature_store
        if self.feature_store:
            feature_dims = self.feature_store.feature_dims()
            if feature_dims["image_feature_dim"] > 0:
                self.img_proj = nn.Linear(feature_dims["image_feature_dim"], self.hidden_units)
            else:
                self.img_proj = None
            if feature_dims["text_feature_dim"] > 0:
                self.text_proj = nn.Linear(feature_dims["text_feature_dim"], self.hidden_units)
            else:
                self.text_proj = None

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList([
            SASRecLayer(
                self.hidden_units, 
                config.num_heads, 
                config.dropout_rate
            ) for _ in range(config.num_blocks)
        ])
        
        self.last_layernorm = nn.LayerNorm(self.hidden_units)

        # --- Loss Function ---
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """ Initialize the weights (Automated by PretrainedModel) """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self, 
        input_ids, 
        labels=None, 
        input_ratings=None, 
        **kwargs
    ):
        """
        Args:
            input_ids: (Batch, SeqLen) - Item history
            labels: (Batch, SeqLen) - Target items (Next item at each position)
            image_feats: (Batch, SeqLen, ImgDim) - Optional
            text_feats: (Batch, SeqLen, TxtDim) - Optional
            input_ratings: (Batch, SeqLen) - Optional
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]

        # 1. Embedding
        seqs = self.item_emb(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        seqs += self.pos_emb(positions)

        # Side Information Integration (Add)
        if input_ratings is not None and hasattr(self, 'rating_emb'):
            seqs += self.rating_emb(input_ratings)

        if self.feature_store:
            features = self.feature_store(input_ids)
            if self.img_proj is not None:
                # Projection: (B, L, ImgDim) -> (B, L, Hidden)
                seqs += self.img_proj(features["image_features"])
            if self.text_proj is not None:
                # Projection: (B, L, TxtDim) -> (B, L, Hidden)
                seqs += self.text_proj(features["text_features"])

        seqs = self.emb_dropout(seqs)

        # 2. Causal Masking (Look-ahead mask)
        # (SeqLen, SeqLen) - Upper triangular is -inf
        attn_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device) * float('-inf'), 
            diagonal=1
        )

        # 3. Transformer Encoder
        # Pass through all blocks
        for block in self.blocks:
            seqs = block(seqs, attn_mask)
        
        seqs = self.last_layernorm(seqs)

        # 4. Prediction Head
        # Many-to-Many: Compute logits for ALL positions
        # Weight tying: Use item embedding weight as output layer
        # (Batch, SeqLen, Hidden) @ (NumItems, Hidden)^T -> (Batch, SeqLen, NumItems)
        logits = torch.matmul(seqs, self.item_emb.weight.transpose(0, 1))

        loss = None
        if labels is not None:
            loss = self.criterion(logits.view(-1, self.num_items), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=1):
        """
        推論用 (Autoregressive Generation)
        Semantic IDの場合は max_new_tokens > 1 になる
        """
        self.eval()
        
        # 現在の系列のコピーを作成
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # forwardを呼ぶ
            outputs = self.forward(generated)
            logits = outputs["logits"]
            
            # 最後のステップのロジットを取得
            next_token_logits = logits[:, -1, :]
            
            # Greedy Decoding (一番確率が高いものを選ぶ)
            # ※ 必要ならここで Sampling や Beam Search を実装する
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 生成されたトークンを系列に追加
            generated = torch.cat([generated, next_token], dim=1)
            
            # Semantic IDの場合、ここで「アイテム終端トークン」が出たら打ち切る等の処理が入る
        
        # 入力部分を除いた「生成された部分」だけ返す
        return generated[:, input_ids.shape[1]:]