from typing import List, Union
from PIL import Image
from seqrec.dataset import Item, ItemDataset
from tqdm import tqdm

import torch
import torch.nn as nn

from transformers import ViTModel, ViTImageProcessor
from sentence_transformers import SentenceTransformer

class ItemFeatureExtractor(nn.Module):
    def __init__(self, 
                 text_model_name: str = 'all-mpnet-base-v2',
                 image_model_name: str = 'google/vit-base-patch16-224-in21k',
                 cache_images: bool = False):
        super().__init__()
        
        # --- Text Module ---
        if text_model_name:
            self.text_model = SentenceTransformer(text_model_name)
        else:
            self.text_model = None
        
        # --- Image Module ---
        if image_model_name:
            self.img_processor = ViTImageProcessor.from_pretrained(image_model_name)
            self.img_model = ViTModel.from_pretrained(image_model_name)
            self.missing_image_embedding = nn.Parameter(torch.zeros(self.img_model.config.hidden_size))
        else:
            self.img_processor = None
            self.img_model = None
            self.missing_image_embedding = None

        self.cache_images = cache_images
        if self.cache_images and self.img_model:
            self.image_cache = {}

        feature_dims = self.feature_dims()
        self.fallback_text = torch.zeros(feature_dims["text_feature_dim"])   # should be CPU tensor
        self.fallback_image = torch.zeros(feature_dims["image_feature_dim"]) # should be CPU tensor

    def feature_dims(self):
        text_dim = self.text_model.get_sentence_embedding_dimension() if self.text_model else 0
        image_dim = self.img_model.config.hidden_size if self.img_model else 0
        return {"text_feature_dim": text_dim, "image_feature_dim": image_dim}

    def _load_images(self, paths: List[str]) -> List[Image.Image]:
        """画像パスからPIL Imageを読み込む内部ヘルパー"""
        images = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                images.append(None)
        return images

    def forward(self, items: List[Item]):
        """
        Args:
            items: Itemオブジェクトのリスト
        Returns:
            text_features: (batch_size, text_dim)
            image_features: (batch_size, image_dim)
        """
        # データ展開
        texts = [None if item is None else item.serialize() for item in items]
        img_paths = [None if item is None else item.image_path for item in items]
        
        device = next(self.parameters()).device

        # -------------------------------------------------------
        # 1. Text Feature Extraction
        # -------------------------------------------------------
        # SentenceTransformerの tokenize -> forward を手動で行い、勾配計算の道を残す
        valid_index = [i for i, t in enumerate(texts) if t is not None]
        text_features = torch.zeros((len(items), self.feature_dims()["text_feature_dim"]), device=device)
        if self.text_model and valid_index:
            valid_texts = [texts[i] for i in valid_index]
            text_inputs = self.text_model.tokenize(valid_texts)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            text_outputs = self.text_model(text_inputs)
            text_features[valid_index] = text_outputs['sentence_embedding']

        # -------------------------------------------------------
        # 2. Image Feature Extraction
        # -------------------------------------------------------
        pil_images = self._load_images(img_paths)

        # pil_images 内の None を missing_image_embedding で置き換えるためのマスク
        if self.img_processor and self.img_model:
            valid_images = [img for img in pil_images if img is not None]
            valid_index = [i for i, img in enumerate(pil_images) if img is not None]
        
        img_inputs = self.img_processor(images=valid_images, return_tensors="pt") # これは何度も呼ばれている。キャッシュできるかも。ただ data augmentation で毎回違う画像を入れる可能性もあるので要検討。
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        
        img_outputs = self.img_model(**img_inputs)
        
        # Using ViT [CLS] token (index 0)
        # last_hidden_state: (batch, seq_len, hidden_size)
        valid_image_features = img_outputs.last_hidden_state[:, 0, :]

        # None だった画像の特徴を missing_image_embedding に置き換える
        image_features = torch.stack([self.missing_image_embedding if img is None else valid_image_features[valid_index.index(i)] for i, img in enumerate(pil_images)], dim=0).to(device)

        return {"text_features": text_features, "image_features": image_features}



class ItemFeatureStore(ItemFeatureExtractor):
    def __init__(self, 
                 item_dataset: ItemDataset,
                 is_trainable: bool = False,
                 text_model_name: str = 'all-mpnet-base-v2',
                 image_model_name: str = 'google/vit-base-patch16-224-in21k'):
        super().__init__(text_model_name, image_model_name)

        self.item_dataset = item_dataset
        self.is_trainable = is_trainable

        if not self.is_trainable:
            # set requires_grad to False for all parameters
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.img_model.parameters():
                param.requires_grad = False

        self.feature_cache = {}

        # self.register_buffer("fallback_text", fallback_text)
        # self.register_buffer("fallback_image", fallback_image)

    def build_cache(self, batch_size=128, verbose=True, debug=False):
        if self.is_trainable:
            return

        with torch.no_grad():
            item_ids = list(sorted(self.item_dataset.items.keys()))
            if debug:
                item_ids = item_ids[:100]  # デバッグ用に最初の100アイテムだけ処理
            iterator = range(0, len(item_ids), batch_size)
            if verbose:
                print(f"Building feature cache for {len(item_ids)} items...")
                iterator = tqdm(iterator)
            for idx in iterator:
                ids = item_ids[idx:idx+batch_size]
                batch_items = [self.item_dataset.items[i] for i in ids]
                features = super().forward(batch_items)
                for item_id, text_feat, img_feat in zip(ids, features["text_features"], features["image_features"]):
                    self.feature_cache[item_id] = {
                        "text_features": text_feat.detach().cpu(),
                        "image_features": img_feat.detach().cpu(),
                    }
        
    def forward(self, item_ids: Union[List[int], torch.Tensor]):
        text_feats = []
        image_feats = []
        if isinstance(item_ids, list):
            original_shape = (len(item_ids),)
        else:
            original_shape = item_ids.shape
            item_ids = item_ids.view(-1).tolist()  # Flatten to 1D

        if self.is_trainable:
            items =[self.item_dataset[item_id] for item_id in item_ids]
            features = super().forward(items)
            text_feats = features["text_features"].view(*original_shape, -1)
            image_feats = features["image_features"].view(*original_shape, -1)
        else:
            for item_id in item_ids:
                if item_id in self.feature_cache:
                    text_feats.append(self.feature_cache[item_id]["text_features"])
                    image_feats.append(self.feature_cache[item_id]["image_features"])
                else:
                    text_feats.append(self.fallback_text)
                    image_feats.append(self.fallback_image)
            text_feats = torch.stack(text_feats).to(next(self.parameters()).device).view(*original_shape, -1)
            image_feats = torch.stack(image_feats).to(next(self.parameters()).device).view(*original_shape, -1)
        
        return {"text_features": text_feats, "image_features": image_feats}

    
if __name__ == "__main__":
    from seqrec.dataset import create_datasets

    categories = ['toys', 'sports', 'beauty']

    for category in categories:
        dataset_path = f"dataset/{category}"
        datasets, item_dataset = create_datasets(dataset_path)

        # feature_extractor = ItemFeatureExtractor().to("cuda")
        # item = [item_dataset.items[1], item_dataset.items[2]]  # 最初のアイテムを取得
        # features = feature_extractor(item)
        # print(f"Features for first item in {category}:")
        # print(f"Text Features Shape: {features['text_features'].shape}")
        # print(f"Image Features Shape: {features['image_features'].shape}")

        feature_store = ItemFeatureStore(item_dataset).to("cuda")
        feature_store.build_cache(batch_size=64, debug=True)  # キャッシュ構築（GPUで実行）
        item_ids = list(item_dataset.items.keys())[:5]  # 最初の5アイテムのIDを取得
        features = feature_store(item_ids)
        print(f"Text features for first 5 items in {category}: {features['text_features'].shape}")
        print(f"Image features for first 5 items in {category}: {features['image_features'].shape}")
