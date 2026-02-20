import abc
from typing import List, Union, Dict, Optional
from PIL import Image
from seqrec.dataset import Item, ItemDataset
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. Encoder Abstractions (API依存の隠蔽)
# ==========================================

class AbstractTextEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abc.abstractmethod
    def forward(self, texts: List[str]) -> torch.Tensor:
        """文字列のリストを受け取り、(batch_size, dim) のテンソルを返す"""
        pass

class AbstractImageEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abc.abstractmethod
    def forward(self, images: List[Optional[Image.Image]]) -> torch.Tensor:
        """PIL画像のリスト（None許容）を受け取り、(batch_size, dim) のテンソルを返す"""
        pass

# --- 具象エンコーダ (外部ライブラリのラッパー) ---

class SentenceTransformerEncoder(AbstractTextEncoder):
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def get_output_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        # Noneを空文字に変換してSentenceTransformerに渡す
        valid_texts = [t if t is not None else "" for t in texts]
        inputs = self.model.tokenize(valid_texts)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(inputs)
        return outputs['sentence_embedding']

class ViTImageEncoder(AbstractImageEncoder):
    def __init__(self, model_name: str = 'google/vit-base-patch16-224-in21k'):
        super().__init__()
        from transformers import ViTModel, ViTImageProcessor
        from seqrec.utils import get_optimal_attention_config
        
        attn_config = get_optimal_attention_config()
        self.processor = ViTImageProcessor.from_pretrained(model_name, **attn_config)
        self.model = ViTModel.from_pretrained(model_name)
        self.missing_image_embedding = nn.Parameter(torch.zeros(self.model.config.hidden_size))

    def get_output_dim(self) -> int:
        return self.model.config.hidden_size

    def forward(self, images: List[Optional[Image.Image]]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_size = len(images)
        
        valid_images = [img for img in images if img is not None]
        valid_indices = [i for i, img in enumerate(images) if img is not None]

        # 初期値として欠損用エンベディングを敷き詰める
        features = torch.stack([self.missing_image_embedding] * batch_size).to(device)

        if valid_images:
            inputs = self.processor(images=valid_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            valid_features = outputs.last_hidden_state[:, 0, :]
            
            # 有効な画像の特徴量を元のインデックスに配置
            for valid_idx, feature in zip(valid_indices, valid_features):
                features[valid_idx] = feature
                
        return features

class SiglipImageEncoder(AbstractImageEncoder):
    def __init__(self, model_name: str = 'google/siglip-base-patch16-224'):
        super().__init__()
        from transformers import AutoProcessor, SiglipVisionModel
        from seqrec.utils import get_optimal_attention_config
        
        attn_config = get_optimal_attention_config()
        # SigLIP用のプロセッサとビジョンモデルを読み込む
        self.processor = AutoProcessor.from_pretrained(model_name, **attn_config)
        self.model = SiglipVisionModel.from_pretrained(model_name)
        
        # デバイス混在エラーを避けるため、nn.Parameterやregister_bufferは使わず、
        # 明示的にCPU上のゼロテンソルとして保持しておく
        self.missing_image_fallback = torch.zeros(self.model.config.hidden_size)

    def get_output_dim(self) -> int:
        return self.model.config.hidden_size

    def forward(self, images: List[Optional[Image.Image]]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_size = len(images)
        
        valid_images = [img for img in images if img is not None]
        valid_indices = [i for i, img in enumerate(images) if img is not None]

        # フォールバック用テンソル（CPU）をベースにバッチを作成し、後から目的のデバイスに転送
        features = torch.stack([self.missing_image_fallback] * batch_size).to(device)

        if valid_images:
            inputs = self.processor(images=valid_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 推論
            outputs = self.model(**inputs)
            
            # [重要] CLSトークン([:, 0, :])ではなく、pooler_outputを使用する
            valid_features = outputs.pooler_output
            
            # 有効な画像の特徴量を元のインデックスに配置
            for valid_idx, feature in zip(valid_indices, valid_features):
                features[valid_idx] = feature
                
        return features

# ==========================================
# 2. Feature Extractors (モダリティの分離)
# ==========================================

class AbstractItemFeatureExtractor(nn.Module, abc.ABC):
    @abc.abstractmethod
    def feature_dims(self) -> Dict[str, int]:
        """出力される特徴量の名前と次元数のマッピングを返す"""
        pass

    @abc.abstractmethod
    def forward(self, items: List[Item]) -> Dict[str, torch.Tensor]:
        """Itemのリストを受け取り、特徴量の辞書を返す"""
        pass


class TextFeatureExtractor(AbstractItemFeatureExtractor):
    def __init__(self, encoder: AbstractTextEncoder, feature_key: str = "text_features"):
        super().__init__()
        self.encoder = encoder
        self.feature_key = feature_key

    def feature_dims(self) -> Dict[str, int]:
        return {self.feature_key: self.encoder.get_output_dim()}

    def forward(self, items: List[Item]) -> Dict[str, torch.Tensor]:
        texts = [item.serialize() if item is not None else None for item in items]
        features = self.encoder(texts)
        return {self.feature_key: features}


class ImageFeatureExtractor(AbstractItemFeatureExtractor):
    def __init__(self, encoder: AbstractImageEncoder, feature_key: str = "image_features"):
        super().__init__()
        self.encoder = encoder
        self.feature_key = feature_key

    def _load_images(self, paths: List[Optional[str]]) -> List[Optional[Image.Image]]:
        images = []
        for p in paths:
            if p:
                try:
                    images.append(Image.open(p).convert("RGB"))
                    continue
                except Exception:
                    pass
            images.append(None)
        return images

    def feature_dims(self) -> Dict[str, int]:
        return {self.feature_key: self.encoder.get_output_dim()}

    def forward(self, items: List[Item]) -> Dict[str, torch.Tensor]:
        img_paths = [item.image_path if item is not None else None for item in items]
        images = self._load_images(img_paths)
        features = self.encoder(images)
        return {self.feature_key: features}


# ==========================================
# 3. Composite Extractor (抽出器の合成)
# ==========================================

class CombinedFeatureExtractor(AbstractItemFeatureExtractor):
    def __init__(self, extractors: Dict[str, AbstractItemFeatureExtractor]):
        super().__init__()
        # ModuleDictを使うことで、内部のパラメータが自動的にPyTorchに認識される
        self.extractors = nn.ModuleDict(extractors)

    def feature_dims(self) -> Dict[str, int]:
        dims = {}
        for ext in self.extractors.values():
            dims.update(ext.feature_dims())
        return dims

    def forward(self, items: List[Item]) -> Dict[str, torch.Tensor]:
        output = {}
        for ext in self.extractors.values():
            output.update(ext(items))
        return output


# ==========================================
# 4. Feature Store (抽出器の利用とキャッシュ)
# ==========================================

class ItemFeatureStore(nn.Module):
    def __init__(self, 
                 item_dataset: ItemDataset,
                 feature_extractor: AbstractItemFeatureExtractor,
                 is_trainable: bool = False):
        super().__init__()
        self.item_dataset = item_dataset
        self.extractor = feature_extractor
        self.is_trainable = is_trainable

        if not self.is_trainable:
            for param in self.extractor.parameters():
                param.requires_grad = False

        self.feature_cache = {}
        self.feature_dimensions = self.extractor.feature_dims()
        
        self.fallbacks = {
            key: torch.zeros(dim) for key, dim in self.feature_dimensions.items()
        }

    @torch.no_grad()
    def build_cache(self, batch_size=128, verbose=True, debug=False):
        if self.is_trainable:
            return

        item_ids = list(sorted(self.item_dataset.items.keys()))
        if debug:
            item_ids = item_ids[:100]
            
        iterator = range(0, len(item_ids), batch_size)
        if verbose:
            print(f"Building feature cache for {len(item_ids)} items...")
            iterator = tqdm(iterator)
            
        for idx in iterator:
            ids = item_ids[idx:idx+batch_size]
            batch_items = [self.item_dataset.items[i] for i in ids]
            features_batch = self.extractor(batch_items)
            
            # キー（text_features等）ごとにキャッシュを展開
            for i, item_id in enumerate(ids):
                self.feature_cache[item_id] = {
                    k: v[i].detach().cpu() for k, v in features_batch.items()
                }

    def load_cache(self, path: str):
        data = np.load(path)
        for item_id in self.item_dataset.items.keys():
            self.feature_cache[item_id] = {
                k: torch.tensor(data[str(k)][item_id]) for k in self.feature_dimensions.keys()
            }

    def save_cache(self, path: str, key: str):
        # numpy の行列形式に変換して保存。item_id の配列も保存する。
        np.savez_compressed(path, **{str(k): np.array([v[key].cpu().numpy() for v in self.feature_cache.values()]) 
                                     for k in self.feature_dimensions.keys()})

    def forward(self, item_ids: Union[List[int], torch.Tensor], device=None):
        if isinstance(item_ids, list):
            original_shape = (len(item_ids),)
        else:
            original_shape = item_ids.shape
            item_ids = item_ids.view(-1).tolist()
            
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        if self.is_trainable:
            items = [self.item_dataset.get_item(item_id) for item_id in item_ids]
            features = self.extractor(items)
            return {k: v.view(*original_shape, -1) for k, v in features.items()}
        else:
            output_features = {k: [] for k in self.feature_dimensions.keys()}
            
            for item_id in item_ids:
                if item_id in self.feature_cache:
                    for k in output_features:
                        output_features[k].append(self.feature_cache[item_id][k])
                else:
                    for k in output_features:
                        output_features[k].append(self.fallbacks[k])
            
            return {k: torch.stack(v).to(device).view(*original_shape, -1) 
                    for k, v in output_features.items()}
    
    def feature_dims(self) -> Dict[str, int]:
        return self.feature_dimensions

def initialize_item_feature_store(
    item_dataset: ItemDataset,
    text_model_name: Optional[str] = None, 
    image_model_name: Optional[str] = None,
    is_trainable: bool = False,
    device: Optional[torch.device] = None
) -> ItemFeatureStore:
    """
    指定されたモデル名に基づいてエンコーダと抽出器を動的に構築し、
    ItemFeatureStoreを初期化して返すヘルパー関数。
    """
    image_model_name_to_extractor = {
        'google/siglip-base-patch16-224': SiglipImageEncoder,
        'google/vit-base-patch16-224-in21k': ViTImageEncoder
    }
    text_model_name_to_extractor = {
        'all-mpnet-base-v2': SentenceTransformerEncoder,
        'all-MiniLM-L6-v2': SentenceTransformerEncoder,
        'all-MiniLM-L12-v2': SentenceTransformerEncoder,
    }

    extractors = {}
    
    # 1. テキストモデルの構築
    if text_model_name:
        if text_model_name not in text_model_name_to_extractor:
            raise ValueError(f"Unsupported text model: {text_model_name}")
        text_encoder = text_model_name_to_extractor[text_model_name](model_name=text_model_name)
        extractors["text"] = TextFeatureExtractor(text_encoder, feature_key="text_features")
        
    # 2. 画像モデルの構築
    if image_model_name:
        if image_model_name not in image_model_name_to_extractor:
            raise ValueError(f"Unsupported image model: {image_model_name}")
        image_encoder = image_model_name_to_extractor[image_model_name](model_name=image_model_name)
        extractors["image"] = ImageFeatureExtractor(image_encoder, feature_key="image_features")
        
    # 3. エラーハンドリング
    if not extractors:
        raise ValueError("At least one of text_model_name or image_model_name must be specified.")
        
    # 4. 抽出器の合成（1つだけならそれをそのまま使い、複数ならCombinedで束ねる）
    if len(extractors) == 1:
        feature_extractor = list(extractors.values())[0]
    else:
        feature_extractor = CombinedFeatureExtractor(extractors)
        
    # 5. ストアの初期化と返却
    store = ItemFeatureStore(
        item_dataset=item_dataset,
        feature_extractor=feature_extractor,
        is_trainable=is_trainable
    )
    if device is not None:
        store.to(device)
    return store