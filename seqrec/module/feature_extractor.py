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
        pass

class AbstractImageEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abc.abstractmethod
    def forward(self, images: List[Optional[Image.Image]]) -> torch.Tensor:
        pass

# --- 具象エンコーダ (外部ライブラリのラッパー) ---
# ※中身は元のコードと同一のため省略せずにそのまま記載します

class SentenceTransformerEncoder(AbstractTextEncoder):
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        from seqrec.utils import get_optimal_attention_config
        
        attn_config = get_optimal_attention_config()
        self.model = SentenceTransformer(model_name, model_kwargs=attn_config)

    def get_output_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
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
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name, **attn_config)
        self.missing_image_embedding = nn.Parameter(torch.zeros(self.model.config.hidden_size))

    def get_output_dim(self) -> int:
        return self.model.config.hidden_size

    def forward(self, images: List[Optional[Image.Image]]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_size = len(images)
        
        valid_images = [img for img in images if img is not None]
        valid_indices = [i for i, img in enumerate(images) if img is not None]

        features = torch.stack([self.missing_image_embedding] * batch_size).to(device)

        if valid_images:
            inputs = self.processor(images=valid_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            valid_features = outputs.last_hidden_state[:, 0, :]
            
            for valid_idx, feature in zip(valid_indices, valid_features):
                features[valid_idx] = feature
                
        return features

class SiglipImageEncoder(AbstractImageEncoder):
    def __init__(self, model_name: str = 'google/siglip-base-patch16-224'):
        super().__init__()
        from transformers import SiglipImageProcessor, SiglipVisionModel
        from seqrec.utils import get_optimal_attention_config
        
        attn_config = get_optimal_attention_config()
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(model_name, **attn_config)
        self.missing_image_fallback = torch.zeros(self.model.config.hidden_size)

    def get_output_dim(self) -> int:
        return self.model.config.hidden_size

    def forward(self, images: List[Optional[Image.Image]]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_size = len(images)
        
        valid_images = [img for img in images if img is not None]
        valid_indices = [i for i, img in enumerate(images) if img is not None]

        features = torch.stack([self.missing_image_fallback] * batch_size).to(device)

        if valid_images:
            inputs = self.processor(images=valid_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            valid_features = outputs.pooler_output
            
            for valid_idx, feature in zip(valid_indices, valid_features):
                features[valid_idx] = feature
                
        return features

# ==========================================
# 2. Feature Extractors (モダリティの分離とキャッシュ構築)
# ==========================================

class AbstractItemFeatureExtractor(nn.Module, abc.ABC):
    @abc.abstractmethod
    def feature_dims(self) -> Dict[str, int]:
        pass

    @abc.abstractmethod
    def forward(self, items: List[Item]) -> Dict[str, torch.Tensor]:
        pass

    @torch.no_grad()
    def build_cache(self, items: List[Item], batch_size: int = 128, save_to: str = None, verbose: bool = True, debug: bool = False) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        List[Item] から特徴量を一括抽出し、アイテムIDをキーとしたキャッシュ辞書を作成する。
        """
        if debug:
            items = items[:100]
            
        feature_cache = {}
        iterator = range(0, len(items), batch_size)
        if verbose:
            print(f"Building feature cache for {len(items)} items...")
            from tqdm import tqdm
            iterator = tqdm(iterator)
            
        for idx in iterator:
            batch_items = items[idx:idx+batch_size]
            features_batch = self(batch_items)
            
            for i, item in enumerate(batch_items):
                item_id = item.item_id 
                feature_cache[item_id] = {
                    k: v[i].detach().cpu() for k, v in features_batch.items()
                }
                
        if save_to:
            self.save_cache(feature_cache, save_to)

        return feature_cache

    def save_cache(self, feature_cache: Dict[int, Dict[str, torch.Tensor]], save_paths: Dict[str, str]):
        """
        build_cache で生成したキャッシュ辞書を .npz ファイルとして保存する。
        
        引数例:
        save_paths = {
            "text_features": "./cache/text_features.npz",
            "image_features": "./cache/image_features.npz"
        }
        """
        valid_keys = self.feature_dims().keys()
        
        # item_id の順序を保証して保存
        sorted_item_ids = sorted(feature_cache.keys())
        
        for key, path in save_paths.items():
            if key not in valid_keys:
                print(f"Warning: Key '{key}' not found in extractor's feature dimensions. Skipping.")
                continue
                
            # 指定されたキーの特徴量だけを抽出し、numpy配列に変換
            features_list = [feature_cache[item_id][key].float().cpu().numpy() for item_id in sorted_item_ids]
            
            # 特徴量の配列と、対応する item_id の配列を一緒に保存
            np.savez_compressed(
                path, 
                **{key: np.array(features_list), "item_ids": np.array(sorted_item_ids)}
            )

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


class CombinedFeatureExtractor(AbstractItemFeatureExtractor):
    def __init__(self, extractors: Dict[str, AbstractItemFeatureExtractor]):
        super().__init__()
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
# 4. Feature Store (役割に応じた分割)
# ==========================================

class AbstractItemFeatureStore(nn.Module, abc.ABC):
    @abc.abstractmethod
    def feature_dims(self) -> Dict[str, int]:
        pass

    @abc.abstractmethod
    def forward(self, item_ids: Union[List[int], torch.Tensor], device=None) -> Dict[str, torch.Tensor]:
        pass


class CachedItemFeatureStore(AbstractItemFeatureStore):
    """
    抽出済みのキャッシュ(.npzファイル)から特徴量をロードして返すためのストア。
    初期化時にファイルを読み込み、メモリ上に保持する。
    """
    def __init__(self, npz_paths: Dict[str, str]):
        '''
        npz_paths: 特徴量の名前をキー、対応する.npzファイルのパスを値とする辞書。E.g., {"text_features": "path/to/text_features.npz", "image_features": "path/to/image_features.npz"}
        '''
        super().__init__()
        self.feature_cache = {}
        self.feature_dimensions = {}
        
        for key, path in npz_paths.items():
            data = np.load(path)
            features = data[key]
            
            # ロードした配列から次元数を自動推論
            self.feature_dimensions[key] = features.shape[-1]
            
            # .npz に item_ids が含まれていればそれを使用し、なければインデックスを使用
            item_ids = data['item_ids'] if 'item_ids' in data else range(len(features))
            
            for item_id, feat in zip(item_ids, features):
                # numpy の int/int64 を標準の int にキャスト
                item_id = int(item_id)
                if item_id not in self.feature_cache:
                    self.feature_cache[item_id] = {}
                self.feature_cache[item_id][key] = torch.tensor(feat)
                
        self.fallbacks = {
            key: torch.zeros(dim) for key, dim in self.feature_dimensions.items()
        }

    def feature_dims(self) -> Dict[str, int]:
        return self.feature_dimensions

    def forward(self, item_ids: Union[List[int], torch.Tensor], device=None) -> Dict[str, torch.Tensor]:
        if isinstance(item_ids, list):
            original_shape = (len(item_ids),)
        else:
            original_shape = item_ids.shape
            item_ids = item_ids.view(-1).tolist()

        if device is None:
            device = torch.device("cpu")

        output_features = {k: [] for k in self.feature_dimensions.keys()}
        
        for item_id in item_ids:
            if item_id in self.feature_cache:
                for k in output_features:
                    output_features[k].append(self.feature_cache[item_id].get(k, self.fallbacks[k]))
            else:
                for k in output_features:
                    output_features[k].append(self.fallbacks[k])
        
        return {k: torch.stack(v).to(device).view(*original_shape, -1) 
                for k, v in output_features.items()}


class TrainableItemFeatureStore(AbstractItemFeatureStore):
    """
    都度 FeatureExtractor を実行し、勾配を流すことができるストア。
    """
    def __init__(self, item_dataset: ItemDataset, feature_extractor: AbstractItemFeatureExtractor):
        super().__init__()
        self.item_dataset = item_dataset
        self.extractor = feature_extractor
        self.feature_dimensions = self.extractor.feature_dims()

    def feature_dims(self) -> Dict[str, int]:
        return self.feature_dimensions

    def forward(self, item_ids: Union[List[int], torch.Tensor], device=None) -> Dict[str, torch.Tensor]:
        if isinstance(item_ids, list):
            original_shape = (len(item_ids),)
        else:
            original_shape = item_ids.shape
            item_ids = item_ids.view(-1).tolist()
            
        items = [self.item_dataset.get_item(item_id) for item_id in item_ids]
        features = self.extractor(items)
        
        return {k: v.view(*original_shape, -1) for k, v in features.items()}

# ==========================================
# 5. Initialization Helpers
# ==========================================

def initialize_feature_extractor(
    text_model_name: Optional[str] = None, 
    image_model_name: Optional[str] = None
) -> AbstractItemFeatureExtractor:
    """
    指定されたモデル名に基づいてエンコーダと抽出器を動的に構築するヘルパー。
    """
    image_model_name_to_extractor = {
        'google/siglip-base-patch16-224': SiglipImageEncoder,
        'google/vit-base-patch16-224-in21k': ViTImageEncoder
    }
    text_model_name_to_extractor = {
        'BAAI/bge-base-en-v1.5': SentenceTransformerEncoder,
        'all-mpnet-base-v2': SentenceTransformerEncoder,
        'all-MiniLM-L6-v2': SentenceTransformerEncoder,
        'all-MiniLM-L12-v2': SentenceTransformerEncoder,
    }

    extractors = {}
    
    if text_model_name:
        if text_model_name not in text_model_name_to_extractor:
            raise ValueError(f"Unsupported text model: {text_model_name}")
        text_encoder = text_model_name_to_extractor[text_model_name](model_name=text_model_name)
        extractors["text"] = TextFeatureExtractor(text_encoder, feature_key="text_features")
        
    if image_model_name:
        if image_model_name not in image_model_name_to_extractor:
            raise ValueError(f"Unsupported image model: {image_model_name}")
        image_encoder = image_model_name_to_extractor[image_model_name](model_name=image_model_name)
        extractors["image"] = ImageFeatureExtractor(image_encoder, feature_key="image_features")
        
    if not extractors:
        raise ValueError("At least one of text_model_name or image_model_name must be specified.")
        
    if len(extractors) == 1:
        return list(extractors.values())[0]
    else:
        return CombinedFeatureExtractor(extractors)