import ast
import numpy as np
import torch
import gzip
from typing import Dict

from seqrec.dataset import Item, ItemDataset
from seqrec.module.feature_extractor import initialize_feature_extractor, CachedItemFeatureStore

def load_data(dataset_path: str, data_size=None):
    for item_id, line in enumerate(gzip.open(dataset_path + '/meta.json.gz', 'rt')):
        if data_size and item_id >= data_size:
            break
        item_info = ast.literal_eval(line)
        categories = item_info.get('categories', None)
        categories = categories[0] if categories else None
        yield Item(
            item_id=item_id,
            asin=item_info['asin'],
            title=item_info.get('title', None),
            description=item_info.get('description', None),
            price=item_info.get('price', None),
            brand=item_info.get('brand', None),
            categories=categories,
            image_path=f'{dataset_path}/images/{item_info["asin"]}.jpg'
        )

def extract(dataset_path: str, mode: str, model_name: str):
    device = 'cuda'

    print(f"Mode: {mode}, Model: {model_name}")

    item_dataset = list(load_data(dataset_path))
    output_path = {}
    if mode == 'image':
        print(f"Loaded {len(item_dataset)} items.")
        feature_extractor = initialize_feature_extractor(image_model_name=model_name, device=device, verbose=True)
        output_path['image_features'] = f'{dataset_path}/feature_cache_image_{model_name.replace("/", "_")}'
    elif mode == 'text':
        print(f"Loaded {len(item_dataset)} items.")
        feature_extractor = initialize_feature_extractor(text_model_name=model_name, device=device, verbose=True) 
        output_path['text_features'] = f'{dataset_path}/feature_cache_text_{model_name.replace("/", "_")}'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    feature_extractor.build_cache(item_dataset, batch_size=512, verbose=True, save_to=output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract features for items using specified models.")
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory containing meta.json.gz')
    parser.add_argument('--mode', type=str, choices=['image', 'text'], required=True, help='Type of features to extract: "image" or "text"')
    parser.add_argument('--model-name', type=str, required=True, help='Model name for feature extraction (e.g., "google/vit-base-patch16-224-in21k" for images, "all-mpnet-base-v2" for text)')
    args = parser.parse_args()

    extract(args.dataset_path, args.mode, args.model_name)

    # test loading the cache
    npz_path = {
        f'{args.mode}_features': f'{args.dataset_path}/feature_cache_{args.mode}_{args.model_name.replace("/", "_")}.npz'
    }
    feature_store = CachedItemFeatureStore(npz_path)
    print(f"Loaded features for {len(feature_store)} items.")
    # # First 3 examples
    # for i, (item_id, features) in enumerate(feature_store.cache.items()):
    #     if i >= 3:
    #         break
    #     print(f"Item ID: {item_id}, Features: {features[i]}")

if __name__ == "__main__":
    main()

# Usage examples:
# image_model=google/siglip-base-patch16-224
# text_model=BAAI/bge-base-en-v1.5

# category=toys
# uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode image --model-name $image_model
# uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode text --model-name $text_model

# category=sports
# uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode image --model-name $image_model
# uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode text --model-name $text_model

# category=beauty
# uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode image --model-name $image_model
# uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode text --model-name $text_model