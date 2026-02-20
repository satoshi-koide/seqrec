import ast
import numpy as np
import torch
import gzip
from typing import Dict

from seqrec.dataset import Item, ItemDataset
from seqrec.module.feature_extractor import initialize_item_feature_store

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
            image_path=item_info.get('image_path', None)
        )

def extract(dataset_path: str, mode: str, model_name: str):
    device = 'cuda'

    print(f"Mode: {mode}, Model: {model_name}")

    if mode == 'image':
        item_dataset = ItemDataset({item.item_id: item for item in load_data(dataset_path)})
        print(f"Loaded {len(item_dataset)} items.")
        feature_extractor = initialize_item_feature_store(item_dataset, image_model_name=model_name, device=device)
    elif mode == 'text':
        item_dataset = ItemDataset({item.item_id: item for item in load_data(dataset_path)})
        print(f"Loaded {len(item_dataset)} items.")
        feature_extractor = initialize_item_feature_store(item_dataset, text_model_name=model_name, device=device) 
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    feature_extractor.build_cache(batch_size=512, verbose=True)
    feature_extractor.save_cache(f'{dataset_path}/feature_cache_{mode}_{model_name.replace("/", "_")}', key=f'{mode}_features')

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract features for items using specified models.")
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory containing meta.json.gz')
    parser.add_argument('--mode', type=str, choices=['image', 'text'], required=True, help='Type of features to extract: "image" or "text"')
    parser.add_argument('--model-name', type=str, required=True, help='Model name for feature extraction (e.g., "google/vit-base-patch16-224-in21k" for images, "all-mpnet-base-v2" for text)')
    args = parser.parse_args()

    extract(args.dataset_path, args.mode, args.model_name)

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