import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SequentialRecDataset(Dataset):
    def __init__(self, 
                 user_seqs, 
                 targets, 
                 max_len=50, 
                 image_features=None, 
                 text_features=None, 
                 pad_token=0):
        """
        Args:
            user_seqs (list): List of item ID histories per user [[1, 2], [3, 4, 5], ...]
            targets (list): Next item ID (ground truth label) for each history [3, 6, ...]
            max_len (int): Maximum sequence length
            image_features (torch.Tensor, optional): Image features corresponding to item IDs (num_items + 1, dim)
            text_features (torch.Tensor, optional): Text features corresponding to item IDs (num_items + 1, dim)
            pad_token (int): ID used for padding (typically 0)
        """
        self.user_seqs = user_seqs
        self.targets = targets
        self.max_len = max_len
        self.pad_token = pad_token
        
        # Hold features as references for memory efficiency (or None)
        self.image_features = image_features
        self.text_features = text_features

    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, index):
        # 1. Get and truncate sequence data
        seq = self.user_seqs[index]
        target = self.targets[index]
        
        # Take from the end to match max_len (recent history is important)
        seq = seq[-self.max_len:]
        seq_len = len(seq)
        
        # 2. Padding
        # Prepare zero-padded array
        input_ids = torch.full((self.max_len,), self.pad_token, dtype=torch.long)
        # Fill data (right-aligned padding is common: [0, 0, 1, 2])
        # Note: Some SASRec implementations use left-aligned, but right-aligned makes Attention Mask easier
        input_ids[self.max_len - seq_len:] = torch.tensor(seq, dtype=torch.long)

        # Create dictionary of data to return
        item = {
            "input_ids": input_ids,      # History (Tensor)
            "target_id": torch.tensor(target, dtype=torch.long),  # Ground truth (Tensor)
        }

        # 3. Optional: Get image features
        if self.image_features is not None:
            # Image features of history items
            # (Assumes pad_token=0 positions have zero vectors, or masked by Embedding layer)
            seq_img = self.image_features[input_ids] 
            
            # Image feature of target item (used for Positive Sample during training, etc.)
            target_img = self.image_features[target]
            
            item["seq_img_feat"] = seq_img
            item["target_img_feat"] = target_img

        # 4. Optional: Get text features
        if self.text_features is not None:
            seq_text = self.text_features[input_ids]
            target_text = self.text_features[target]
            
            item["seq_text_feat"] = seq_text
            item["target_text_feat"] = target_text

        return item

def create_datasets(data_path, max_len=50, image_feat=None, text_feat=None):
    """
    Factory function to create Train/Eval/Test datasets from raw data
    For fairness, excludes cases where the target is in missing_image_ids
    """

    datamaps = json.load(open(f'{data_path}/datamaps.json', 'r'))

    print('Number of items', len(datamaps["item2id"]))

    missing_image_ids = set()  # Define missing image IDs here (e.g., {10, 20, 30})
    for asin in open(f'{data_path}/missing_asins.txt', 'r'):
        asin = asin.strip()
        if asin in datamaps["item2id"]:
            item_id = datamaps["item2id"][asin]
            missing_image_ids.add(int(item_id))

    print(f"Total missing image IDs: {len(missing_image_ids)}")
    
    # Load data (1 line: UserID Item1 Item2 ...)
    # Assumes ItemIDs start from 1. 0 is reserved for padding.
    user_ids = []
    sequences = []
    with open(f'{data_path}/sequential_data.txt', 'r') as f:
        for line in f:
            line_ids = list(map(int, line.strip().split()))
            # line_ids[0] is UserID, rest are Items
            # If dataset IDs already start from 0, need to add +1 to make them 1-based
            # Here we assume "data file is 1-based" and use as is
            sequences.append(line_ids[1:]) 
            user_ids.append(line_ids[0])

    # Lists to store data for each split
    data_splits = {
        "train": {"seqs": [], "targets": []},
        "eval":  {"seqs": [], "targets": []},
        "test":  {"seqs": [], "targets": []}
    }
    
    # Convert missing IDs to set (for O(1) lookup)
    missing_set = set(missing_image_ids) if missing_image_ids else set()

    for seq in sequences:
        # Skip if sequence is too short (e.g., len < 3)
        if len(seq) < 3: continue

        # --- Test Set Construction ---
        # Target: last item (seq[-1])
        # Input: everything before that (seq[:-1])
        test_target = seq[-1]
        test_input = seq[:-1]
        
        # [IMPORTANT] Exclude from test data if target has no image
        if test_target not in missing_set:
            data_splits["test"]["seqs"].append(test_input)
            data_splits["test"]["targets"].append(test_target)

        # --- Eval Set Construction ---
        # Target: second to last (seq[-2])
        # Input: everything before that (seq[:-2])
        eval_target = seq[-2]
        eval_input = seq[:-2]
        
        if eval_target not in missing_set:
            data_splits["eval"]["seqs"].append(eval_input)
            data_splits["eval"]["targets"].append(eval_target)

        # --- Train Set Construction ---
        # Target: same as Eval (seq[-2]) *For SASRec training, sliding window is applied further
        # Here we simply pass "all history up to Eval point"
        # During training, even without target image, typically fill with zero vector to preserve data
        # For strict stability of image-based model training, could exclude here too.
        # Since focus is on "common test data", we use maximum data for Train.
        train_target = seq[-2]
        train_input = seq[:-2]
        
        data_splits["train"]["seqs"].append(train_input)
        data_splits["train"]["targets"].append(train_target)

    # Create Dataset instances
    datasets = {}
    for split in ["train", "eval", "test"]:
        datasets[split] = SequentialRecDataset(
            user_seqs=data_splits[split]["seqs"],
            targets=data_splits[split]["targets"],
            max_len=max_len,
            image_features=image_feat, # Pass to all datasets (ignored if None)
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