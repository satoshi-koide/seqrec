import numpy as np

class PopularityRec:
    def __init__(self, train_data):
        self.popularity = self._compute_popularity(train_data)
    
    def _compute_popularity(self, train_data):
        item_counts = {}
        for seq in train_data:
            for item in list(seq['input_ids'].numpy()):
                if int(item) != 0:  # パディングIDはカウントしない
                    item_counts[int(item)] = item_counts.get(int(item), 0) + 1
        return sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)
    
    def recommend(self, top_k=10):
        return self.popularity[:top_k]

if __name__ == "__main__":
    from dataset import create_datasets

    categories = ['toys', 'sports', 'beauty']

    for category in categories:
        dataset_path = f"dataset/{category}"
        datasets = create_datasets(dataset_path)

        model = PopularityRec(datasets["train"])
        recommendations = model.recommend(top_k=10)
        print(f"Top 10 Popular Items for {category}: {recommendations}")
        # Compute metrics (Recall@10, NDCG@10) for the popularity-based recommendations
        # This is a simplified evaluation since PopularityRec does not personalize recommendations
        test_labels = [seq["labels"][-1].item() for seq in datasets["test"]]

        #print(test_labels)

        hits = [1 if label in recommendations else 0 for label in test_labels]
        recall_at_10 = sum(hits) / len(test_labels)
        ndcg_at_10 = sum([1 / np.log2(recommendations.index(label) + 2) if label in recommendations else 0 for label in test_labels]) / len(test_labels)

        print(f"PopularityRec - Recall@10: {recall_at_10:.5f}, NDCG@10: {ndcg_at_10:.5f}")
    
