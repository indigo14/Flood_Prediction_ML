from pathlib import Path
from typing import Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.load_data import load_split_csv_data

class CSVDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if len(features) != len(labels):
            raise ValueError(
                "features and labels must be the same length. "
                f"{len(features)} != {len(labels)}"
            )

        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx].astype(np.float32)
        y = self.labels[idx]
        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.float32 if self.labels.dtype == np.float64 else torch.long)
        return x, y

def create_dataloader(
    batch_size: int, features_path: str, labels_path: str, shuffle: bool = True
) -> DataLoader[Any]:
    features, labels = load_split_csv_data(features_path, labels_path)
    dataset = CSVDataset(features, labels)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

# Example usage:
# train_dataloader = create_dataloader(batch_size=32, features_path='../data/processed/preprocessed_X_train.csv', labels_path='../data/processed/preprocessed_y_train.csv', shuffle=True)
# val_dataloader = create_dataloader(batch_size=32, features_path='../data/processed/preprocessed_X_val.csv', labels_path='../data/processed/preprocessed_y_val.csv', shuffle=False)
# test_dataloader = create_dataloader(batch_size=32, features_path='../data/processed/preprocessed_X_test.csv', labels_path='../data/processed/preprocessed_y_test.csv', shuffle=False)
