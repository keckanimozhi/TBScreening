"""PyTorch Dataset for cough log-mel spectrograms, reading from the
manifest CSVs produced by src/cough/manifest.py.

Uses SpecAugment-style masking (not transfer-learned ImageNet
augmentation) for the train split, since spectrograms aren't natural
images -- random frequency/time masking is the standard regularizer for
small audio datasets.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.cough.features import extract_log_mel


def spec_augment(mel: np.ndarray, freq_mask_width=8, time_mask_width=20) -> np.ndarray:
    mel = mel.copy()
    n_mels, n_frames = mel.shape

    f0 = np.random.randint(0, max(1, n_mels - freq_mask_width))
    mel[f0 : f0 + freq_mask_width, :] = 0.0

    t0 = np.random.randint(0, max(1, n_frames - time_mask_width))
    mel[:, t0 : t0 + time_mask_width] = 0.0

    return mel


class CoughDataset(Dataset):
    def __init__(self, manifest_csv, train: bool):
        self.df = pd.read_csv(manifest_csv)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = extract_log_mel(row["filepath"])
        if self.train:
            mel = spec_augment(mel)
        tensor = torch.tensor(mel).unsqueeze(0)  # (1, n_mels, n_frames)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return tensor, label
