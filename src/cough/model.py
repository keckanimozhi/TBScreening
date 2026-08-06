"""Small CNN trained from scratch for TB vs non-TB cough classification
on log-mel spectrograms.

Spectrograms aren't natural images, so ImageNet transfer learning (used
for the X-ray model) doesn't transfer meaningfully here -- a small custom
CNN with dropout is the more appropriate choice for ~140 training clips.
"""

import torch.nn as nn


class CoughCNN(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model() -> nn.Module:
    return CoughCNN()
