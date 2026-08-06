"""ResNet18 transfer-learning model for TB vs Normal chest X-ray
classification. Backbone frozen except layer4; single-logit output for
binary classification (BCEWithLogitsLoss)."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def trainable_param_groups(model: nn.Module, layer4_lr: float, fc_lr: float):
    return [
        {"params": model.layer4.parameters(), "lr": layer4_lr},
        {"params": model.fc.parameters(), "lr": fc_lr},
    ]
