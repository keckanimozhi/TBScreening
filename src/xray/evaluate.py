"""Evaluate the trained chest X-ray TB classifier on the held-out test set.

Usage:
    .venv/Scripts/python.exe -m src.xray.evaluate
"""

import json

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.common.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.xray.dataset import XrayDataset
from src.xray.model import build_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.numpy())
    return all_labels, all_probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = XrayDataset(PROCESSED_DATA_DIR / "xray_test.csv", train=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = build_model().to(device)
    ckpt_path = MODELS_DIR / "xray" / "resnet18_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    labels, probs = predict(model, test_loader, device)
    preds = [1 if p >= 0.5 else 0 for p in probs]

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "n_test": len(labels),
    }

    print(json.dumps(metrics, indent=2))

    out_path = REPORTS_DIR / "xray" / "test_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
