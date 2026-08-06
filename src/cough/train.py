"""Train the cough audio TB classifier.

Usage:
    .venv/Scripts/python.exe -m src.cough.train
"""

import json

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.common.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, REPORTS_DIR
from src.cough.dataset import CoughDataset
from src.cough.model import build_model

EPOCHS = 40
BATCH_SIZE = 16
PATIENCE = 10


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * feats.size(0)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


def main():
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CoughDataset(PROCESSED_DATA_DIR / "cough_train.csv", train=True)
    val_ds = CoughDataset(PROCESSED_DATA_DIR / "cough_val.csv", train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_auc = -1.0
    epochs_without_improvement = 0
    history = []

    ckpt_dir = MODELS_DIR / "cough"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cough").mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "cnn_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * feats.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_auc = evaluate(model, val_loader, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc}
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_auc={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val_auc={best_val_auc:.4f})")
                break

    with open(REPORTS_DIR / "cough" / "train_history.json", "w") as f:
        json.dump({"history": history, "best_val_auc": best_val_auc}, f, indent=2)

    print(f"Best val AUC: {best_val_auc:.4f}. Checkpoint saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()
