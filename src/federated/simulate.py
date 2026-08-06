"""Federated Averaging (FedAvg) simulation for the X-ray TB classifier,
demonstrating the proposal's "privacy-preserving Federated Learning
framework" architecture: each simulated institution (client) trains
locally on its own private data partition; only model weight updates are
exchanged with the (simulated) aggregation server, never raw images.

Only `layer4` + `fc` are federated (aggregated across clients each
round) -- the rest of the ResNet18 backbone is the same frozen,
publicly-available ImageNet-pretrained weights on every client (nothing
learned from private data lives there), matching how the centralized
model in src/xray/train.py was already trained (see src/xray/model.py:
only layer4+fc have requires_grad=True). This also mirrors the real
communication-efficiency motivation for federated learning: send only
what was actually trained, not the whole network.

Usage:
    .venv/Scripts/python.exe -m src.federated.simulate
"""

import copy
import json

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.common.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, REPORTS_DIR
from src.federated.partition import N_CLIENTS, partition_clients
from src.xray.dataset import XrayDataset
from src.xray.model import build_model

ROUNDS = 15
LOCAL_EPOCHS = 2
BATCH_SIZE = 16
LR = 1e-4


def get_trainable_state(model: nn.Module) -> dict:
    return {
        "layer4": copy.deepcopy(model.layer4.state_dict()),
        "fc": copy.deepcopy(model.fc.state_dict()),
    }


def set_trainable_state(model: nn.Module, state: dict):
    model.layer4.load_state_dict(state["layer4"])
    model.fc.load_state_dict(state["fc"])


def fedavg(states: list, weights: list) -> dict:
    total = sum(weights)
    avg = {"layer4": {}, "fc": {}}
    for part in ("layer4", "fc"):
        for key in states[0][part]:
            stacked = sum(s[part][key].float() * w for s, w in zip(states, weights))
            avg[part][key] = (stacked / total).to(states[0][part][key].dtype)
    return avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.numpy())
    return all_labels, all_probs


def train_local(model, loader, device, epochs, lr):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(model.layer4.parameters()) + list(model.fc.parameters()), lr=lr
    )
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


def main():
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")

    client_frames = partition_clients(N_CLIENTS)
    client_loaders = []
    for i, df in enumerate(client_frames):
        csv_path = PROCESSED_DATA_DIR / f"xray_client_{i}.csv"
        df.to_csv(csv_path, index=False)
        ds = XrayDataset(csv_path, train=True)
        client_loaders.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True))

    val_loader = DataLoader(XrayDataset(PROCESSED_DATA_DIR / "xray_val.csv", train=False), batch_size=16)
    test_loader = DataLoader(XrayDataset(PROCESSED_DATA_DIR / "xray_test.csv", train=False), batch_size=16)

    global_model = build_model().to(device)
    global_state = get_trainable_state(global_model)

    history = []
    for round_num in range(1, ROUNDS + 1):
        client_states, client_sizes = [], []
        for loader, df in zip(client_loaders, client_frames):
            local_model = build_model().to(device)
            set_trainable_state(local_model, global_state)
            train_local(local_model, loader, device, LOCAL_EPOCHS, LR)
            client_states.append(get_trainable_state(local_model))
            client_sizes.append(len(df))

        global_state = fedavg(client_states, client_sizes)
        set_trainable_state(global_model, global_state)

        labels, probs = evaluate(global_model, val_loader, device)
        val_auc = roc_auc_score(labels, probs)
        history.append({"round": round_num, "val_auc": val_auc})
        print(f"Round {round_num:02d}/{ROUNDS} | val_auc={val_auc:.4f}")

    ckpt_dir = MODELS_DIR / "federated"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(global_state, ckpt_dir / "xray_fedavg_trainable.pt")

    labels, probs = evaluate(global_model, test_loader, device)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    fed_metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "n_test": len(labels),
        "n_clients": N_CLIENTS,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
    }

    central_metrics_path = REPORTS_DIR / "xray" / "test_metrics.json"
    central_metrics = None
    if central_metrics_path.exists():
        with open(central_metrics_path) as f:
            central_metrics = json.load(f)

    result = {
        "federated": fed_metrics,
        "centralized_baseline": central_metrics,
        "history": history,
        "note": (
            "Same test set as the centralized X-ray model (Task 3). Federated "
            "vs centralized gap here reflects the very small per-client data "
            "(~46-47 images split 3 ways) more than any inherent FedAvg "
            "limitation -- with the proposal's real 800-participant multicentric "
            "data this gap would be expected to narrow substantially."
        ),
    }

    print(json.dumps(result, indent=2))

    out_path = REPORTS_DIR / "federated" / "test_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
