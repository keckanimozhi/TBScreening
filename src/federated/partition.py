"""Partition the X-ray training set into N simulated institutions
("clients") for the federated learning demo (Task 13) -- matching the
proposal's "each center trains on its own X-ray dataset" federated
learning use case.

Chose the X-ray model for this demo (over cough/symptom) because it's
the only track with a real, sizeable image dataset from independent
real patients (200 distinct people) -- the cough track only has 7 real
subjects, too few to meaningfully simulate multiple institutions each
holding a private slice of data.

Stratified so each simulated client has a similar TB/Normal ratio,
matching how the proposal describes real participating hospitals each
seeing a mix of TB and non-TB patients, not one hospital seeing only
TB cases.

Usage:
    .venv/Scripts/python.exe -m src.federated.partition
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.common.config import PROCESSED_DATA_DIR, RANDOM_SEED

N_CLIENTS = 3


def partition_clients(n_clients: int = N_CLIENTS) -> list[pd.DataFrame]:
    train = pd.read_csv(PROCESSED_DATA_DIR / "xray_train.csv")
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=RANDOM_SEED)

    clients = []
    for _, client_idx in skf.split(train, train["label"]):
        clients.append(train.iloc[client_idx].reset_index(drop=True))
    return clients


def main():
    clients = partition_clients()
    for i, client_df in enumerate(clients):
        path = PROCESSED_DATA_DIR / f"xray_client_{i}.csv"
        client_df.to_csv(path, index=False)
        print(f"Client {i}: {len(client_df)} images ({client_df['label'].sum()} TB) -> {path}")


if __name__ == "__main__":
    main()
