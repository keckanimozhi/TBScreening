"""Build a manifest of Tb disease symptoms.csv with the domain-knowledge
label from src/symptom/labeling.py, and a stratified train/val/test
split.

Unlike the cough dataset, each row here has a unique `id` with no
duplicates (verified during Task 9 data audit) -- a plain stratified
random split at the row level is valid, no subject-grouping needed.

Usage:
    .venv/Scripts/python.exe -m src.symptom.manifest
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import PROCESSED_DATA_DIR, RANDOM_SEED
from src.symptom.labeling import (
    SYMPTOM_COLUMNS,
    assign_presumptive_tb_label,
    compute_risk_score,
    load_symptoms,
)


def build_manifest() -> pd.DataFrame:
    df = load_symptoms()
    manifest = df[["id"] + SYMPTOM_COLUMNS].copy()
    manifest["label"] = assign_presumptive_tb_label(df)
    manifest["risk_score"] = compute_risk_score(df)
    return manifest


def split_manifest(manifest: pd.DataFrame):
    train, temp = train_test_split(
        manifest, test_size=0.30, stratify=manifest["label"], random_state=RANDOM_SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=RANDOM_SEED
    )
    return train, val, test


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    manifest.to_csv(PROCESSED_DATA_DIR / "symptom_manifest.csv", index=False)

    train, val, test = split_manifest(manifest)
    train.to_csv(PROCESSED_DATA_DIR / "symptom_train.csv", index=False)
    val.to_csv(PROCESSED_DATA_DIR / "symptom_val.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "symptom_test.csv", index=False)

    print(f"Manifest: {len(manifest)} patients ({manifest['label'].sum()} presumptive-TB, {(manifest['label']==0).sum()} not)")
    print(f"Train: {len(train)} ({train['label'].sum()} pos) | Val: {len(val)} ({val['label'].sum()} pos) | Test: {len(test)} ({test['label'].sum()} pos)")


if __name__ == "__main__":
    main()
