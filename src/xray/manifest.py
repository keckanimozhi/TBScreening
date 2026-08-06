"""Build a manifest of the 100_TBDataset chest X-rays and a stratified
train/val/test split.

Label source of truth: folder placement (Normal/ vs Tuberculosis/), cross
-checked against metadata.xlsx's `findings` column and the CHNCXR_*_{0,1}
filename suffix convention -- all three agree for all 200 images.

Usage:
    .venv/Scripts/python.exe -m src.xray.manifest
"""

import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import (
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    XRAY_METADATA_XLSX,
    XRAY_NORMAL_DIR,
    XRAY_TB_DIR,
)


def load_metadata() -> pd.DataFrame:
    wb = openpyxl.load_workbook(XRAY_METADATA_XLSX)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    return pd.DataFrame(rows, columns=["study_id", "gender", "age", "findings"])


def build_manifest() -> pd.DataFrame:
    meta = load_metadata()

    records = []
    for path in sorted(XRAY_NORMAL_DIR.glob("*.png")):
        records.append({"filename": path.name, "filepath": str(path), "label": 0})
    for path in sorted(XRAY_TB_DIR.glob("*.png")):
        records.append({"filename": path.name, "filepath": str(path), "label": 1})

    manifest = pd.DataFrame(records)
    manifest = manifest.merge(
        meta, left_on="filename", right_on="study_id", how="left"
    ).drop(columns=["study_id"])

    # Sanity check: folder-derived label must agree with metadata findings.
    mismatch = manifest[
        (manifest["label"] == 0) & (manifest["findings"] != "normal")
        | (manifest["label"] == 1) & (manifest["findings"] == "normal")
    ]
    if not mismatch.empty:
        raise ValueError(f"Label mismatch between folder and metadata for: {mismatch['filename'].tolist()}")

    return manifest


def split_manifest(manifest: pd.DataFrame):
    train, temp = train_test_split(
        manifest,
        test_size=0.30,
        stratify=manifest["label"],
        random_state=RANDOM_SEED,
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["label"],
        random_state=RANDOM_SEED,
    )
    return train, val, test


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    manifest.to_csv(PROCESSED_DATA_DIR / "xray_manifest.csv", index=False)

    train, val, test = split_manifest(manifest)
    train.to_csv(PROCESSED_DATA_DIR / "xray_train.csv", index=False)
    val.to_csv(PROCESSED_DATA_DIR / "xray_val.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "xray_test.csv", index=False)

    print(f"Manifest: {len(manifest)} images ({manifest['label'].sum()} TB, {(manifest['label']==0).sum()} Normal)")
    print(f"Train: {len(train)} ({train['label'].sum()} TB) | Val: {len(val)} ({val['label'].sum()} TB) | Test: {len(test)} ({test['label'].sum()} TB)")


if __name__ == "__main__":
    main()
