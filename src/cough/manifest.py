"""Build a manifest of the dataset_200/cough recordings and a *subject
-grouped* train/val/test split.

IMPORTANT — data leakage fix: the 200 .wav files are NOT 200 independent
patients. Filename inspection shows they are repeated cough-event clips
from only 7 real subjects total:
    TB (label 1):     10A (53 clips), 113A (27 clips), 108A (20 clips)
    non-TB (label 0): 2C (51 clips), 6C (23 clips), 3C (15 clips), 4C (11 clips)
A naive random file-level split (the original version of this script)
put clips from the same subject into train AND val/test, letting a model
learn to recognize individuals/microphones instead of TB-relevant cough
acoustics -- this produced a spurious 1.0 val AUC in the first Task 6 run.
This version splits whole subjects into whole splits so no subject's
clips appear in more than one split.

Given only 3-4 subjects per class, the resulting split proportions are
necessarily coarse (not an exact 70/15/15) -- see group_split_by_subject
for the assignment method.

Usage:
    .venv/Scripts/python.exe -m src.cough.manifest
"""

import re

import pandas as pd

from src.common.config import COUGH_NOTB_DIR, COUGH_TB_DIR, PROCESSED_DATA_DIR

SUBJECT_PATTERN = re.compile(r"^PID_(\d*[A-Za-z]+)")


def extract_subject(filename: str) -> str:
    """Real subject code, e.g. 'PID_2C101_yeti_0.wav' -> '2C',
    'PID_108A_25_yeti_0.wav' -> '108A'."""
    m = SUBJECT_PATTERN.match(filename)
    if not m:
        raise ValueError(f"Could not parse subject id from filename: {filename}")
    return m.group(1)


def build_manifest() -> pd.DataFrame:
    records = []
    for path in sorted(COUGH_NOTB_DIR.glob("*.wav")):
        records.append(
            {
                "filename": path.name,
                "filepath": str(path),
                "label": 0,
                "subject_id": extract_subject(path.name),
            }
        )
    for path in sorted(COUGH_TB_DIR.glob("*.wav")):
        records.append(
            {
                "filename": path.name,
                "filepath": str(path),
                "label": 1,
                "subject_id": extract_subject(path.name),
            }
        )
    return pd.DataFrame(records)


def group_split_by_subject(subject_counts: dict, ratios=(0.70, 0.15, 0.15)) -> dict:
    """Assign whole subjects to train/val/test targeting `ratios`, without
    ever splitting one subject's clips across sets.

    Method: seed the split by giving the `n_splits` largest subjects one
    each to train/val/test (largest -> train, guaranteeing every split is
    non-empty even with as few as 3 subjects). Any remaining subjects (only
    possible when a class has more subjects than splits) are then assigned
    greedily to whichever split is furthest below its target count.
    """
    splits = ["train", "val", "test"]
    targets = {s: r * sum(subject_counts.values()) for s, r in zip(splits, ratios)}
    current = {s: 0 for s in splits}
    assignment = {}

    subjects_desc = sorted(subject_counts.items(), key=lambda kv: -kv[1])

    for split, (subject, count) in zip(splits, subjects_desc[: len(splits)]):
        assignment[subject] = split
        current[split] += count

    for subject, count in subjects_desc[len(splits):]:
        deficits = {s: targets[s] - current[s] for s in splits}
        best_split = max(deficits, key=deficits.get)
        assignment[subject] = best_split
        current[best_split] += count

    return assignment


def assign_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `manifest` with a `split` column (train/val/test),
    assigned per-class via group_split_by_subject so no subject_id crosses
    split boundaries. Reused by src/fusion/manifest.py to keep the
    multimodal pairing's split membership consistent with the cough
    model's own split."""
    manifest = manifest.copy()
    manifest["split"] = None

    for label in manifest["label"].unique():
        subset = manifest[manifest["label"] == label]
        subject_counts = subset.groupby("subject_id").size().to_dict()
        assignment = group_split_by_subject(subject_counts)
        manifest.loc[subset.index, "split"] = subset["subject_id"].map(assignment)

    return manifest


def split_manifest(manifest: pd.DataFrame):
    manifest = assign_splits(manifest)
    train = manifest[manifest["split"] == "train"].drop(columns=["split"])
    val = manifest[manifest["split"] == "val"].drop(columns=["split"])
    test = manifest[manifest["split"] == "test"].drop(columns=["split"])
    return train, val, test


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    manifest.to_csv(PROCESSED_DATA_DIR / "cough_manifest.csv", index=False)

    train, val, test = split_manifest(manifest)
    train.to_csv(PROCESSED_DATA_DIR / "cough_train.csv", index=False)
    val.to_csv(PROCESSED_DATA_DIR / "cough_val.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "cough_test.csv", index=False)

    print(f"Manifest: {len(manifest)} recordings ({manifest['label'].sum()} TB, {(manifest['label']==0).sum()} non-TB)")
    print(f"Unique subjects: {manifest['subject_id'].nunique()} total -- {manifest[manifest.label==1]['subject_id'].nunique()} TB, {manifest[manifest.label==0]['subject_id'].nunique()} non-TB")
    print(f"Train: {len(train)} ({train['label'].sum()} TB) | Val: {len(val)} ({val['label'].sum()} TB) | Test: {len(test)} ({test['label'].sum()} TB)")

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        subjects = sorted(split_df["subject_id"].unique())
        print(f"  {name} subjects: {subjects}")


if __name__ == "__main__":
    main()
