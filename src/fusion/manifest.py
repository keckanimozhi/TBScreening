"""Build a SYNTHETIC multimodal pairing between the chest X-ray, cough,
and symptom datasets, for fusion-model pipeline development ahead of the
proposal's real paired clinical data collection (Phase II, 800
participants with matched X-ray + cough + symptoms per patient).

Why synthetic pairing is needed: 100_TBDataset (chest X-rays),
dataset_200/cough (cough recordings), and Tb disease symptoms.csv are
three unrelated data sources -- there is no real per-patient
correspondence between them. Until real paired data exists, this script
pairs each X-ray image with a cough clip AND a symptom record all sharing
the *same label* (TB with TB, Normal/non-TB with non-TB) and assigns a
new shared synthetic patient_id. This is useful for building/testing the
fusion architecture, but any performance number from a model trained on
this pairing reflects each modality's own signal, not a validated
multimodal correlation -- clearly flag this wherever these numbers are
reported.

Split membership is driven entirely by the cough side, since cough is
the leakage-constrained modality (only 7 real subjects -- see
src/cough/manifest.py). X-ray (200 independent real patients) and symptom
(1000 independent real rows, but a rule-derived label -- see
src/symptom/labeling.py) records carry no leakage risk, so they're freely
allocated to match whatever per-split counts the cough grouping requires.

Usage:
    .venv/Scripts/python.exe -m src.fusion.manifest
"""

import pandas as pd

from src.common.config import PROCESSED_DATA_DIR, RANDOM_SEED
from src.cough.manifest import assign_splits
from src.cough.manifest import build_manifest as build_cough_manifest
from src.symptom.manifest import build_manifest as build_symptom_manifest
from src.xray.manifest import build_manifest as build_xray_manifest


def build_pairing() -> pd.DataFrame:
    xray = build_xray_manifest().rename(
        columns={"filepath": "xray_filepath", "filename": "xray_filename"}
    )
    cough = assign_splits(build_cough_manifest()).rename(
        columns={
            "filepath": "cough_filepath",
            "filename": "cough_filename",
            "subject_id": "cough_subject_id",
        }
    )
    symptom = build_symptom_manifest().rename(columns={"id": "symptom_id"})

    paired_rows = []
    for label in (0, 1):
        xray_pool = xray[xray["label"] == label].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        symptom_pool = symptom[symptom["label"] == label].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        cough_label = cough[cough["label"] == label]

        xray_cursor = 0
        symptom_cursor = 0
        for split in ("train", "val", "test"):
            cough_split = cough_label[cough_label["split"] == split].sample(
                frac=1, random_state=RANDOM_SEED
            ).reset_index(drop=True)
            n = len(cough_split)

            xray_chunk = xray_pool.iloc[xray_cursor : xray_cursor + n].reset_index(drop=True)
            xray_cursor += n
            symptom_chunk = symptom_pool.iloc[symptom_cursor : symptom_cursor + n].reset_index(drop=True)
            symptom_cursor += n

            if len(xray_chunk) != n or len(symptom_chunk) != n:
                raise ValueError(
                    f"label={label} split={split}: {n} cough clips but only "
                    f"{len(xray_chunk)} unused X-ray images / {len(symptom_chunk)} "
                    f"unused symptom rows left to pair with."
                )

            for i in range(n):
                paired_rows.append(
                    {
                        "label": label,
                        "split": split,
                        "xray_filename": xray_chunk.loc[i, "xray_filename"],
                        "xray_filepath": xray_chunk.loc[i, "xray_filepath"],
                        "cough_filename": cough_split.loc[i, "cough_filename"],
                        "cough_filepath": cough_split.loc[i, "cough_filepath"],
                        "cough_subject_id": cough_split.loc[i, "cough_subject_id"],
                        "symptom_id": symptom_chunk.loc[i, "symptom_id"],
                    }
                )

    manifest = pd.DataFrame(paired_rows)
    prefix = manifest["label"].map({1: "TB", 0: "NORM"})
    manifest["patient_id"] = [
        f"{p}-{i+1:04d}" for p, i in zip(prefix, manifest.groupby("label").cumcount())
    ]
    cols = [
        "patient_id", "label", "split",
        "xray_filename", "xray_filepath",
        "cough_filename", "cough_filepath", "cough_subject_id",
        "symptom_id",
    ]
    return manifest[cols]


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_pairing()
    manifest.to_csv(PROCESSED_DATA_DIR / "multimodal_manifest.csv", index=False)

    for split in ("train", "val", "test"):
        subset = manifest[manifest["split"] == split]
        subset.drop(columns=["split"]).to_csv(
            PROCESSED_DATA_DIR / f"multimodal_{split}.csv", index=False
        )
        print(f"{split}: {len(subset)} pairs ({subset['label'].sum()} TB), cough subjects: {sorted(subset['cough_subject_id'].unique())}")

    print(f"\nTotal: {len(manifest)} synthetic patient pairs -> data/processed/multimodal_manifest.csv")
    print("Reminder: X-ray<->cough<->symptom pairing is SYNTHETIC (matched by label only), not a real per-patient correspondence.")


if __name__ == "__main__":
    main()
