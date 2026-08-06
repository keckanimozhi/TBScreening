"""Run the frozen X-ray, cough, and symptom models over the multimodal
manifest to produce per-pair (p_xray, p_cough, p_symptom, label) rows for
the fusion model.

Usage:
    .venv/Scripts/python.exe -m src.fusion.build_features
"""

import joblib
import pandas as pd

from src.common.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.fusion.components import (
    load_cough_model,
    load_xray_model,
    predict_cough,
    predict_xray,
)
from src.symptom.labeling import SYMPTOM_COLUMNS, load_symptoms


def main():
    xray_model = load_xray_model()
    cough_model = load_cough_model()
    symptom_model = joblib.load(MODELS_DIR / "symptom" / "logreg.pkl")
    symptoms_by_id = load_symptoms().set_index("id")

    for split in ("train", "val", "test"):
        df = pd.read_csv(PROCESSED_DATA_DIR / f"multimodal_{split}.csv")
        df["p_xray"] = df["xray_filepath"].apply(lambda p: predict_xray(xray_model, p))
        df["p_cough"] = df["cough_filepath"].apply(lambda p: predict_cough(cough_model, p))

        symptom_rows = symptoms_by_id.loc[df["symptom_id"], SYMPTOM_COLUMNS]
        df["p_symptom"] = symptom_model.predict_proba(symptom_rows)[:, 1]

        out = df[["patient_id", "label", "p_xray", "p_cough", "p_symptom"]]
        out.to_csv(PROCESSED_DATA_DIR / f"fusion_{split}.csv", index=False)
        print(f"{split}: {len(out)} rows -> data/processed/fusion_{split}.csv")


if __name__ == "__main__":
    main()
