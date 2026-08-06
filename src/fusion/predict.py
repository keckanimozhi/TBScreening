"""Single-patient fusion inference for the Streamlit app (Task 11).

Handles the case where a modality wasn't provided (e.g. no X-ray
uploaded) by imputing that input with its training-set mean probability
-- a neutral "we have no evidence either way from this modality" value,
rather than an arbitrary 0.5. This is a simple, documented fallback for
a research prototype UI, not a missing-data strategy validated for
clinical use.
"""

import joblib
import pandas as pd

from src.common.config import MODELS_DIR, PROCESSED_DATA_DIR

_FEATURES = ["p_xray", "p_cough", "p_symptom"]


def _train_means() -> dict:
    train = pd.read_csv(PROCESSED_DATA_DIR / "fusion_train.csv")
    return train[_FEATURES].mean().to_dict()


def load_fusion_model():
    return joblib.load(MODELS_DIR / "fusion" / "logreg.pkl")


def predict_fused(fusion_model, p_xray=None, p_cough=None, p_symptom=None) -> dict:
    """Returns {"prob": float, "used": {modality: bool, ...}} -- `used`
    records which inputs were real vs. imputed, so the UI can be
    transparent about it."""
    means = _train_means()

    values = {"p_xray": p_xray, "p_cough": p_cough, "p_symptom": p_symptom}
    used = {k: v is not None for k, v in values.items()}
    filled = {k: (v if v is not None else means[k]) for k, v in values.items()}

    row = pd.DataFrame([filled])[_FEATURES]
    prob = fusion_model.predict_proba(row)[0, 1]
    return {"prob": float(prob), "used": used}


def risk_band(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.3:
        return "Medium"
    return "Low"
