"""Evaluate the fused model against each standalone unimodal model on the
same held-out test pairs, so any lift from fusion is a fair comparison
(same rows for all three).

Usage:
    .venv/Scripts/python.exe -m src.fusion.evaluate
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR


def metrics_from_probs(labels, probs):
    preds = [1 if p >= 0.5 else 0 for p in probs]
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
    }


def main():
    test = pd.read_csv(PROCESSED_DATA_DIR / "fusion_test.csv")
    fusion_model = joblib.load(MODELS_DIR / "fusion" / "logreg.pkl")

    fused_probs = fusion_model.predict_proba(test[["p_xray", "p_cough", "p_symptom"]])[:, 1]

    results = {
        "n_test": len(test),
        "xray_only": metrics_from_probs(test["label"], test["p_xray"]),
        "cough_only": metrics_from_probs(test["label"], test["p_cough"]),
        "symptom_only": metrics_from_probs(test["label"], test["p_symptom"]),
        "fused": metrics_from_probs(test["label"], fused_probs),
        "note": (
            "X-ray<->cough<->symptom pairing is synthetic (matched by label "
            "only, not a real per-patient correspondence); cough labels come "
            "from only 7 real subjects (2 in this test split); symptom labels "
            "are a domain-knowledge rule applied to what looks like synthetic "
            "symptom data -- see docs/PROJECT_CHECKLIST.md. These numbers "
            "demonstrate the fusion architecture works end-to-end, not a "
            "validated multimodal signal."
        ),
    }

    print(json.dumps(results, indent=2))

    out_path = REPORTS_DIR / "fusion" / "test_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
