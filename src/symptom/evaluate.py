"""Evaluate the symptom classifier on the held-out test set.

Usage:
    .venv/Scripts/python.exe -m src.symptom.evaluate
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.symptom.labeling import SYMPTOM_COLUMNS


def main():
    test = pd.read_csv(PROCESSED_DATA_DIR / "symptom_test.csv")
    model = joblib.load(MODELS_DIR / "symptom" / "logreg.pkl")

    probs = model.predict_proba(test[SYMPTOM_COLUMNS])[:, 1]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    labels = test["label"]

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "n_test": len(labels),
        "note": (
            "Label is a domain-knowledge (NTEP/WHO presumptive-TB) rule "
            "applied to what looks like synthetic/randomly generated symptom "
            "data (see docs/PROJECT_CHECKLIST.md), not a microbiologically "
            "confirmed diagnosis. High accuracy here means the model "
            "recovered the screening rule from the symptom pattern, not that "
            "it predicts real TB status."
        ),
    }

    print(json.dumps(metrics, indent=2))

    out_path = REPORTS_DIR / "symptom" / "test_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
