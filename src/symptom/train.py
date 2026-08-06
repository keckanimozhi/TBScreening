"""Train the symptom-based presumptive-TB classifier.

Logistic regression, not a deep model: 13 binary inputs, a rule-derived
label, and interpretability is the point here (coefficients double as
the "feature importance" explanation the proposal's XAI module calls
for) -- a heavier model would just obscure that for no accuracy benefit.

Usage:
    .venv/Scripts/python.exe -m src.symptom.train
"""

import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.common.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, REPORTS_DIR
from src.symptom.labeling import SYMPTOM_COLUMNS


def main():
    train = pd.read_csv(PROCESSED_DATA_DIR / "symptom_train.csv")
    val = pd.read_csv(PROCESSED_DATA_DIR / "symptom_val.csv")

    X_train, y_train = train[SYMPTOM_COLUMNS], train["label"]
    X_val, y_val = val[SYMPTOM_COLUMNS], val["label"]

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Val AUC: {val_auc:.4f}")

    coefs = dict(zip(SYMPTOM_COLUMNS, model.coef_[0].round(3).tolist()))
    print("Learned feature importance (coefficients):")
    for name, coef in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name}: {coef:+.3f}")

    out_dir = MODELS_DIR / "symptom"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "logreg.pkl")

    (REPORTS_DIR / "symptom").mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "symptom" / "feature_importance.json", "w") as f:
        json.dump(coefs, f, indent=2)

    print(f"Saved model to {out_dir / 'logreg.pkl'}")


if __name__ == "__main__":
    main()
