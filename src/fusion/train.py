"""Train the late-fusion model: logistic regression on
[p_xray, p_cough, p_symptom] -> TB label. A simple linear meta-learner is
enough here since there are only 3 input features and ~115 training
pairs -- anything heavier would overfit three numbers.

Usage:
    .venv/Scripts/python.exe -m src.fusion.train
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.common.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED

FEATURES = ["p_xray", "p_cough", "p_symptom"]


def main():
    train = pd.read_csv(PROCESSED_DATA_DIR / "fusion_train.csv")
    val = pd.read_csv(PROCESSED_DATA_DIR / "fusion_val.csv")

    X_train, y_train = train[FEATURES], train["label"]
    X_val, y_val = val[FEATURES], val["label"]

    model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Val AUC (fused): {val_auc:.4f}")
    weights = ", ".join(f"{name}={coef:.3f}" for name, coef in zip(FEATURES, model.coef_[0]))
    print(f"Learned weights: {weights}, intercept={model.intercept_[0]:.3f}")

    out_dir = MODELS_DIR / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "logreg.pkl")
    print(f"Saved fusion model to {out_dir / 'logreg.pkl'}")


if __name__ == "__main__":
    main()
