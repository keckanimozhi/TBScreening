"""Domain-knowledge labeling for Tb disease symptoms.csv.

There is no ground-truth TB/non-TB column in the source CSV, so a label
is derived here from the standard presumptive pulmonary TB screening
definition used by India's National TB Elimination Programme (NTEP) and
WHO: a person is "presumptive TB" if they have cough of >=2 weeks
duration, OR fever of >=2 weeks together with at least one of weight
loss, night sweats, or blood in cough/sputum.

This is a clinical screening rule, not a diagnosis -- it is exactly what
frontline health workers use to decide who to refer for confirmatory
testing (GeneXpert/TrueNat/sputum microscopy), which is precisely the
role the proposal assigns to the symptom module ("risk-factor scoring
module" / "referral recommendation"), not a stand-alone diagnostic
classifier.

IMPORTANT CAVEAT: inspection of Tb disease symptoms.csv shows every
symptom column present at ~50% prevalence independently, no correlation
structure between symptoms, and placeholder names (e.g. "Noe", "Genna")
-- this looks like randomly generated / synthetic data, not real patient
records. Labeling and modeling proceeds as directed, but any resulting
metrics demonstrate the labeling + modeling *methodology* working
correctly, not real predictive validity. See docs/PROJECT_CHECKLIST.md.
"""

import pandas as pd

from src.common.config import SYMPTOM_CSV

# Column names after stripping whitespace (source CSV has trailing spaces
# on some headers, e.g. "night sweats ").
COL_FEVER_2W = "fever for two weeks"
COL_HEMOPTYSIS = "coughing blood"
COL_BLOOD_SPUTUM = "sputum mixed with blood"
COL_NIGHT_SWEATS = "night sweats"
COL_CHEST_PAIN = "chest pain"
COL_BACK_PAIN = "back pain in certain parts"
COL_SOB = "shortness of breath"
COL_WEIGHT_LOSS = "weight loss"
COL_FATIGUE = "body feels tired"
COL_ARMPIT_NECK_LUMPS = "lumps that appear around the armpits and neck"
COL_COUGH_2_4W = "cough and phlegm continuously for two weeks to four weeks"
COL_SWOLLEN_NODES = "swollen lymph nodes"
COL_APPETITE_LOSS = "loss of appetite"

SYMPTOM_COLUMNS = [
    COL_FEVER_2W,
    COL_HEMOPTYSIS,
    COL_BLOOD_SPUTUM,
    COL_NIGHT_SWEATS,
    COL_CHEST_PAIN,
    COL_BACK_PAIN,
    COL_SOB,
    COL_WEIGHT_LOSS,
    COL_FATIGUE,
    COL_ARMPIT_NECK_LUMPS,
    COL_COUGH_2_4W,
    COL_SWOLLEN_NODES,
    COL_APPETITE_LOSS,
]

# Weighted clinical risk score, for explainability / stratification
# alongside the binary label. Weights reflect how specific each symptom
# is to TB in standard clinical teaching, not a fitted/learned value:
#   3 = cardinal pulmonary TB symptom (cough >=2wk, hemoptysis)
#   2 = classic supportive symptom (fever/night sweats/weight loss/blood
#       sputum) or a sign suggestive of extrapulmonary (lymphatic) TB
#   1 = nonspecific systemic symptom, common to many illnesses
SYMPTOM_WEIGHTS = {
    COL_COUGH_2_4W: 3,
    COL_HEMOPTYSIS: 3,
    COL_FEVER_2W: 2,
    COL_NIGHT_SWEATS: 2,
    COL_WEIGHT_LOSS: 2,
    COL_BLOOD_SPUTUM: 2,
    COL_SWOLLEN_NODES: 2,
    COL_ARMPIT_NECK_LUMPS: 2,
    COL_CHEST_PAIN: 1,
    COL_BACK_PAIN: 1,
    COL_SOB: 1,
    COL_FATIGUE: 1,
    COL_APPETITE_LOSS: 1,
}
MAX_WEIGHTED_SCORE = sum(SYMPTOM_WEIGHTS.values())  # 23


def load_symptoms() -> pd.DataFrame:
    df = pd.read_csv(SYMPTOM_CSV)
    df.columns = [c.strip() for c in df.columns]
    return df


def assign_presumptive_tb_label(df: pd.DataFrame) -> pd.Series:
    """NTEP/WHO-style presumptive pulmonary TB screening rule."""
    cough_2w = df[COL_COUGH_2_4W] == 1
    fever_2w = df[COL_FEVER_2W] == 1
    supportive = (
        (df[COL_WEIGHT_LOSS] == 1)
        | (df[COL_NIGHT_SWEATS] == 1)
        | (df[COL_HEMOPTYSIS] == 1)
        | (df[COL_BLOOD_SPUTUM] == 1)
    )
    presumptive = cough_2w | (fever_2w & supportive)
    return presumptive.astype(int)


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """Continuous 0-1 weighted clinical risk score, for stratification /
    explainability -- separate from the binary label used for
    classifier training."""
    weighted = sum(df[col] * weight for col, weight in SYMPTOM_WEIGHTS.items())
    return weighted / MAX_WEIGHTED_SCORE
