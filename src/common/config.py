"""Central paths and constants shared across the AI4TB-Mobile ML pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

XRAY_DATA_DIR = PROJECT_ROOT / "100_TBDataset"
XRAY_NORMAL_DIR = XRAY_DATA_DIR / "Normal"
XRAY_TB_DIR = XRAY_DATA_DIR / "Tuberculosis"
XRAY_METADATA_XLSX = XRAY_DATA_DIR / "metadata.xlsx"

COUGH_DATA_DIR = PROJECT_ROOT / "dataset_200" / "cough"
COUGH_TB_DIR = COUGH_DATA_DIR / "tb"
COUGH_NOTB_DIR = COUGH_DATA_DIR / "notb"

SYMPTOM_CSV = PROJECT_ROOT / "Tb disease symptoms.csv"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42
IMAGE_SIZE = 224
