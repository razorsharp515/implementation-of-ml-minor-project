from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_DATASET = RAW_DATA_DIR / "online_payments.csv"
DEMO_DATASET = SAMPLE_DATA_DIR / "demo_transactions.csv"
MODEL_PATH = MODELS_DIR / "best_fraud_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.csv"
ROC_PATH = REPORTS_DIR / "roc_curve.png"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
SHAP_PATH = REPORTS_DIR / "shap_summary.png"

RANDOM_STATE = 42
CV_FOLDS = (5, 10)

TARGET_COLUMN = "isFraud"
DROP_COLUMNS = [
    "nameOrig",
    "nameDest",
    "isFlaggedFraud",
    "newbalanceDest",
    "newbalanceOrig",
]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
