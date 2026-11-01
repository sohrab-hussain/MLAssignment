"""Shared settings: file paths, column groups and the random seed.

Every script imports from here so that all five models are trained and
evaluated on exactly the same data split and the same feature set.
"""
from pathlib import Path

# this file is src/common/config.py -> project root is three levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "train_data.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"   # required at repo root by the brief
MODEL_DIR = PROJECT_ROOT / "model"   # one sub-folder per model, see model_dir()

# file name used for each model's folder under model/ and src/models/
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}


def model_dir(model_key: str) -> Path:
    """model/<model_key>/ holds model.joblib and metrics.json for that model."""
    return MODEL_DIR / model_key

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "y"
POSITIVE_LABEL = "yes"   # the client subscribed to a term deposit

# 'duration' is the length of the sales call. It is only known after the call
# ends, so using it to predict the call's outcome would be target leakage.
LEAKY_COLUMNS = ["duration"]

NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
