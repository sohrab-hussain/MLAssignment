"""Feature preprocessing shared by all models.

The preprocessor lives *inside* each saved model pipeline, so the Streamlit
app can pass the raw columns of test_data.csv straight to model.predict()
without repeating any encoding or scaling by hand.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, POSITIVE_LABEL, TARGET_COLUMN


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """One-hot encode categoricals; optionally standardise numeric columns.

    Scaling matters for distance/gradient based models (Logistic Regression,
    KNN, SVM). Tree models don't need it, hence the flag.
    """
    numeric_step = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, NUMERIC_FEATURES),
            (
                "cat",
                # ignore: a category never seen in training becomes all zeros
                # instead of crashing the app on new data
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def split_features_target(df: pd.DataFrame):
    """Return X (feature columns) and y as 0/1 (1 = subscribed)."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
    return X, y
