"""Model 1 of 5 - Logistic Regression.

Run from the project root (after python -m src.data_prep):
    python -m src.models.logistic_regression.train

A linear model: learns one weight per (scaled / one-hot) feature and passes
the weighted sum through a sigmoid to get P(subscribe).
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.common.config import RANDOM_STATE
from src.common.preprocessing import build_preprocessor
from src.common.train_utils import tune_evaluate_save


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            # scaling matters: otherwise features with large ranges
            # (e.g. nr.employed ~5000) dominate the L2 penalty
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


# C = inverse regularisation strength. Smaller C = stronger penalty = simpler model.
PARAM_GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0]}


if __name__ == "__main__":
    tune_evaluate_save(build_pipeline(), PARAM_GRID, "logistic_regression")
