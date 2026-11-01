"""Model 4 of 5 - Naive Bayes (Gaussian).

Run from the project root (after python -m src.data_prep):
    python -m src.models.naive_bayes.train

Applies Bayes' theorem assuming every feature is independent of the others
given the class, and that each feature is normally distributed within a class.

Why Gaussian and not Multinomial? Multinomial NB expects non-negative counts
(e.g. word frequencies). Our standardised numeric columns contain negative
values, so GaussianNB is the appropriate variant here.
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.common.preprocessing import build_preprocessor
from src.common.train_utils import tune_evaluate_save


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("clf", GaussianNB()),
        ]
    )


# var_smoothing adds a small value to every feature's variance. The one-hot
# columns are 0/1 and can have near-zero variance within a class, which makes
# the Gaussian likelihood explode; more smoothing stabilises that.
PARAM_GRID = {"clf__var_smoothing": np.logspace(-9, 0, 10)}


if __name__ == "__main__":
    tune_evaluate_save(build_pipeline(), PARAM_GRID, "naive_bayes")
