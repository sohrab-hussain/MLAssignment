"""Model 5 of 5 - Support Vector Machine (RBF kernel).

Run from the project root (after python -m src.data_prep):
    python -m src.models.svm.train      # takes a few minutes

Finds the boundary with the widest margin between the classes. The RBF kernel
lets that boundary be non-linear.

Training time: kernel SVMs scale roughly with the square of the number of
rows, so the grid search runs on a 6,000-row stratified sample. The chosen
parameters are then used to refit on the full 32,940-row training set.
"""
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.common.config import RANDOM_STATE
from src.common.preprocessing import build_preprocessor
from src.common.train_utils import tune_evaluate_save


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            # scaling is essential: the RBF kernel is distance based
            ("preprocess", build_preprocessor(scale_numeric=True)),
            # probability=False: AUC is computed from decision_function instead,
            # which avoids an extra internal 5-fold fit (≈5x slower)
            ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]
    )


# C: penalty for misclassified training points (higher = tighter fit).
# gamma: how far one training point's influence reaches (higher = more local).
# class_weight: 'balanced' makes an error on the rare 'yes' class cost ~8x
# more than one on 'no'. Without it the SVM margin is dominated by the 89%
# majority class and its scores rank 'yes' clients poorly (CV AUC ~0.72 vs
# ~0.76 with balancing in our experiments) - so we let the grid search decide.
PARAM_GRID = {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__gamma": ["scale", 0.01],
    "clf__class_weight": [None, "balanced"],
}


if __name__ == "__main__":
    tune_evaluate_save(
        build_pipeline(), PARAM_GRID, "svm",
        tuning_sample_size=6000,
    )
