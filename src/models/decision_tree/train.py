"""Model 2 of 5 - Decision Tree Classifier.

Run from the project root (after python -m src.data_prep):
    python -m src.models.decision_tree.train

Splits the data with a sequence of if/else rules on single features.
Trees only compare a feature against a threshold, so scaling is unnecessary.
"""
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.common.config import RANDOM_STATE
from src.common.preprocessing import build_preprocessor
from src.common.train_utils import tune_evaluate_save


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )


# An unrestricted tree memorises the training set (overfits). max_depth and
# min_samples_leaf limit how specific the rules can get.
PARAM_GRID = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [4, 6, 8, 10, None],
    "clf__min_samples_leaf": [1, 20, 50, 100],
}


if __name__ == "__main__":
    tune_evaluate_save(build_pipeline(), PARAM_GRID, "decision_tree")
