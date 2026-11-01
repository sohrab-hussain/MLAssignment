"""Model 3 of 5 - K-Nearest Neighbors.

Run from the project root (after python -m src.data_prep):
    python -m src.models.knn.train

No real "training": the model stores the training rows and, for a new client,
looks at the k most similar stored clients and takes a (weighted) vote.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.common.preprocessing import build_preprocessor
from src.common.train_utils import tune_evaluate_save


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            # scaling is essential: KNN uses Euclidean distance, so unscaled
            # nr.employed (~5000) would drown out everything else
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("clf", KNeighborsClassifier()),
        ]
    )


# Small k = noisy, flexible boundary; large k = smoother boundary.
# 'distance' weighting lets closer neighbours count more.
PARAM_GRID = {
    "clf__n_neighbors": [5, 15, 25, 51, 75, 101],
    "clf__weights": ["uniform", "distance"],
}


if __name__ == "__main__":
    tune_evaluate_save(build_pipeline(), PARAM_GRID, "knn")
