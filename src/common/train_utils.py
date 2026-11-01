"""Common training routine used by all five model scripts.

Each model script only defines its pipeline and its hyper-parameter grid;
this module does the tuning, the single evaluation on the held-out test set,
and the saving. That keeps the comparison between models fair: same split,
same CV folds, same metrics, same output format.
"""
import json

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.common.config import MODEL_NAMES, RANDOM_STATE, TEST_DATA_PATH, TRAIN_DATA_PATH, model_dir
from src.common.metrics import compute_metrics, positive_class_scores
from src.common.preprocessing import split_features_target


def load_train_test():
    X_train, y_train = split_features_target(pd.read_csv(TRAIN_DATA_PATH))
    X_test, y_test = split_features_target(pd.read_csv(TEST_DATA_PATH))
    return X_train, y_train, X_test, y_test


def tune_evaluate_save(
    pipeline,
    param_grid: dict,
    model_key: str,
    tuning_sample_size: int | None = None,
) -> dict:
    """Grid-search on the training set, refit, evaluate on test, save.

    tuning_sample_size: if set, the grid search runs on a stratified sample of
    this many training rows (used for SVM, whose training time grows roughly
    quadratically with rows). The final model is always refit on the FULL
    training set with the best parameters.
    """
    X_train, y_train, X_test, y_test = load_train_test()

    X_tune, y_tune = X_train, y_train
    if tuning_sample_size and tuning_sample_size < len(X_train):
        X_tune = X_train.groupby(y_train, group_keys=False).sample(
            frac=tuning_sample_size / len(X_train), random_state=RANDOM_STATE
        )
        y_tune = y_train.loc[X_tune.index]
        print(f"Tuning on a stratified sample of {len(X_tune)} rows")

    # ROC-AUC is the tuning target: with only ~11% positives, accuracy would
    # reward a model that simply predicts 'no' for everyone.
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(X_tune, y_tune)
    print(f"Best params: {search.best_params_}  (CV AUC {search.best_score_:.4f})")

    model = search.best_estimator_
    if X_tune is not X_train:
        model.fit(X_train, y_train)

    display_name = MODEL_NAMES[model_key]

    # --- single evaluation on the untouched test set ---
    y_pred = model.predict(X_test)
    y_score = positive_class_scores(model, X_test)
    results = compute_metrics(y_test, y_pred, y_score)

    print(f"\n{display_name} - test-set metrics")
    for name, value in results.items():
        print(f"  {name:9s} {value:.4f}")
    print("\nConfusion matrix [rows = actual no/yes, cols = predicted no/yes]")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + classification_report(y_test, y_pred, target_names=["no", "yes"], digits=4))

    out_dir = model_dir(model_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    summary = {
        "model": display_name,
        "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
        "cv_auc": round(search.best_score_, 4),
        "metrics": {k: round(v, 4) for k, v in results.items()},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved model/{model_key}/model.joblib and metrics.json")
    return summary
