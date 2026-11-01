"""The six evaluation metrics required by the assignment.

Used both by the training scripts and by the Streamlit app, so the numbers in
the README and in the app are computed the same way.
"""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_class_scores(model, X):
    """Score for class 1, used for AUC.

    Most models give a probability via predict_proba. SVC is trained without
    probability=True (that would add a slow internal 5-fold calibration), so
    we use its decision_function instead - AUC only needs a ranking score,
    not a calibrated probability.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """y_pred = predicted class (0/1); y_score = probability of class 1.

    AUC must use the probability (or decision score), not the hard 0/1
    prediction - otherwise it only reflects a single threshold.
    Precision/recall/F1 are for the positive class ('yes' = subscribed).
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
