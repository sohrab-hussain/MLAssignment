"""Term Deposit Predictor - Streamlit app.

Loads the five pre-trained pipelines from model/<name>/model.joblib and
evaluates them on a CSV uploaded by the user (normally test_data.csv).
Nothing is retrained here: the app only calls predict / predict_proba.

Run locally:
    streamlit run app.py
"""
import hashlib
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.common.config import (
    FEATURE_COLUMNS,
    MODEL_NAMES,
    POSITIVE_LABEL,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    model_dir,
)
from src.common.metrics import compute_metrics, positive_class_scores

# --- visual constants ----------------------------------------------------------
BLUE = "#2a78d6"          # single-series colour for bars / ROC line
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
# sequential blue ramp for the confusion-matrix heatmap (light -> dark)
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
METRIC_HELP = {
    "Accuracy": "Share of all clients classified correctly. Inflated by the 89% 'no' majority.",
    "AUC": "Probability a random subscriber is scored higher than a random non-subscriber.",
    "Precision": "Of the clients predicted 'yes', the share who really subscribed.",
    "Recall": "Of the clients who really subscribed, the share the model found.",
    "F1": "Harmonic mean of precision and recall for the 'yes' class.",
    "MCC": "Correlation between predictions and truth, -1 to +1. Robust to class imbalance.",
}

st.set_page_config(page_title="Term Deposit Predictor", page_icon="🏦", layout="wide")


# --- loading -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading saved models…")
def load_models() -> dict:
    models = {}
    for key, name in MODEL_NAMES.items():
        path = model_dir(key) / "model.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


def read_csv(raw: bytes) -> pd.DataFrame:
    # Accept both comma-separated (our test_data.csv) and the original UCI ';' format.
    sample = raw[:2048].decode("utf-8", errors="ignore")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    df = pd.read_csv(io.BytesIO(raw), sep=sep)
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def encode_target(series: pd.Series) -> pd.Series:
    """Map yes/no (any case) or 1/0 to 1/0."""
    s = series.astype(str).str.strip().str.lower()
    return s.isin([POSITIVE_LABEL, "1", "true"]).astype(int)


@st.cache_data(show_spinner="Scoring…")
def evaluate(file_hash: str, _df: pd.DataFrame, model_name: str) -> dict:
    """Run one saved model on the uploaded data. Cached per (file, model)."""
    model = load_models()[model_name]
    X = _df[FEATURE_COLUMNS]
    y_pred = model.predict(X)
    y_score = positive_class_scores(model, X)
    out = {"y_pred": y_pred, "y_score": y_score}
    if TARGET_COLUMN in _df.columns:
        y_true = encode_target(_df[TARGET_COLUMN]).to_numpy()
        out["y_true"] = y_true
        out["metrics"] = compute_metrics(y_true, y_pred, y_score)
    return out


# --- charts --------------------------------------------------------------------
def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.xaxis.label.set_color(INK_MUTED)


def confusion_matrix_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(row_pct, cmap=SEQ_BLUE, vmin=0, vmax=1)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            dark_cell = row_pct[i, j] > 0.55
            ax.text(j, i - 0.12, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white" if dark_cell else INK)
            ax.text(j, i + 0.18, f"{labels[i][j]}\n{row_pct[i, j]:.1%} of row", ha="center",
                    va="center", fontsize=8, color="white" if dark_cell else INK_MUTED)
    ax.set_xticks([0, 1], ["no", "yes"])
    ax.set_yticks([0, 1], ["no", "yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(length=0, colors=INK_MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def roc_figure(y_true, y_score, auc):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#a8a7a1", linewidth=1, label="Random (0.50)")
    ax.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"Model (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_MUTED)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def metric_bar_figure(table: pd.DataFrame, metric: str):
    data = table[metric].sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    bars = ax.barh(data.index, data.values, color=BLUE, height=0.55)
    for bar, value in zip(bars, data.values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(metric)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style_axes(ax)
    fig.tight_layout()
    return fig


# --- page ----------------------------------------------------------------------
st.title("🏦 Term Deposit Subscription Predictor")
st.caption(
    "Bank Marketing dataset (UCI) · five classifiers trained offline and loaded from "
    "`model/` · upload held-out test data to evaluate them. No retraining happens in this app."
)

models = load_models()
if not models:
    st.error("No trained models found in `model/`. Run the training scripts in `src/models/` first.")
    st.stop()

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload a CSV with the 19 features and the `y` column", type="csv")
    if TEST_DATA_PATH.exists():
        st.download_button(
            "Download sample test_data.csv",
            data=TEST_DATA_PATH.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            help="The 20% hold-out split the README metrics were computed on.",
        )

    st.header("2 · Model")
    model_name = st.selectbox("Classifier", list(models.keys()))

    with st.expander("Expected columns"):
        st.write(", ".join(f"`{c}`" for c in FEATURE_COLUMNS))
        st.write(f"Target: `{TARGET_COLUMN}` (yes / no). `duration` is ignored if present.")

if uploaded is None:
    st.info("⬅️ Upload a test CSV in the sidebar to begin. "
            "No file handy? Download the sample `test_data.csv` there first.")
    st.stop()

raw = uploaded.getvalue()
file_hash = hashlib.md5(raw).hexdigest()
try:
    df = read_csv(raw)
except Exception as exc:  # malformed file
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
if missing:
    st.error("The file is missing required feature columns: " + ", ".join(f"`{c}`" for c in missing))
    st.stop()
has_target = TARGET_COLUMN in df.columns

n_yes = int(encode_target(df[TARGET_COLUMN]).sum()) if has_target else None
st.write(
    f"**{uploaded.name}** · {len(df):,} rows"
    + (f" · {n_yes:,} subscribers ({n_yes / len(df):.1%})" if has_target else "")
)
if not has_target:
    st.warning(f"No `{TARGET_COLUMN}` column found, so metrics can't be computed. Showing predictions only.")

tab_model, tab_compare, tab_preds = st.tabs(
    [f"📊 {model_name}", "🏁 Compare all models", "🔎 Predictions"]
)

result = evaluate(file_hash, df, model_name)

# --- tab 1: selected model -------------------------------------------------------
with tab_model:
    if has_target:
        m = result["metrics"]
        cols = st.columns(6)
        for col, name in zip(cols, METRIC_ORDER):
            col.metric(name, f"{m[name]:.4f}", help=METRIC_HELP[name])

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            st.pyplot(confusion_matrix_figure(result["y_true"], result["y_pred"]), width="content")
        with right:
            st.subheader("ROC curve")
            st.pyplot(roc_figure(result["y_true"], result["y_score"], m["AUC"]), width="content")

        st.subheader("Classification report")
        report = classification_report(
            result["y_true"], result["y_pred"], target_names=["no", "yes"], output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).T
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                             "f1-score": "{:.4f}", "support": "{:,}"}),
                     width="stretch")
    else:
        st.write("Upload a file with the target column to see metrics.")

# --- tab 2: all models side by side ---------------------------------------------
with tab_compare:
    if has_target:
        rows = {}
        for name in models:
            rows[name] = evaluate(file_hash, df, name)["metrics"]
        table = pd.DataFrame(rows).T[METRIC_ORDER]
        table.index.name = "Model"

        st.subheader("All five models on this file")
        st.caption("Best value in each column is highlighted.")
        st.dataframe(
            table.style.format("{:.4f}").highlight_max(axis=0, color="#cde2fb"),
            width="stretch",
        )

        metric = st.radio("Rank models by", METRIC_ORDER, index=5, horizontal=True)
        st.caption(METRIC_HELP[metric])
        st.pyplot(metric_bar_figure(table, metric), width="content")

        best = table[metric].idxmax()
        st.success(f"Best by **{metric}**: **{best}** ({table.loc[best, metric]:.4f})")
        st.download_button("Download comparison as CSV", table.to_csv().encode(),
                           file_name="model_comparison.csv", mime="text/csv")
    else:
        st.write("Upload a file with the target column to compare models.")

# --- tab 3: row-level predictions -----------------------------------------------
with tab_preds:
    preds = df[FEATURE_COLUMNS].copy()
    # SVM gives a decision score (distance from the boundary), the others a probability
    score_label = "score" if model_name == "Support Vector Machine" else "P(yes)"
    preds.insert(0, score_label, np.round(result["y_score"], 4))
    preds.insert(0, "predicted", np.where(result["y_pred"] == 1, "yes", "no"))
    if has_target:
        preds.insert(0, "actual", df[TARGET_COLUMN].values)
    st.dataframe(preds, width="stretch", height=420)
    st.download_button("Download predictions", preds.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv", mime="text/csv")
