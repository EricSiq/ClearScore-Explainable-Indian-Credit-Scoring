import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc as sklearn_auc,
)
from sklearn.preprocessing import label_binarize
from interpret.glassbox import ExplainableBoostingClassifier
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_COL = "Approved_Flag"
MODEL_DIR  = "app/models"
_LABEL_MAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}

# Colour palette — one per class, consistent across all charts
CLASS_COLOURS = {
    "P1": "#2ecc71",   # green  — excellent
    "P2": "#3498db",   # blue   — good
    "P3": "#f39c12",   # amber  — marginal
    "P4": "#e74c3c",   # red    — poor
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _label_map() -> dict:
    return st.session_state.get("label_map", _LABEL_MAP)


def _class_dist_df(y: pd.Series, label_map: dict) -> pd.DataFrame:
    """Return a tidy DataFrame of class counts and percentages."""
    dist = (
        y.map(label_map)
         .value_counts()
         .reindex(["P1", "P2", "P3", "P4"])
         .fillna(0)
         .astype(int)
         .rename("Count")
         .to_frame()
    )
    dist["Pct"] = (dist["Count"] / dist["Count"].sum() * 100).round(1).astype(str) + "%"
    dist.index.name = "Class"
    return dist


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, label_map: dict) -> dict:
    """
    Compute the full evaluation suite for one model.

    Returns a dict containing:
      - Headline scalars: f1_macro, f1_weighted, auc_ovr
      - Per-class report dict (from classification_report)
      - Confusion matrix (np.ndarray)
      - Per-class ROC curve data: {class_int: (fpr, tpr, roc_auc)}
      - Raw predictions and probabilities for downstream use
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)   # shape (n, 4)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    f1_macro    = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    auc_ovr     = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    # ── Per-class classification report ──────────────────────────────────────
    class_labels = [label_map[i] for i in sorted(label_map)]
    report = classification_report(
        y_test, y_pred,
        target_names=class_labels,
        output_dict=True,
    )

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)

    # ── Per-class OvR ROC curves ──────────────────────────────────────────────
    classes   = sorted(label_map.keys())          # [0, 1, 2, 3]
    y_bin     = label_binarize(y_test, classes=classes)   # (n, 4)
    roc_data  = {}
    for i, cls_int in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc_cls  = sklearn_auc(fpr, tpr)
        roc_data[cls_int] = (fpr, tpr, roc_auc_cls)

    return {
        # Headline scalars
        "f1_macro":    f1_macro,
        "f1_weighted": f1_weighted,
        "auc_ovr":     auc_ovr,
        # Detailed outputs
        "classification_report": report,
        "confusion_matrix":      cm,
        "roc_data":              roc_data,
        # Raw arrays (used by Page 6 Fairness and Page 8 Business Summary)
        "y_pred":  y_pred,
        "y_proba": y_proba,
    }


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, title: str, label_map: dict) -> plt.Figure:
    """Annotated confusion matrix heatmap with percentage annotations."""
    class_labels = [label_map[i] for i in sorted(label_map)]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=np.array([
            [f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
            for row_v, row_p in zip(cm, cm_pct)
        ]),
        fmt="",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def _plot_roc_curves(roc_data: dict, label_map: dict, title: str) -> plt.Figure:
    """
    One-vs-Rest ROC curves for all four classes on a single axes.
    Each class gets its own colour; macro-average AUC shown in legend.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    for cls_int, (fpr, tpr, roc_auc) in roc_data.items():
        cls_label = label_map[cls_int]
        colour    = CLASS_COLOURS.get(cls_label, "#555555")
        ax.plot(
            fpr, tpr,
            color=colour,
            lw=2,
            label=f"{cls_label} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    return fig


def _plot_class_dist_comparison(
    y_train: pd.Series,
    y_test: pd.Series,
    label_map: dict,
) -> plt.Figure:
    """
    Side-by-side bar chart comparing class distribution in train vs test split.
    Confirms stratification worked correctly.
    """
    classes = ["P1", "P2", "P3", "P4"]

    train_pct = (
        y_train.map(label_map).value_counts(normalize=True)
        .reindex(classes).fillna(0) * 100
    )
    test_pct = (
        y_test.map(label_map).value_counts(normalize=True)
        .reindex(classes).fillna(0) * 100
    )

    x     = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars_train = ax.bar(x - width/2, train_pct, width, label="Train",
                        color=[CLASS_COLOURS[c] for c in classes], alpha=0.85)
    bars_test  = ax.bar(x + width/2, test_pct,  width, label="Test",
                        color=[CLASS_COLOURS[c] for c in classes], alpha=0.45,
                        edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("% of split", fontsize=10)
    ax.set_title("Class Distribution — Train vs Test", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    # Annotate bars with percentage values
    for bar in list(bars_train) + list(bars_test):
        h = bar.get_height()
        if h > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                f"{h:.1f}%",
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()
    return fig


def _show_metrics(metrics: dict, model_name: str, label_map: dict):
    """
    Render the full evaluation panel for one model:
      Row 1 — headline metric cards
      Row 2 — classification report table
      Row 3 — confusion matrix | ROC curves (side by side)
    """
    # ── Headline cards ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Macro F1",    f"{metrics['f1_macro']:.3f}",
                help="Unweighted average F1 across all 4 classes")
    col2.metric("Weighted F1", f"{metrics['f1_weighted']:.3f}",
                help="F1 weighted by class support — reflects real-world class imbalance")
    col3.metric("AUC (OvR)",   f"{metrics['auc_ovr']:.3f}",
                help="Macro-averaged one-vs-rest ROC AUC across all 4 classes")

    # ── Per-class classification report ──────────────────────────────────────
    report_df = pd.DataFrame(metrics["classification_report"]).T
    keep_rows = [label_map[i] for i in sorted(label_map)] + ["macro avg", "weighted avg"]
    report_df = report_df.loc[report_df.index.isin(keep_rows)]
    report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
    report_df["support"] = report_df["support"].astype("Int64")
    st.dataframe(
        report_df.style.background_gradient(subset=["f1-score"], cmap="Blues"),
        use_container_width=True,
    )

    # ── Confusion matrix + ROC curves side by side ───────────────────────────
    left, right = st.columns(2)

    with left:
        fig_cm = _plot_confusion_matrix(
            metrics["confusion_matrix"],
            f"{model_name} — Confusion Matrix",
            label_map,
        )
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with right:
        fig_roc = _plot_roc_curves(
            metrics["roc_data"],
            label_map,
            f"{model_name} — ROC Curves (OvR)",
        )
        st.pyplot(fig_roc)
        plt.close(fig_roc)


# ── Main page ─────────────────────────────────────────────────────────────────

def main():
    st.title("🤖 Train Models")

    # ── Guard ─────────────────────────────────────────────────────────────────
    if "processed_df" not in st.session_state:
        st.error("⚠ Run **Preprocessing** first.")
        return

    df = st.session_state["processed_df"]

    if TARGET_COL not in df.columns:
        st.error(
            f"Target column `{TARGET_COL}` not found in processed data. "
            "Re-run preprocessing."
        )
        return

    feature_names = st.session_state.get(
        "feature_names",
        [c for c in df.columns if c != TARGET_COL],
    )
    X         = df[feature_names]   # named DataFrame — EBM reads feature_names_in_ from this
    y         = df[TARGET_COL]
    label_map = _label_map()

    # ── Dataset overview ──────────────────────────────────────────────────────
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", f"{len(df):,}")
    col2.metric("Features",      len(feature_names))
    col3.metric("Classes",       y.nunique())

    with st.expander("Full class distribution"):
        st.table(_class_dist_df(y, label_map))

    st.markdown("---")

    # ── Train ─────────────────────────────────────────────────────────────────
    if st.button("▶ Train Models"):

        # ── Split ─────────────────────────────────────────────────────────────
        with st.spinner("Splitting data (80 / 20 stratified)..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

        # Split summary with stratification verification
        st.subheader("Train / Test Split")
        s1, s2 = st.columns(2)
        s1.metric("Train samples", f"{len(X_train):,}",
                  help="80% of total, stratified by class")
        s2.metric("Test samples",  f"{len(X_test):,}",
                  help="20% of total, stratified by class")

        fig_dist = _plot_class_dist_comparison(y_train, y_test, label_map)
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        st.caption(
            "Bar heights should be nearly identical between Train and Test — "
            "confirming stratification preserved class proportions."
        )

        st.markdown("---")

        # ── Logistic Regression ───────────────────────────────────────────────
        # multi_class param removed in sklearn 1.5+; lbfgs uses multinomial automatically.
        with st.spinner("Training Logistic Regression..."):
            lr_model = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            lr_model.fit(X_train, y_train)

        # ── EBM ───────────────────────────────────────────────────────────────
        # X_train is a named pd.DataFrame — EBM stores feature_names_in_ from it.
        # This is what keeps real column names in all downstream explainability charts.
        with st.spinner("Training Explainable Boosting Machine (~2–3 min)..."):
            ebm_model = ExplainableBoostingClassifier(random_state=42)
            ebm_model.fit(X_train, y_train)

        # ── Persist ───────────────────────────────────────────────────────────
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(lr_model,  os.path.join(MODEL_DIR, "logistic_regression.pkl"))
        joblib.dump(ebm_model, os.path.join(MODEL_DIR, "ebm_model.pkl"))

        st.session_state["lr_model"]  = lr_model
        st.session_state["ebm_model"] = ebm_model
        st.session_state["X_test"]    = X_test
        st.session_state["y_test"]    = y_test
        st.session_state["X_train"]   = X_train
        st.session_state["y_train"]   = y_train

        st.success("✅ Models trained and saved to `app/models/`")

        # EBM feature name sanity check
        ebm_features = getattr(ebm_model, "feature_names_in_", None)
        if ebm_features is not None:
            st.caption(
                f"EBM feature names confirmed — "
                f"first 5: {list(ebm_features[:5])}"
            )
        else:
            st.warning(
                "EBM `feature_names_in_` not set — "
                "explainability charts may show generic indices."
            )

        # ── Evaluate ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Model Evaluation")

        with st.spinner("Evaluating both models on test set..."):
            lr_metrics  = _evaluate(lr_model,  X_test, y_test, label_map)
            ebm_metrics = _evaluate(ebm_model, X_test, y_test, label_map)

        # ── Store full metrics schema for downstream pages ────────────────────
        # Page 6 (Fairness) uses y_pred / y_proba.
        # Page 8 (Business Summary) uses f1_macro, auc_ovr, confusion_matrix,
        # n_train, n_test, class_counts.
        st.session_state["model_metrics"] = {
            "lr":  {
                "f1_macro":             lr_metrics["f1_macro"],
                "f1_weighted":          lr_metrics["f1_weighted"],
                "auc_ovr":              lr_metrics["auc_ovr"],
                "confusion_matrix":     lr_metrics["confusion_matrix"],
                "classification_report": lr_metrics["classification_report"],
                "y_pred":               lr_metrics["y_pred"],
                "y_proba":              lr_metrics["y_proba"],
            },
            "ebm": {
                "f1_macro":             ebm_metrics["f1_macro"],
                "f1_weighted":          ebm_metrics["f1_weighted"],
                "auc_ovr":              ebm_metrics["auc_ovr"],
                "confusion_matrix":     ebm_metrics["confusion_matrix"],
                "classification_report": ebm_metrics["classification_report"],
                "y_pred":               ebm_metrics["y_pred"],
                "y_proba":              ebm_metrics["y_proba"],
            },
            "n_train":      len(X_train),
            "n_test":       len(X_test),
            "class_counts": y.value_counts().to_dict(),
            # Train/test class distributions for business summary
            "train_class_dist": y_train.value_counts().to_dict(),
            "test_class_dist":  y_test.value_counts().to_dict(),
        }

        # ── Per-model tabs ────────────────────────────────────────────────────
        tab_lr, tab_ebm = st.tabs(["Logistic Regression", "EBM"])
        with tab_lr:
            _show_metrics(lr_metrics, "Logistic Regression", label_map)
        with tab_ebm:
            _show_metrics(ebm_metrics, "EBM", label_map)

        # ── Side-by-side headline comparison ──────────────────────────────────
        st.markdown("---")
        st.subheader("Model Comparison")

        comparison = pd.DataFrame({
            "Model":       ["Logistic Regression", "EBM"],
            "Macro F1":    [lr_metrics["f1_macro"],    ebm_metrics["f1_macro"]],
            "Weighted F1": [lr_metrics["f1_weighted"],  ebm_metrics["f1_weighted"]],
            "AUC (OvR)":   [lr_metrics["auc_ovr"],     ebm_metrics["auc_ovr"]],
        }).round(4).set_index("Model")

        st.dataframe(
            comparison.style.highlight_max(axis=0, color="#d4edda"),
            use_container_width=True,
        )
        st.caption(
            "Green cells = better model for that metric. "
            "EBM is expected to outperform LR on all three — "
            "if it doesn't, check that preprocessing ran correctly."
        )

        st.info("➡ Proceed to **Explainability** to inspect SHAP, LIME, and PDP.")


if __name__ == "__main__":
    main()
