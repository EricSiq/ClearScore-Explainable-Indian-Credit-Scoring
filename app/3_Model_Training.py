import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, roc_curve, auc as sklearn_auc,
)
from sklearn.preprocessing import label_binarize
from interpret.glassbox import ExplainableBoostingClassifier
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

TARGET_COL  = "Approved_Flag"
_LABEL_MAP  = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}

# Model directory: /tmp on Streamlit Cloud (read-only repo), app/models/ locally
_TMP_MODELS = os.path.join(tempfile.gettempdir(), "creditlens_models")
MODEL_DIR   = _TMP_MODELS if not os.access("app/models", os.W_OK) else "app/models"

CLASS_COLOURS = {
    "P1": "#2ecc71",
    "P2": "#3498db",
    "P3": "#f39c12",
    "P4": "#e74c3c",
}


def _label_map() -> dict:
    return st.session_state.get("label_map", _LABEL_MAP)


def _class_dist_df(y: pd.Series, label_map: dict) -> pd.DataFrame:
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


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, label_map: dict) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    f1_macro    = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    auc_ovr     = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    class_labels = [label_map[i] for i in sorted(label_map)]
    report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    classes  = sorted(label_map.keys())
    y_bin    = label_binarize(y_test, classes=classes)
    roc_data = {}
    for i, cls_int in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_data[cls_int] = (fpr, tpr, sklearn_auc(fpr, tpr))
    return {
        "f1_macro": f1_macro, "f1_weighted": f1_weighted, "auc_ovr": auc_ovr,
        "classification_report": report, "confusion_matrix": cm, "roc_data": roc_data,
        "y_pred": y_pred, "y_proba": y_proba,
    }


def _plot_confusion_matrix(cm: np.ndarray, title: str, label_map: dict) -> plt.Figure:
    class_labels = [label_map[i] for i in sorted(label_map)]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(rv, rp)] for rv, rp in zip(cm, cm_pct)]),
        fmt="", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def _plot_roc_curves(roc_data: dict, label_map: dict, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    for cls_int, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(fpr, tpr, color=CLASS_COLOURS.get(label_map[cls_int], "#555"), lw=2,
                label=f"{label_map[cls_int]} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    return fig


def _plot_class_dist_comparison(y_train: pd.Series, y_test: pd.Series, label_map: dict) -> plt.Figure:
    classes = ["P1", "P2", "P3", "P4"]
    train_pct = y_train.map(label_map).value_counts(normalize=True).reindex(classes).fillna(0) * 100
    test_pct  = y_test.map(label_map).value_counts(normalize=True).reindex(classes).fillna(0) * 100
    x = np.arange(len(classes)); width = 0.35
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars_train = ax.bar(x - width/2, train_pct, width, label="Train",
                        color=[CLASS_COLOURS[c] for c in classes], alpha=0.85)
    bars_test  = ax.bar(x + width/2, test_pct,  width, label="Test",
                        color=[CLASS_COLOURS[c] for c in classes], alpha=0.45,
                        edgecolor="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("% of split", fontsize=10)
    ax.set_title("Class Distribution — Train vs Test", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    for bar in list(bars_train) + list(bars_test):
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    return fig


def _show_metrics(metrics: dict, model_name: str, label_map: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("Macro F1",    f"{metrics['f1_macro']:.3f}",
                help="Unweighted average F1 across P1, P2, P3, P4. Best choice for imbalanced classes — treats all four tiers equally regardless of size.")
    col2.metric("Weighted F1", f"{metrics['f1_weighted']:.3f}",
                help="F1 weighted by class support. Reflects real-world performance accounting for the fact that P2 (62.7% of applicants) dominates the portfolio.")
    col3.metric("AUC (OvR)",   f"{metrics['auc_ovr']:.3f}",
                help="One-vs-Rest ROC AUC macro-averaged across all 4 tiers. Measures the model's ability to rank applicants correctly — 1.0 = perfect separation, 0.5 = random.")

    report_df = pd.DataFrame(metrics["classification_report"]).T
    keep_rows = [label_map[i] for i in sorted(label_map)] + ["macro avg", "weighted avg"]
    report_df = report_df.loc[report_df.index.isin(keep_rows)]
    report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
    report_df["support"] = report_df["support"].astype("Int64")
    st.dataframe(report_df.style.background_gradient(subset=["f1-score"], cmap="Blues"),
                 use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig_cm = _plot_confusion_matrix(metrics["confusion_matrix"],
                                        f"{model_name} — Confusion Matrix", label_map)
        st.pyplot(fig_cm); plt.close(fig_cm)
    with right:
        fig_roc = _plot_roc_curves(metrics["roc_data"], label_map,
                                   f"{model_name} — ROC Curves (OvR)")
        st.pyplot(fig_roc); plt.close(fig_roc)


def main():
    st.title("Train Models")
    st.caption("CreditLens · Explainable AI Credit Scoring")

    if "processed_df" not in st.session_state:
        st.error(
            "No preprocessed data found. "
            "Run the full pipeline: **Home > Launch Demo** (or Upload Data > Preprocess) > then return here."
        )
        return

    df = st.session_state["processed_df"]
    if TARGET_COL not in df.columns:
        st.error(f"Target column `{TARGET_COL}` not found. Re-run preprocessing.")
        return

    feature_names = st.session_state.get("feature_names",
                                         [c for c in df.columns if c != TARGET_COL])
    X = df[feature_names]
    y = df[TARGET_COL]
    label_map = _label_map()

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", f"{len(df):,}")
    col2.metric("Features", len(feature_names),
                help="80 numeric + 23 OHE-expanded categorical = 103 total after preprocessing")
    col3.metric("Classes", y.nunique(),
                help="P1 (Excellent), P2 (Good), P3 (Marginal), P4 (Poor)")
    with st.expander("Full class distribution"):
        st.table(_class_dist_df(y, label_map))

    # Model configuration transparency
    with st.expander("Model configuration"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Logistic Regression (baseline)**")
            st.code(
                "solver='lbfgs'         # efficient for multinomial, handles L2\n"
                "max_iter=1000          # sufficient for 103-feature OHE input\n"
                "multi_class=multinomial # automatic in sklearn 1.5+ with lbfgs\n"
                "random_state=42",
                language="python"
            )
        with c2:
            st.markdown("**EBM — Explainable Boosting Machine (primary)**")
            st.code(
                "# InterpretML ExplainableBoostingClassifier\n"
                "random_state=42\n"
                "# Pairwise interactions auto-detected\n"
                "# Shape functions learned per feature\n"
                "# Explanations exact (not post-hoc SHAP)",
                language="python"
            )
        st.caption(
            "Train/test split: 80/20, stratified by class to preserve P1–P4 proportions. "
            "Stratification ensures the 11.3% P1 minority class is represented in both splits."
        )

    # Sample size selector — keeps memory within Streamlit Cloud free tier (1 GB RAM)
    st.subheader("Training Configuration")
    sample_pct = st.select_slider(
        "Training sample size",
        options=[10, 25, 50, 75, 100],
        value=100,
        format_func=lambda x: f"{x}%  ({int(len(df) * x / 100):,} rows)",
        help=(
            "Use 100% for full accuracy. "
            "Reduce to 25-50% if running on Streamlit Cloud free tier (1 GB RAM limit). "
            "EBM on 51K rows requires ~800 MB."
        ),
    )
    if sample_pct < 100:
        n_sample = int(len(df) * sample_pct / 100)
        st.info(
            f"Training on {n_sample:,} rows ({sample_pct}% sample, stratified). "
            "Metrics will be slightly lower than full-dataset results."
        )

    st.markdown("---")

    if st.button("Train Models"):
        # Apply sample if requested
        if sample_pct < 100:
            n_sample = int(len(df) * sample_pct / 100)
            X_s, _, y_s, _ = train_test_split(
                X, y, train_size=n_sample, random_state=42, stratify=y
            )
        else:
            X_s, y_s = X, y

        with st.spinner("Splitting data (80/20 stratified)..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X_s, y_s, test_size=0.2, random_state=42, stratify=y_s)

        st.subheader("Train / Test Split")
        s1, s2 = st.columns(2)
        s1.metric("Train samples", f"{len(X_train):,}")
        s2.metric("Test samples",  f"{len(X_test):,}")
        fig_dist = _plot_class_dist_comparison(y_train, y_test, label_map)
        st.pyplot(fig_dist); plt.close(fig_dist)
        st.caption("Bar heights should be nearly identical — confirming stratification.")
        st.markdown("---")

        with st.spinner("Training Logistic Regression..."):
            try:
                lr_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
                lr_model.fit(X_train, y_train)
            except Exception as e:
                st.error(f"Logistic Regression failed: {e}")
                return

        with st.spinner("Training Explainable Boosting Machine (~2-3 min on full data)..."):
            try:
                ebm_model = ExplainableBoostingClassifier(random_state=42)
                ebm_model.fit(X_train, y_train)
            except MemoryError:
                st.error(
                    "Out of memory training EBM. "
                    "Set the Training sample size slider to **25%** and try again. "
                    "Streamlit Cloud free tier has 1 GB RAM — EBM on 51K rows needs ~800 MB."
                )
                return
            except Exception as e:
                st.error(f"EBM training failed: {e}")
                return

        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(lr_model,  os.path.join(MODEL_DIR, "logistic_regression.pkl"))
            joblib.dump(ebm_model, os.path.join(MODEL_DIR, "ebm_model.pkl"))
            st.caption(f"Models persisted to `{MODEL_DIR}`")
        except Exception as e:
            st.warning(
                f"Could not save models to disk ({e}). "
                "Models are held in session memory — they will be lost on page refresh."
            )

        st.session_state["lr_model"]  = lr_model
        st.session_state["ebm_model"] = ebm_model
        st.session_state["X_test"]    = X_test
        st.session_state["y_test"]    = y_test
        st.session_state["X_train"]   = X_train
        st.session_state["y_train"]   = y_train

        st.success("Models trained successfully.")

        ebm_features = getattr(ebm_model, "feature_names_in_", None)
        if ebm_features is not None:
            st.caption(f"EBM feature names confirmed — first 5: {list(ebm_features[:5])}")
        else:
            st.warning("EBM feature_names_in_ not set — charts may show generic indices.")

        st.markdown("---")
        st.subheader("Model Evaluation")

        with st.spinner("Evaluating both models on test set..."):
            try:
                lr_metrics  = _evaluate(lr_model,  X_test, y_test, label_map)
                ebm_metrics = _evaluate(ebm_model, X_test, y_test, label_map)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                return

        st.session_state["model_metrics"] = {
            "lr":  {k: lr_metrics[k]  for k in ["f1_macro","f1_weighted","auc_ovr",
                    "confusion_matrix","classification_report","y_pred","y_proba"]},
            "ebm": {k: ebm_metrics[k] for k in ["f1_macro","f1_weighted","auc_ovr",
                    "confusion_matrix","classification_report","y_pred","y_proba"]},
            "n_train": len(X_train), "n_test": len(X_test),
            "class_counts": y.value_counts().to_dict(),
            "train_class_dist": y_train.value_counts().to_dict(),
            "test_class_dist":  y_test.value_counts().to_dict(),
        }

        tab_lr, tab_ebm = st.tabs(["Logistic Regression", "EBM"])
        with tab_lr:
            _show_metrics(lr_metrics, "Logistic Regression", label_map)
        with tab_ebm:
            _show_metrics(ebm_metrics, "EBM", label_map)

        st.markdown("---")
        st.subheader("Model Comparison")
        comparison = pd.DataFrame({
            "Model":       ["Logistic Regression", "EBM"],
            "Macro F1":    [lr_metrics["f1_macro"],    ebm_metrics["f1_macro"]],
            "Weighted F1": [lr_metrics["f1_weighted"],  ebm_metrics["f1_weighted"]],
            "AUC (OvR)":   [lr_metrics["auc_ovr"],     ebm_metrics["auc_ovr"]],
        }).round(4).set_index("Model")
        st.dataframe(comparison.style.highlight_max(axis=0, color="#d4edda"),
                     use_container_width=True)
        st.caption("Green cells = better model for that metric.")

        # Recommendation based on live results
        ebm_f1  = ebm_metrics["f1_macro"]
        lr_f1   = lr_metrics["f1_macro"]
        ebm_auc = ebm_metrics["auc_ovr"]
        lr_auc  = lr_metrics["auc_ovr"]
        if ebm_f1 > lr_f1 and ebm_auc > lr_auc:
            st.success(
                f"**Recommended model: EBM.** "
                f"Outperforms Logistic Regression on both Macro F1 "
                f"({ebm_f1:.3f} vs {lr_f1:.3f}) and AUC "
                f"({ebm_auc:.3f} vs {lr_auc:.3f}). "
                "EBM is also the correct choice for regulatory compliance — "
                "its shape functions provide exact, auditable explanations that satisfy "
                "RBI MRM documentation requirements without post-hoc approximation."
            )
        elif lr_f1 >= ebm_f1:
            st.info(
                "Logistic Regression matches or exceeds EBM on this sample size. "
                "Try 100% training data for a more representative comparison — "
                "EBM typically outperforms LR on the full 51K dataset."
            )

        st.info("Proceed to Explainability to inspect SHAP, LIME, and PDP.")


main()
