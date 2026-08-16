"""Train the Logistic Regression baseline and InterpretML EBM primary model."""

import os
import tempfile

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

TARGET_COL = "Approved_Flag"
LABEL_MAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
MODEL_DIR = os.path.join(tempfile.gettempdir(), "xclearscore_models")


def _evaluate(model, X_test, y_test):
    prediction, probability = model.predict(X_test), model.predict_proba(X_test)
    return {
        "f1_macro": f1_score(y_test, prediction, average="macro"),
        "auc_ovr": roc_auc_score(y_test, probability, multi_class="ovr", average="macro"),
        "report": classification_report(y_test, prediction, target_names=[LABEL_MAP[i] for i in model.classes_], output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=model.classes_),
        "y_pred": prediction, "y_proba": probability,
    }


def _matrix(matrix, title):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=LABEL_MAP.values(), yticklabels=LABEL_MAP.values(), xlabel="Predicted", ylabel="Observed", title=title)
    for r in range(4):
        for c in range(4):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center")
    fig.tight_layout()
    return fig


def main():
    st.title("Train and validate models")
    st.caption("Logistic Regression baseline + Explainable Boosting Machine primary model.")
    if "processed_df" not in st.session_state:
        st.error("Prepare data before training.")
        return
    df = st.session_state["processed_df"]
    features = st.session_state.get("feature_names", [c for c in df if c != TARGET_COL])
    X, y = df[features], df[TARGET_COL]
    mode = st.radio("Execution profile", ["Community Cloud demo (10,000 rows)", "Local full validation (51,336 rows)"], horizontal=True)
    cap = 10_000 if mode.startswith("Community") else len(df)
    if mode.startswith("Local"):
        st.warning("Run the full profile locally. It produces the 10,268-row holdout used for the project benchmark and can exceed Community Cloud memory limits.")
    else:
        st.info("The cloud profile preserves the full EBM + SHAP + LIME stack while bounding training and explanation work. It is a demo run, not the published full-data benchmark.")
    if not st.button("Train baseline and EBM", type="primary"):
        return
    if len(df) > cap:
        X_fit, _, y_fit, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=42)
    else:
        X_fit, y_fit = X, y
    X_train, X_test, y_train, y_test = train_test_split(X_fit, y_fit, test_size=0.2, stratify=y_fit, random_state=42)
    with st.spinner("Training logistic-regression baseline..."):
        lr = LogisticRegression(max_iter=800, random_state=42, n_jobs=1).fit(X_train, y_train)
    with st.spinner("Training Explainable Boosting Machine..."):
        ebm = ExplainableBoostingClassifier(interactions=0, max_bins=64, outer_bags=4, random_state=42, n_jobs=1)
        ebm.fit(X_train, y_train)
    lr_metrics, ebm_metrics = _evaluate(lr, X_test, y_test), _evaluate(ebm, X_test, y_test)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(ebm, os.path.join(MODEL_DIR, "ebm_model.pkl"))
    st.session_state.update({"lr_model": lr, "ebm_model": ebm, "X_train": X_train, "X_test": X_test, "y_test": y_test, "model_metrics": {"lr": lr_metrics, "ebm": ebm_metrics, "n_train": len(X_train), "n_test": len(X_test)}})
    st.success(f"Training complete. Holdout: {len(X_test):,} applicants.")
    table = pd.DataFrame({"Logistic Regression": [lr_metrics["f1_macro"], lr_metrics["auc_ovr"]], "EBM (primary)": [ebm_metrics["f1_macro"], ebm_metrics["auc_ovr"]]}, index=["Macro F1", "Macro AUC (OvR)"])
    st.dataframe(table.round(3), use_container_width=True)
    fig = _matrix(ebm_metrics["confusion_matrix"], "EBM holdout confusion matrix")
    st.pyplot(fig); plt.close(fig)
    st.caption("Published benchmark: 95.6% accuracy and 0.982 macro AUC on a 10,268-applicant full-data holdout. Reproduce it with the Local full validation profile; do not substitute demo-run metrics for it.")


main()
