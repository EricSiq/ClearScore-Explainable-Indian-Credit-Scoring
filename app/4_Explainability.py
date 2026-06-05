import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

from app.components.model_loader import load_model as _load_model_from_disk
from app.components.utils        import get_label_map, get_X_y
from app.components.shap_plotter import get_shap_values, shap_global_plot, shap_local_waterfall
from app.components.lime_plotter import lime_local_explanation
from app.components.pdp_plotter  import plot_pdp

TARGET_COL = "Approved_Flag"
MODEL_DIR  = "app/models"
_LABEL_MAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}


def _load_model(filename, session_key):
    return _load_model_from_disk(filename, session_key)


def _get_shap_values(model, X, model_name):
    return get_shap_values(model, X, f"train_{model_name}")


def main():
    st.title("Explainability — SHAP, LIME, PDP")
    st.caption(
        "Three complementary techniques for understanding the model's decisions: "
        "SHAP (global and local feature attribution), LIME (local linear approximation), "
        "and PDP (feature-response relationship)."
    )

    X, y = get_X_y()
    if X is None:
        st.error(
            "No preprocessed data found. "
            "Run the pipeline: Home > Launch Demo (or Upload Data) > Preprocess > Train Models."
        )
        return

    lr  = _load_model("logistic_regression.pkl", "lr_model")
    ebm = _load_model("ebm_model.pkl", "ebm_model")

    model_options = {}
    if lr:
        model_options["EBM (recommended)"] = ("EBM", ebm) if ebm else None
        model_options["Logistic Regression"] = ("LR", lr)
    if ebm:
        model_options = {}
        model_options["EBM (recommended)"] = ("EBM", ebm)
        if lr:
            model_options["Logistic Regression"] = ("LR", lr)

    if not model_options:
        st.error("No trained models found. Run Train Models first.")
        return

    model_display = st.selectbox("Model", list(model_options.keys()))
    model_key, model = model_options[model_display]

    if model is None:
        st.error("Selected model not available. Run Train Models first.")
        return

    st.markdown("---")

    # ── Section 1: Global SHAP — auto-computed on load ────────────────────────
    st.subheader("1 — Global Feature Importance (SHAP)")

    with st.expander("What this shows"):
        st.markdown(
            "**Mean absolute SHAP value per feature** across all training applicants. "
            "A high value for `num_times_delinquent` means delinquency history is the "
            "strongest predictor of credit tier — consistent with standard credit risk theory. "
            "For EBM, SHAP values equal the model's own shape function values exactly. "
            "This is not a post-hoc approximation."
        )

    max_display = st.slider("Features to display", min_value=5, max_value=30, value=15)

    # Auto-compute if SHAP values are already cached from training (fast path)
    cache_key = f"shap_values_train_{model_key}"
    if cache_key in st.session_state:
        st.caption("SHAP values loaded from cache (computed during training).")
        shap_global_plot(model, X, f"train_{model_key}", max_display)
    else:
        st.caption("Click below to compute SHAP values for the full training set (~30–60 seconds).")
        if st.button("Compute Global SHAP Summary"):
            with st.spinner("Computing SHAP values and caching..."):
                shap_global_plot(model, X, f"train_{model_key}", max_display)

    st.markdown("---")

    # ── Section 2: Local SHAP + LIME ─────────────────────────────────────────
    st.subheader("2 — Local Explanations (per applicant)")

    with st.expander("SHAP waterfall vs LIME — when to use which"):
        st.markdown(
            "**SHAP waterfall**: Decomposes one prediction into additive feature contributions. "
            "For EBM this is exact — the bar length equals the shape function output for that "
            "feature value. Use this when you need an auditable, defensible explanation.\n\n"
            "**LIME**: Fits a local linear model around the applicant's neighbourhood in feature "
            "space. Less exact than SHAP for EBM, but produces a simpler linear equation "
            "that some audiences find more intuitive. Use this as a cross-check."
        )

    idx = st.number_input(
        "Applicant row index",
        min_value=0, max_value=max(0, len(X) - 1), value=0, step=1,
        help="Select any row from the training set to explain"
    )

    col_shap, col_lime = st.columns(2)
    with col_shap:
        if st.button("Show SHAP Waterfall", use_container_width=True):
            with st.spinner("Computing SHAP waterfall..."):
                shap_local_waterfall(model, X, f"train_{model_key}", int(idx))
    with col_lime:
        if st.button("Show LIME Explanation", use_container_width=True):
            with st.spinner("Computing LIME explanation (~5s)..."):
                lime_local_explanation(model, X, int(idx), get_label_map(), num_features=12)

    st.markdown("---")

    # ── Section 3: PDP ───────────────────────────────────────────────────────
    st.subheader("3 — Partial Dependence Plot")

    with st.expander("What a PDP shows"):
        st.markdown(
            "A PDP answers: *holding all other features fixed, how does changing this one "
            "feature from its minimum to maximum value affect the predicted credit tier?* "
            "Useful for communicating model behaviour to non-technical audiences. "
            "Try `CC_utilization` — the PDP will show that predicted creditworthiness "
            "deteriorates sharply as utilization exceeds ~60%."
        )

    # Pre-select the most important numeric features for convenience
    numeric_features = [c for c in X.columns if "_" not in c or c.startswith(("Tot", "pct", "num", "CC", "PL", "Age"))]
    all_features = X.columns.tolist()

    feature_for_pdp = st.selectbox(
        "Feature",
        all_features,
        index=all_features.index("CC_utilization") if "CC_utilization" in all_features else 0,
        help="CC_utilization and num_times_delinquent typically show the clearest relationships"
    )

    if st.button("Show PDP", use_container_width=True):
        with st.spinner("Computing partial dependence..."):
            plot_pdp(model, X, feature_for_pdp)

    st.markdown("---")
    st.info("Proceed to Score New Applicants or Fairness Audit.")


main()
