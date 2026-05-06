import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# ── Shared components ─────────────────────────────────────────────────────────
from app.components.model_loader import load_model as _load_model_from_disk
from app.components.utils        import get_label_map, get_X_y
from app.components.shap_plotter import get_shap_values, shap_global_plot, shap_local_waterfall, shap_compare_rows
from app.components.lime_plotter import lime_local_explanation
from app.components.pdp_plotter  import plot_pdp

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_COL = "Approved_Flag"
MODEL_DIR  = "app/models"
_LABEL_MAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_model(filename: str, session_key: str):
    """Thin wrapper — delegates to shared model_loader component."""
    return _load_model_from_disk(filename, session_key)


# ── SHAP helpers — delegated to shap_plotter component ───────────────────────

def _get_shap_values(model, X: pd.DataFrame, model_name: str):
    """Cache key uses model_name to separate train-set SHAP from unseen-set SHAP."""
    return get_shap_values(model, X, f"train_{model_name}")


def _show_shap_global(model, X: pd.DataFrame, model_name: str, max_display: int = 20):
    st.write("### 🌍 Global SHAP Summary")
    shap_global_plot(model, X, f"train_{model_name}", max_display)


def _show_shap_local(model, X: pd.DataFrame, model_name: str, idx: int):
    st.write(f"### 🔍 SHAP Waterfall — Applicant {idx}")
    shap_local_waterfall(model, X, f"train_{model_name}", idx)


# ── LIME helper — delegated to lime_plotter component ────────────────────────

def _show_lime(model, X: pd.DataFrame, y: pd.Series, idx: int, num_features: int = 12):
    st.write(f"### 🧩 LIME Local Explanation — Applicant {idx}")
    lime_local_explanation(model, X, idx, get_label_map(), num_features)


# ── PDP helper — delegated to pdp_plotter component ──────────────────────────

def _show_pdp(model, X: pd.DataFrame, feature: str):
    st.write(f"### 📈 Partial Dependence Plot — `{feature}`")
    plot_pdp(model, X, feature)


# ── Main page ─────────────────────────────────────────────────────────────────

def main():
    st.title("🔍 Explainability — SHAP · LIME · PDP")

    # ── Data ─────────────────────────────────────────────────────────────────
    X, y = get_X_y()
    if X is None:
        return

    # ── Models ───────────────────────────────────────────────────────────────
    lr  = _load_model("logistic_regression.pkl", "lr_model")
    ebm = _load_model("ebm_model.pkl",           "ebm_model")

    model_options = {}
    if lr:
        model_options["Logistic Regression"] = lr
    if ebm:
        model_options["EBM"] = ebm

    if not model_options:
        st.error("No trained models found. Run **Train Models** first.")
        return

    model_name = st.selectbox("Model", list(model_options.keys()))
    model      = model_options[model_name]

    # Confirm EBM feature names are real (diagnostic, shown once)
    if model_name == "EBM":
        ebm_features = getattr(model, "feature_names_in_", None)
        if ebm_features is not None:
            st.caption(
                f"EBM feature names: {list(ebm_features[:5])} … "
                f"({len(ebm_features)} total)"
            )
        else:
            st.warning(
                "EBM `feature_names_in_` is not set — charts may show generic indices. "
                "Re-train the model on the **Train Models** page."
            )

    st.markdown("---")

    # ── Section 1: Global SHAP ───────────────────────────────────────────────
    st.subheader("1 — Global Feature Importance (SHAP)")
    max_display = st.slider("Features to display", min_value=5, max_value=30, value=20)
    if st.button("Compute Global SHAP Summary"):
        _show_shap_global(model, X, model_name, max_display)

    st.markdown("---")

    # ── Section 2: Local explanations ────────────────────────────────────────
    st.subheader("2 — Local Explanations (per applicant)")

    idx = st.number_input(
        "Applicant row index",
        min_value=0,
        max_value=max(0, len(X) - 1),
        value=0,
        step=1,
    )

    col_shap, col_lime = st.columns(2)
    with col_shap:
        if st.button("Show SHAP Waterfall"):
            _show_shap_local(model, X, model_name, int(idx))
    with col_lime:
        if st.button("Show LIME Explanation"):
            _show_lime(model, X, y, int(idx))

    st.markdown("---")

    # ── Section 3: PDP ───────────────────────────────────────────────────────
    st.subheader("3 — Partial Dependence Plot")

    # Group features for easier navigation: numeric first, then OHE-expanded
    all_features = X.columns.tolist()
    feature_for_pdp = st.selectbox("Feature", all_features)

    if st.button("Show PDP"):
        _show_pdp(model, X, feature_for_pdp)

    st.markdown("---")
    st.info("➡ Proceed to **Score New Applicants** or **Fairness Audit**.")


if __name__ == "__main__":
    main()
