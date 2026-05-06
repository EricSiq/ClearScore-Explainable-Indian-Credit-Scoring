"""
app/components/shap_plotter.py
Shared SHAP computation and rendering used by Pages 4, 5, and 7.

Key design decisions:
  - SHAP values are computed once per (model, dataset) pair and cached in
    session_state under "shap_values_{cache_key}". This avoids recomputing
    on every button press — critical for 51K-row datasets where SHAP takes
    30–60 seconds.
  - All functions accept a named pd.DataFrame for X so that SHAP reads real
    column names from X.columns and displays them in all plots.
  - Multi-class SHAP values have shape (n_samples, n_features, n_classes).
    Rendering functions handle both 2D and 3D shapes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

__all__ = [
    "get_shap_values",
    "shap_global_plot",
    "shap_local_waterfall",
    "shap_compare_rows",
    "shap_top_features",
]


def get_shap_values(model, X: pd.DataFrame, cache_key: str):
    """
    Compute SHAP values for the full dataset and cache in session_state.

    Parameters
    ----------
    model      : fitted sklearn / EBM model
    X          : named pd.DataFrame — column names flow into SHAP plots
    cache_key  : unique string key for session_state caching
                 (e.g. "shap_train_ebm" or "shap_unseen_lr")

    Returns
    -------
    shap.Explanation object (cached after first call)
    """
    full_key = f"shap_values_{cache_key}"
    if full_key in st.session_state:
        return st.session_state[full_key]

    with st.spinner(f"Computing SHAP values [{cache_key}] — cached after first run..."):
        explainer   = shap.Explainer(model, X)
        shap_values = explainer(X)
        st.session_state[full_key] = shap_values

    return shap_values


def shap_global_plot(model, X: pd.DataFrame, cache_key: str, max_display: int = 20):
    """
    Render a global SHAP summary plot (beeswarm) in Streamlit.
    Feature names come from X.columns — no generic indices.
    """
    try:
        shap_values = get_shap_values(model, X, cache_key)
        fig, _ = plt.subplots(figsize=(9, 6))
        shap.summary_plot(shap_values, X, show=False, max_display=max_display)
        st.pyplot(plt.gcf())
        plt.close("all")
    except Exception as e:
        st.error(f"Global SHAP failed: {e}")


def shap_local_waterfall(model, X: pd.DataFrame, cache_key: str, idx: int):
    """
    Render a SHAP waterfall plot for a single applicant row.
    """
    try:
        shap_values = get_shap_values(model, X, cache_key)
        single = shap_values[idx]
        fig, _ = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(single, show=False)
        st.pyplot(plt.gcf())
        plt.close("all")
    except Exception as e:
        st.error(f"Local SHAP failed: {e}")


def shap_compare_rows(model, X: pd.DataFrame, cache_key: str, row_list: list):
    """
    Render a SHAP summary plot comparing multiple applicant rows.
    """
    if len(row_list) < 2:
        st.warning("Select at least two applicants to compare.")
        return
    try:
        shap_values = get_shap_values(model, X, cache_key)
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values[row_list], X.iloc[row_list], show=False)
        st.pyplot(plt.gcf())
        plt.close("all")
    except Exception as e:
        st.error(f"SHAP comparison failed: {e}")


def shap_top_features(shap_values, idx: int, n: int = 10) -> list[tuple[str, float]]:
    """
    Return the top-n (feature_name, shap_value) tuples for a single instance.

    For multi-class SHAP (shape n_features × n_classes), uses mean |SHAP|
    across classes for ranking and mean SHAP for the display value.

    Used by Page 5 (PDF report) and Page 7 (Agent explanations).
    """
    sv = shap_values[idx]
    vals = sv.values

    if vals.ndim == 2:
        importance   = np.abs(vals).mean(axis=1)
        display_vals = vals.mean(axis=1)
    else:
        importance   = np.abs(vals)
        display_vals = vals

    top_idx = np.argsort(importance)[::-1][:n]
    return [(sv.feature_names[i], float(display_vals[i])) for i in top_idx]
