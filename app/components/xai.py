"""Bounded, lazy SHAP and LIME helpers for the hosted XClearScore demo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def shap_values(model, X: pd.DataFrame, cache_key: str, max_rows: int = 50):
    """Compute SHAP only after an explicit request and cache a small representative sample."""
    key = f"shap_{cache_key}"
    if key in st.session_state:
        return st.session_state[key]
    import shap

    sample = X.sample(min(max_rows, len(X)), random_state=42)
    background = sample.sample(min(20, len(sample)), random_state=42)
    with st.spinner("Computing bounded SHAP values (cached for this session)..."):
        explainer = shap.Explainer(model.predict_proba, background, algorithm="permutation")
        values = explainer(sample, max_evals=2 * X.shape[1] + 1)
    st.session_state[key] = (sample, values)
    return sample, values


def lime_explanation(model, X: pd.DataFrame, row_index: int, class_names: list[str]):
    """Create one local LIME explanation from a capped background sample."""
    from lime.lime_tabular import LimeTabularExplainer

    training = X.sample(min(1_000, len(X)), random_state=42)
    explainer = LimeTabularExplainer(
        training.values,
        feature_names=training.columns.tolist(),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    return explainer.explain_instance(X.iloc[row_index].values, model.predict_proba, num_features=10)
