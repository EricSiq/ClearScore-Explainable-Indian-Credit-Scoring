"""
app/components/lime_plotter.py
Shared LIME local explanation rendering used by Pages 4 and 5.

Design notes:
  - class_names must be P1/P2/P3/P4 (not "Reject"/"Approve") to match the
    4-class target. The caller passes label_map for this.
  - feature_names are read from X.columns — no generic indices.
  - num_features defaults to 12 to show enough context for credit decisions.
"""

import streamlit as st
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

__all__ = ["lime_local_explanation"]


def lime_local_explanation(
    model,
    X: pd.DataFrame,
    idx: int,
    label_map: dict,
    num_features: int = 12,
):
    """
    Render a LIME local explanation for a single applicant row.

    Parameters
    ----------
    model      : fitted sklearn / EBM model with predict_proba
    X          : named pd.DataFrame — feature names read from X.columns
    idx        : row index to explain
    label_map  : {0:'P1', 1:'P2', 2:'P3', 3:'P4'} for class name display
    num_features: number of features to show in the explanation
    """
    try:
        mode        = "classification" if hasattr(model, "predict_proba") else "regression"
        class_names = [label_map[i] for i in sorted(label_map)] if mode == "classification" else None

        explainer = LimeTabularExplainer(
            X.values,
            feature_names=X.columns.tolist(),
            class_names=class_names,
            mode=mode,
            discretize_continuous=True,
        )

        predict_fn = model.predict_proba if mode == "classification" else model.predict
        exp = explainer.explain_instance(X.values[idx], predict_fn, num_features=num_features)

        st.components.v1.html(exp.as_html(), height=440, scrolling=True)

    except Exception as e:
        st.error(f"LIME explanation failed: {e}")
