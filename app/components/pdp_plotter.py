"""
app/components/pdp_plotter.py
Shared Partial Dependence Plot rendering used by Pages 4 and 5.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

__all__ = ["plot_pdp"]


def plot_pdp(model, X: pd.DataFrame, feature: str):
    """
    Render a Partial Dependence Plot for a single feature in Streamlit.

    Parameters
    ----------
    model   : fitted sklearn / EBM model
    X       : named pd.DataFrame — used as background data for PDP computation
    feature : column name to plot (must exist in X.columns)
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        PartialDependenceDisplay.from_estimator(model, X, [feature], ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"PDP failed for `{feature}`: {e}")
