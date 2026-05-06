"""
app/components/utils.py
Shared session-state helpers used across all pages.

Provides:
  - load_model()       — model loading with session cache
  - get_label_map()    — {0:'P1', 1:'P2', 2:'P3', 3:'P4'} from session or default
  - get_X_y()          — split processed_df into (X, y) named DataFrames
  - get_processed_df() — return processed_df or show error
  - get_unseen_df()    — return unseen_df or show error
"""

import streamlit as st
import pandas as pd

from app.components.model_loader import load_model  # re-export for convenience

TARGET_COL = "Approved_Flag"
_LABEL_MAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}

__all__ = [
    "load_model",
    "get_label_map",
    "get_X_y",
    "get_processed_df",
    "get_unseen_df",
]


def get_label_map() -> dict:
    """
    Return the integer→class-label mapping from session state.
    Falls back to the default {0:'P1', 1:'P2', 2:'P3', 3:'P4'} if not set.
    """
    return st.session_state.get("label_map", _LABEL_MAP)


def get_processed_df() -> pd.DataFrame | None:
    """
    Return processed_df from session state, or display an error and return None.
    processed_df is written by Page 2 (Preprocessing) and contains the full
    OHE-encoded feature matrix plus the integer-encoded target column.
    """
    if "processed_df" not in st.session_state:
        st.error("⚠ Data not preprocessed yet. Run **Preprocessing** first.")
        return None
    return st.session_state["processed_df"]


def get_unseen_df() -> pd.DataFrame | None:
    """
    Return unseen_df from session state, or display an error and return None.
    unseen_df is the raw (pre-transform) unseen dataset uploaded on Page 1 or 5.
    """
    if "unseen_df" not in st.session_state:
        st.error("⚠ Unseen dataset not found. Upload it on the **Upload Data** page.")
        return None
    return st.session_state["unseen_df"]


def get_X_y(target_col: str = TARGET_COL) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """
    Split processed_df into (X, y) using the authoritative feature_names list
    from session state.

    X is a named pd.DataFrame — SHAP, LIME, and EBM all read column names from it.
    y is the integer-encoded target series (0=P1, 1=P2, 2=P3, 3=P4).

    Returns (None, None) if processed_df is missing or target column not found.
    """
    df = get_processed_df()
    if df is None:
        return None, None

    if target_col not in df.columns:
        st.error(f"Target column `{target_col}` not found. Re-run **Preprocessing**.")
        return None, None

    feature_names = st.session_state.get(
        "feature_names",
        [c for c in df.columns if c != target_col],
    )
    X = df[feature_names]
    y = df[target_col]
    return X, y
