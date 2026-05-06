"""
app/components/model_loader.py
Single source of truth for loading trained models from disk or session state.
"""

import os
import joblib
import streamlit as st

# Default model directory — relative to the project root where streamlit is run
MODEL_DIR = "app/models"


def load_model(model_filename: str, session_key: str):
    """
    Return a trained model, loading from session_state if already present,
    otherwise from disk at MODEL_DIR/<model_filename>.

    Stores the loaded model back into session_state under session_key so
    subsequent calls within the same session are instant.

    Returns None (with a Streamlit warning) if the file is not found.
    """
    if session_key in st.session_state:
        return st.session_state[session_key]

    path = os.path.join(MODEL_DIR, model_filename)
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            st.session_state[session_key] = model
            return model
        except Exception as e:
            st.error(f"Error loading model `{model_filename}`: {e}")
            return None

    st.warning(f"Model `{model_filename}` not found in `{MODEL_DIR}`. Run **Train Models** first.")
    return None
