"""
app/components/model_loader.py
Single source of truth for loading trained models from disk or session state.

Model directory resolution:
  - Streamlit Cloud: repo filesystem is read-only, so models are saved to /tmp.
  - Local dev: models saved to app/models/ (writable).
  The training page (3_Model_Training.py) uses the same logic to determine
  where to write, so this loader will always look in the same place.
"""

import os
import tempfile
import joblib
import streamlit as st

# Mirror the same directory logic used when saving models in Page 3
_TMP_MODELS = os.path.join(tempfile.gettempdir(), "creditlens_models")
MODEL_DIR   = _TMP_MODELS if not os.access("app/models", os.W_OK) else "app/models"


def load_model(model_filename: str, session_key: str):
    """
    Return a trained model from session_state (fast path) or disk (cold path).
    Checks both the resolved MODEL_DIR and the fallback /tmp directory.
    """
    if session_key in st.session_state:
        return st.session_state[session_key]

    # Try resolved MODEL_DIR first, then the other location as fallback
    candidates = [
        os.path.join(MODEL_DIR, model_filename),
        os.path.join(_TMP_MODELS, model_filename),
        os.path.join("app/models", model_filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.session_state[session_key] = model
                return model
            except Exception as e:
                st.error(f"Error loading model `{model_filename}` from `{path}`: {e}")
                return None

    st.warning(
        f"Model `{model_filename}` not found. "
        "Run **3 · Train Models** first — models are stored in memory for the current session."
    )
    return None
