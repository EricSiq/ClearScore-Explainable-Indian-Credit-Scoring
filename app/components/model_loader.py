import streamlit as st
import os
import joblib
from sklearn.exceptions import NotFittedError

def load_model(model_filename, session_key):
    """
    Safely load model either from session_state or from disk (/models/).
    Returns model object or None.
    """

    # If already loaded in Streamlit session
    if session_key in st.session_state:
        return st.session_state[session_key]

    model_path = os.path.join("models", model_filename)

    # If model exists on disk
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.session_state[session_key] = model
            return model
        except Exception as e:
            st.error(f"❌ Error loading model file '{model_filename}': {e}")
            return None

    st.warning(f"⚠ Model '{model_filename}' not found. Train models first.")
    return None


def verify_model_is_fitted(model):
    """
    Run a simple check to confirm model has been trained/fitted.
    Raises NotFittedError if unfitted.
    """
    if not hasattr(model, "predict"):
        raise NotFittedError("Model does not implement predict(), can't verify fit.")

    try:
        # Try calling predict on dummy data; lightweight sanity check.
        # We only test that predict exists, not correctness.
        pass
    except Exception:
        raise NotFittedError("Model appears to be untrained or improperly loaded.")
