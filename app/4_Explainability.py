import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import shap
from lime.lime_tabular import LimeTabularExplainer

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.exceptions import NotFittedError

# Helper to load models (prefer session_state, fallback to disk)
def load_model(name, session_key):
    if session_key in st.session_state:
        return st.session_state[session_key]
    model_path = os.path.join("models", name)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.session_state[session_key] = model
        return model
    return None

def ensure_data():
    if "processed_df" not in st.session_state:
        st.error("Please run preprocessing first (Upload -> Preprocess).")
        return None
    return st.session_state["processed_df"]

def try_get_Xy(df, target_col="Approved_Flag"):
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in processed data.")
        return None, None
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def show_shap_global(model, X, max_display=20):
    st.write("### Global SHAP Summary")
    try:
        expl = shap.Explainer(model, X)
        shap_values = expl(X)
        plt.figure(figsize=(8,6))
        # shap.summary_plot returns a matplotlib figure when show=False (older/newer versions differ).
        shap.summary_plot(shap_values, X, show=False, max_display=max_display)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"Failed to compute global SHAP: {e}")

def show_shap_local(model, X, instance_idx):
    st.write("### Local SHAP Explanation (Waterfall / Force Plot)")
    try:
        expl = shap.Explainer(model, X)
        shap_values = expl(X)
        # select instance
        instance = X.iloc[[instance_idx]]
        single_shap = shap_values[instance_idx]
        # Waterfall (matplotlib)
        try:
            plt.figure(figsize=(8,4))
            shap.plots.waterfall(single_shap, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception:
            # fallback to force_plot (may need JS; shap.save_html could be used but here provide textual fallback)
            try:
                force_html = shap.plots.force(single_shap, matplotlib=False)
                st.write("Interactive force plot may not render in this environment.")
            except Exception as ee:
                st.error(f"Could not render local SHAP waterfall/force: {ee}")
    except Exception as e:
        st.error(f"Failed to compute local SHAP: {e}")

def show_lime_local(model, X, y, instance_idx, mode='classification', num_features=10):
    st.write("### LIME Local Explanation")
    try:
        # Prepare numpy arrays for LIME
        X_np = X.values
        feature_names = X.columns.tolist()

        # Determine class names if classification
        class_names = None
        if mode == "classification":
            unique_y = np.unique(y)
            class_names = [str(c) for c in unique_y]

        explainer = LimeTabularExplainer(
            X_np,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            discretize_continuous=True
        )

        instance = X_np[instance_idx]
        if mode == "classification":
            exp = explainer.explain_instance(instance, model.predict_proba, num_features=num_features)
        else:
            exp = explainer.explain_instance(instance, model.predict, num_features=num_features)

        # Show LIME explanation as list
        st.write("LIME explanation (feature contributions):")
        html = exp.as_html()
        st.components.v1.html(html, height=400, scrolling=True)

    except Exception as e:
        st.error(f"Failed to compute LIME explanation: {e}")

def show_pdp(model, X, features):
    st.write("### Partial Dependence Plot (PDP)")
    if not features:
        st.info("Pick a feature for PDP.")
        return
    try:
        fig, ax = plt.subplots(figsize=(8,4))
        # sklearn's PartialDependenceDisplay convenience
        PartialDependenceDisplay.from_estimator(model, X, features=features, ax=ax)
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.error(f"Failed to compute PDP: {e}")

def main():
    st.title("🔍 Explainability: SHAP, LIME, PDP")

    df = ensure_data()
    if df is None:
        return

    X, y = try_get_Xy(df)
    if X is None:
        return

    # Load models (prefer session state)
    st.write("Loading models (session first, then disk)...")
    lr = load_model("logistic_regression.pkl", "lr_model")
    ebm = load_model("ebm_model.pkl", "ebm_model")

    model_options = {}
    if lr:
        model_options["LogisticRegression"] = lr
    if ebm:
        model_options["EBM"] = ebm

    if not model_options:
        st.error("No models found. Train models first (03_Model_Training).")
        return

    model_name = st.selectbox("Choose model for explanations", list(model_options.keys()))
    model = model_options[model_name]

    # try to ensure model is fitted
    try:
        # some estimators have `predict` only after fitting; this will raise if not fitted
        model_predict = getattr(model, "predict", None)
        if model_predict is None:
            raise NotFittedError("Model has no predict method.")
    except Exception as e:
        st.error(f"Model appears not fitted or invalid: {e}")
        return

    st.markdown("---")
    # Global SHAP
    if st.button("Compute Global SHAP Summary"):
        with st.spinner("Computing global SHAP..."):
            show_shap_global(model, X)

    st.markdown("---")
    # Local explanations — choose instance
    instance_idx = st.number_input("Select row index for local explanations", min_value=0, max_value=max(0, len(X)-1), value=0, step=1)

    if st.button("Show Local SHAP Explanation"):
        with st.spinner("Computing local SHAP..."):
            show_shap_local(model, X, int(instance_idx))

    if st.button("Show LIME Explanation"):
        with st.spinner("Computing LIME explanation..."):
            # LIME requires model.predict_proba for classification
            mode = "classification"
            # detect if model has predict_proba; if not, use predict (regression or surrogate)
            if hasattr(model, "predict_proba"):
                mode = "classification"
            else:
                mode = "regression"
            show_lime_local(model, X, y, int(instance_idx), mode=mode)

    st.markdown("---")
    # PDP
    st.write("Partial Dependence Plots")
    feature_for_pdp = st.selectbox("Choose feature for PDP", X.columns.tolist())
    if st.button("Show PDP"):
        with st.spinner("Computing PDP..."):
            show_pdp(model, X, [feature_for_pdp])

if __name__ == "__main__":
    main()
