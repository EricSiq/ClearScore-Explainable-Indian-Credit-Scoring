"""Batch scoring with the trained EBM primary model or LR baseline."""

import pandas as pd
import streamlit as st

from app.components.model_loader import load_model
from app.components.utils import get_label_map

TARGET_COL = "Approved_Flag"
TIER_DESCRIPTIONS = {
    "P1": "Excellent credit profile", "P2": "Good credit profile",
    "P3": "Marginal profile", "P4": "Poor credit profile",
}


def _transform_unseen(unseen_df):
    preprocessor = st.session_state.get("preprocessor")
    feature_names = st.session_state.get("feature_names")
    if preprocessor is None or feature_names is None:
        raise ValueError("Run Prepare data before scoring new applicants.")
    raw = unseen_df.drop(columns=[TARGET_COL], errors="ignore").copy()
    for _, _, columns in preprocessor.transformers_:
        for column in columns:
            if column not in raw:
                raw[column] = pd.NA
    transformed = preprocessor.transform(raw)
    return pd.DataFrame(transformed, columns=feature_names)


def main():
    st.title("Score new data")
    st.caption("Batch model output for analyst review. Downloaded results are not lending decisions.")
    uploaded = st.file_uploader("Applicant data (.xlsx or .csv)", type=["xlsx", "csv"])
    unseen = st.session_state.get("unseen_df")
    if uploaded:
        unseen = pd.read_excel(uploaded) if uploaded.name.lower().endswith("xlsx") else pd.read_csv(uploaded)
        st.session_state["unseen_df"] = unseen
    if unseen is None:
        st.info("Upload a file or load the bundled demo from Overview.")
        return
    lr = load_model("logistic_regression.pkl", "lr_model")
    ebm = load_model("ebm_model.pkl", "ebm_model")
    models = {"EBM (primary)": ebm, "Logistic Regression (baseline)": lr}
    models = {name: value for name, value in models.items() if value is not None}
    if not models:
        st.error("Run Train and validate before scoring.")
        return
    selected = st.selectbox("Scoring model", list(models))
    model = models[selected]
    if not st.button("Score applicants", type="primary"):
        return
    try:
        X = _transform_unseen(unseen)
        predicted = model.predict(X)
        probabilities = model.predict_proba(X)
    except Exception as error:
        st.error(f"Scoring could not run: {error}")
        return
    labels = get_label_map()
    output = pd.DataFrame({"predicted_tier": [labels[int(item)] for item in predicted]})
    output["tier_description"] = output["predicted_tier"].map(TIER_DESCRIPTIONS)
    output["model_confidence"] = probabilities.max(axis=1).round(4)
    for position, class_id in enumerate(model.classes_):
        output[f"probability_{labels[int(class_id)]}"] = probabilities[:, position].round(4)
    st.session_state["score_X"] = X
    st.session_state["score_output"] = output
    st.dataframe(output, use_container_width=True)
    st.download_button("Download scored CSV", output.to_csv(index=False), "creditlens_scored.csv", "text/csv")

    st.subheader("Review one result")
    idx = int(st.number_input("Applicant row", 0, len(output) - 1, 0, 1))
    class_id = int(predicted[idx])
    st.write(f"Selected tier: **{labels[class_id]}**. Review the EBM shape functions, SHAP attribution, and LIME approximation on the Explainability page for the trained model.")


main()
