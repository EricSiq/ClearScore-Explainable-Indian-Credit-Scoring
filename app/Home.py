"""Overview and intentionally lazy demo-data launcher."""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("Datasets")


@st.cache_data(show_spinner=False)
def _read_demo_data():
    return (
        pd.read_excel(DATA_DIR / "Internal_Bank_Dataset.xlsx"),
        pd.read_excel(DATA_DIR / "External_Cibil_Dataset.xlsx"),
        pd.read_excel(DATA_DIR / "Unseen_Dataset.xlsx"),
    )


def main():
    st.title("XClearScore")
    st.caption("Explainable AI Indian Credit Scoring System")
    st.markdown("An interpretable credit-risk pipeline for CIBIL and bank data, combining an Explainable Boosting Machine with SHAP and LIME evidence.")
    left, middle, right = st.columns(3)
    left.metric("Primary model", "InterpretML EBM")
    middle.metric("XAI stack", "EBM + SHAP + LIME")
    right.metric("Published benchmark", "95.6% | AUC 0.982")
    st.subheader("Workflow")
    st.markdown("**Data intake → Prepare data → Train EBM → Explain with EBM, SHAP & LIME → Fairness monitoring → Score new data**")
    st.caption("The hosted demo uses bounded training and lazy explanations. The 95.6% / 0.982 benchmark comes from the full 51,336-row run with a 10,268-applicant holdout, reproduced locally.")
    if "internal_df" in st.session_state:
        st.success("Demo data is available in this session. Continue with Prepare data.")
    elif st.button("Load bundled demo data", type="primary"):
        try:
            internal, external, unseen = _read_demo_data()
            st.session_state.update({"internal_df": internal, "external_df": external, "unseen_df": unseen, "demo_mode": True})
            st.success("Demo data loaded. Continue with Prepare data.")
        except FileNotFoundError:
            st.error("Bundled demo files were not found. Use Data intake to upload files.")
    st.subheader("Prototype boundary")
    st.warning("This project uses anonymised/synthetic-style case-study data and session-only state. It is not a lending system, a production risk model, or a source of customer-facing adverse-action reasons. Every output requires human review and governed policy outside this app.")
    with st.expander("What the explanation means"):
        st.write("For the selected class, each displayed contribution equals the transformed feature value multiplied by that class's fitted coefficient. It explains the model score exactly. It does not establish causation, affordability, or a fair lending outcome.")


main()
