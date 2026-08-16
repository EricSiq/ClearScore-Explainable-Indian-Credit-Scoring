"""CreditLens multi-page entry point."""

import streamlit as st

st.set_page_config(page_title="XClearScore | Explainable Credit Scoring", page_icon="X", layout="wide")

with st.sidebar:
    st.markdown("### XClearScore")
    st.caption("Explainable AI Indian Credit Scoring")
    with st.expander("XAI and review principles"):
        st.markdown(
            "**Primary model** — Explainable Boosting Machine (InterpretML), supported by "
            "a Logistic Regression baseline.\n\n"
            "**XAI views** — EBM shape functions, SHAP feature attribution, and LIME local approximation.\n\n"
            "**Fairness monitoring** — group metrics are diagnostics. They do not prove a model "
            "is fair or make it suitable for automated decisions.\n\n"
            "**Human review** — all outputs are for analyst review in this prototype."
        )

navigation = st.navigation(
    {
        "": [st.Page("app/Home.py", title="Overview", default=True)],
        "Workflow": [
            st.Page("app/1_Data_Upload.py", title="Data intake"),
            st.Page("app/2_Preprocessing.py", title="Prepare data"),
            st.Page("app/3_Model_Training.py", title="Train and validate"),
            st.Page("app/4_Explainability.py", title="Model explanations"),
            st.Page("app/5_Score_New_Data.py", title="Score new data"),
        ],
        "Governance": [
            st.Page("app/6_Fairness_Audit.py", title="Fairness monitoring"),
            st.Page("app/7_Credit_Analyst_Agent.py", title="Decision review"),
        ],
    }
)
navigation.run()
