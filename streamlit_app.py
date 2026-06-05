"""
CreditLens — Explainable AI Credit Scoring
Entry point for Streamlit multi-page app.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="CreditLens — Credit Scoring",
    page_icon="💳",
    layout="wide",
)

# ── Sidebar concept panel — visible on every page ────────────────────────────
with st.sidebar:
    with st.expander("Key concepts"):
        st.markdown(
            "**EBM** — Glass-box GA2M model. Shape functions = the model. "
            "Explanations are exact, not SHAP approximations.\n\n"
            "**SHAP** — Shapley value attribution. For EBM: SHAP = shape function output. "
            "Global = mean |SHAP|. Local = one prediction decomposed.\n\n"
            "**DPD** — Demographic Parity Difference. "
            "max − min approval rate across groups. RBI limit: < 0.05.\n\n"
            "**EOD** — Equalized Odds Difference. "
            "max(|ΔTPR|, |ΔFPR|). RBI limit: < 0.05.\n\n"
            "**NPA** — Non-Performing Asset. Loan in default (> 90 days). "
            "Core risk metric for all Indian banks.\n\n"
            "**Tiers** — P1 Excellent · P2 Good · P3 Marginal · P4 Reject"
        )

pg = st.navigation(
    {
        "": [
            st.Page("app/Home.py", title="Home", default=True),
        ],
        "Pipeline": [
            st.Page("app/1_Data_Upload.py",    title="1 · Upload Data"),
            st.Page("app/2_Preprocessing.py",  title="2 · Preprocess"),
            st.Page("app/3_Model_Training.py", title="3 · Train Models"),
        ],
        "Explainability": [
            st.Page("app/4_Explainability.py", title="4 · Explainability"),
            st.Page("app/5_Score_New_Data.py", title="5 · Score New Data"),
        ],
        "Governance": [
            st.Page("app/6_Fairness_Audit.py",       title="6 · Fairness Audit"),
            st.Page("app/7_Credit_Analyst_Agent.py", title="7 · Credit Analyst Agent"),
            st.Page("app/8_Business_Summary.py",     title="8 · Business Summary"),
        ],
    }
)

pg.run()
