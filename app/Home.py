import streamlit as st

st.title("CreditLens — Explainable AI Credit Scoring")
st.caption("Interpretable loan approval for the Indian credit market · 51,336 CIBIL records")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Overview")
    st.markdown(
    "An end-to-end ML dashboard that ingests internal bank trade-line data "
    "and external CIBIL bureau data, trains interpretable models, and explains "
    "every credit decision — globally and per applicant."
    )

with col2:
    st.markdown("### Primary model")
    st.markdown(
    "**Explainable Boosting Machine (EBM)** — a glass-box model whose shape "
    "functions are the explanation, not a post-hoc approximation. "
    "Achieves 95.9% accuracy on the 4-class credit tier task (P1–P4)."
    )

with col3:
    st.markdown("### Fairness")
    st.markdown(
    "Demographic parity and equalized odds computed across GENDER, EDUCATION, "
    "and MARITAL STATUS. Thresholds aligned with RBI Model Risk Management "
    "Guidelines (2023)."
    )

st.markdown("---")
st.markdown("### Pipeline")

steps = [
    ("1 · Upload Data", "Upload Internal Bank and External CIBIL datasets"),
    ("2 · Preprocess", "Merge on PROSPECTID · OHE encode · Scale · Label encode target"),
    ("3 · Train Models", "Logistic Regression + EBM · Evaluation metrics · ROC curves"),
    ("4 · Explainability", "Global SHAP · Local SHAP waterfall · LIME · PDP"),
    ("5 · Score New Data", "Transform unseen applicants · Predict P1–P4 · PDF reports"),
    ("6 · Fairness Audit", "DPD · EOD · Selection rate heatmap · RBI threshold indicators"),
    ("7 · Credit Analyst Agent", "Chat interface · Intent detection · llama.cpp SLM explanations"),
    ("8 · Business Summary", "NPA exposure avoided · Tier breakdown · Downloadable PDF"),
]

for name, desc in steps:
    st.markdown(f"**{name}** — {desc}")

st.markdown("---")
st.info("Start on **1 · Upload Data** in the sidebar.")
