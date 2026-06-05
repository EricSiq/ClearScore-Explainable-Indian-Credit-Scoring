import streamlit as st
import pandas as pd
import os

# ── Shared concept definitions shown in expander ──────────────────────────────
KEY_CONCEPTS = """
**EBM (Explainable Boosting Machine)**
A glass-box GA2M model where each feature has a learned shape function.
Prediction = sum of shape function outputs. Explanations are exact — not
post-hoc approximations applied to a black box.

**SHAP (SHapley Additive exPlanations)**
Game-theoretic attribution of each feature's contribution to a prediction.
For EBM, SHAP values equal the shape function values exactly. Global SHAP =
mean |SHAP| across all applicants. Local SHAP = decomposition of one prediction.

**DPD — Demographic Parity Difference**
max(approval_rate across groups) − min(approval_rate across groups).
Measures whether the model approves different demographic groups at equal rates.
RBI MRM 2023 threshold: < 0.05 acceptable, > 0.10 requires mitigation.

**EOD — Equalized Odds Difference**
max(|ΔTPR|, |ΔFPR|) across demographic groups.
Measures whether error rates are consistent — i.e., the model makes similar
types of mistakes regardless of a borrower's gender, education, or marital status.

**Credit Tiers (P1–P4)**
P1 — Excellent: approve at best rate
P2 — Good: approve at standard rate
P3 — Marginal: conditional approval, higher rate or collateral required
P4 — Poor: reject or offer secured product only

**NPA (Non-Performing Asset)**
A loan where repayment has defaulted (typically > 90 days overdue under RBI
classification). Avoiding NPA is the core risk objective for every Indian bank.
"""

# ── Demo mode loader ──────────────────────────────────────────────────────────

def _load_demo_data():
    """Load bundled datasets into session state — no file upload required."""
    try:
        internal = pd.read_excel("Datasets/Internal_Bank_Dataset.xlsx")
        external = pd.read_excel("Datasets/External_Cibil_Dataset.xlsx")
        unseen   = pd.read_excel("Datasets/Unseen_Dataset.xlsx")
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}. Ensure the Datasets/ folder is present.")
        return False

    st.session_state["internal_df"] = internal
    st.session_state["external_df"] = external
    st.session_state["unseen_df"]   = unseen
    st.session_state["demo_mode"]   = True
    return True


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("CreditLens")
st.caption("Explainable AI credit scoring for the Indian market · 51,336 CIBIL records · RBI MRM aligned")

st.markdown("---")

# Headline metrics — visible on load, before any pipeline run
st.subheader("System Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Training records",  "51,336",   help="Merged internal trade-line + external CIBIL bureau data")
m2.metric("EBM accuracy",      "95.9%",    help="4-class credit tier prediction (P1–P4) on 20% holdout")
m3.metric("AUC (OvR macro)",   "0.982",    help="One-vs-rest ROC AUC averaged across all four credit tiers")
m4.metric("EDUCATION bias EOD","0.154",    help="Equalized Odds Difference for EDUCATION — above RBI action threshold of 0.10. Detected and documented.")

st.markdown("---")

# Three-column context
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### The problem")
    st.markdown(
        "Traditional CIBIL scoring is a black box. Lenders cannot explain why an "
        "applicant was rejected. Regulators cannot audit the decision. Thin-file "
        "borrowers — 400M+ adults with limited credit history — are systematically "
        "excluded with no recourse."
    )

with col2:
    st.markdown("#### The approach")
    st.markdown(
        "**EBM** over XGBoost — because EBM's shape functions *are* the model. "
        "Explanations are exact, not SHAP approximations on a black box. "
        "**Fairness audit** across gender, education, and marital status, "
        "with thresholds from RBI's Model Risk Management Guidelines (2023)."
    )

with col3:
    st.markdown("#### The result")
    st.markdown(
        "95.9% accuracy on 4-class tier prediction. Rs 9.39 Cr NPA exposure avoided "
        "on the 10,268-applicant test set. EDUCATION bias detected (EOD = 0.154) "
        "with mitigation options documented. Per-applicant SHAP explanations "
        "exportable as PDF credit reports."
    )

st.markdown("---")

# Demo launch
st.subheader("Get started")

already_loaded = "internal_df" in st.session_state and "external_df" in st.session_state

if already_loaded:
    st.success(
        "Demo data is loaded. Use the sidebar to navigate the pipeline."
    )
    st.caption(
        f"internal_df: {st.session_state['internal_df'].shape[0]:,} rows  |  "
        f"external_df: {st.session_state['external_df'].shape[0]:,} rows"
    )
else:
    col_demo, col_manual = st.columns([2, 1])

    with col_demo:
        st.markdown(
            "**Launch Demo** loads the bundled CIBIL datasets automatically and "
            "navigates to preprocessing. No file upload required."
        )
        if st.button("Launch Demo", type="primary", use_container_width=True):
            with st.spinner("Loading datasets..."):
                ok = _load_demo_data()
            if ok:
                st.success("Demo data loaded. Navigating to Preprocessing...")
                st.switch_page("app/2_Preprocessing.py")

    with col_manual:
        st.markdown("**Manual upload**")
        st.markdown("Upload your own datasets on the Upload Data page.")
        if st.button("Go to Upload", use_container_width=True):
            st.switch_page("app/1_Data_Upload.py")

st.markdown("---")

# Pipeline reference
st.subheader("Pipeline")

pipeline = [
    ("1 · Upload Data",          "Load Internal Bank + External CIBIL datasets (or use demo)"),
    ("2 · Preprocess",           "Merge on PROSPECTID · OHE encode · Median impute · StandardScale"),
    ("3 · Train Models",         "Logistic Regression baseline + EBM · Stratified 80/20 split · F1, AUC, confusion matrix"),
    ("4 · Explainability",       "Global SHAP (mean |SHAP| per feature) · Local SHAP waterfall · LIME · PDP"),
    ("5 · Score New Data",       "Transform unseen applicants · Predict P1–P4 · Per-class probabilities · PDF reports"),
    ("6 · Fairness Audit",       "DPD · EOD · Selection rate heatmap · RBI MRM traffic-light thresholds"),
    ("7 · Credit Analyst Agent", "Natural language Q&A · Intent classification · SHAP-grounded responses"),
    ("8 · Business Summary",     "NPA exposure avoided · Tier breakdown · Configurable assumptions · PDF export"),
]

for name, desc in pipeline:
    st.markdown(f"**{name}** — {desc}")

st.markdown("---")

# Key concepts
with st.expander("Key concepts — EBM, SHAP, DPD, EOD, NPA"):
    st.markdown(KEY_CONCEPTS)

# Default rate rationale
with st.expander("Default rate assumptions (used in Business Summary)"):
    st.markdown(
        "The Business Summary uses **P4 default rate = 40%** and "
        "**P3 default rate = 15%** as defaults. These are conservative estimates "
        "for unsecured personal loans in India based on CRISIL rating agency data "
        "and RBI Financial Stability Reports (2022–2024):\n\n"
        "- Subprime (NPA-risk) unsecured loans: 30–50% historical default rates\n"
        "- Sub-standard (watch-list) loans: 10–20% roll-rate to NPA\n\n"
        "Both values are adjustable via sliders on the Business Summary page."
    )
