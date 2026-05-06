import streamlit as st

st.set_page_config(page_title="Explainable AI Credit Scoring", page_icon="💳", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.page_link("app/Home.py",              label="🏠 Home")
st.sidebar.page_link("app/1_Data_Upload.py",     label="📁 Upload Data")
st.sidebar.page_link("app/2_Preprocessing.py",   label="🔧 Preprocess & Engineer Features")
st.sidebar.page_link("app/3_Model_Training.py",  label="🤖 Train Models")
st.sidebar.page_link("app/4_Explainability.py",  label="🔍 Explainability (SHAP/LIME/PDP)")
st.sidebar.page_link("app/5_Score_New_Data.py",  label="✅ Score New Applicants")
st.sidebar.page_link("app/6_Fairness_Audit.py",  label="⚖️ Fairness Audit")
st.sidebar.page_link("app/7_Credit_Analyst_Agent.py", label="🤖 Credit Analyst Agent")
st.sidebar.page_link("app/8_Business_Summary.py",    label="₹ Business Impact Summary")

st.write("# Explainable AI-based Indian Credit Scoring System")
