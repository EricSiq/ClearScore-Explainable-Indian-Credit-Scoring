import streamlit as st

st.set_page_config(page_title="Explainable AI Credit Scoring", page_icon="💳", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.page_link("app/Home.py", label="🏠 Home")
st.sidebar.page_link("app/01_Data_Upload.py", label="📁 Upload Data")
st.sidebar.page_link("app/02_Preprocessing.py", label="🔧 Preprocess & Engineer Features")
st.sidebar.page_link("app/03_Model_Training.py", label="🤖 Train Models")
st.sidebar.page_link("app/04_Explainability.py", label="🔍 Explainability (SHAP/LIME/PDP)")
st.sidebar.page_link("app/05_Score_New_Data.py", label="✅ Score New Applicants")

st.write("# Explainable AI-based Indian Credit Scoring System")
