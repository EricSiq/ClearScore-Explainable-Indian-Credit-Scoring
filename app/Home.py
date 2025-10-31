import streamlit as st

def main():
    st.title("💳 Explainable AI-based Indian Credit Scoring System")
    st.write("""
    Welcome to the XAI Credit Scoring Dashboard.
    
    Use the left sidebar to:
    - Upload datasets  
    - Run preprocessing & feature engineering  
    - Train models (Logistic Regression & EBM)  
    - View SHAP, LIME, PDP explainability  
    - Score unseen applicants
    """)

    st.markdown("---")
    st.info("➡ Select an option from the sidebar to get started.")

if __name__ == "__main__":
    main()
