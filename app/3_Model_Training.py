import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier

def main():
    st.title("🤖 Train Models")

    if "processed_df" not in st.session_state:
        st.error("⚠ Preprocess data first.")
        return

    df = st.session_state["processed_df"]

    # Replace below with original feature/target selection
    try:
        X = df.drop("Approved_Flag", axis=1)
        y = df["Approved_Flag"]
    except:
        st.error("Could not find target column Approved_Flag — ensure preprocessing step added it.")
        return

    if st.button("Train Models"):

        try:
            # Original train-test split code
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Logistic Regression (paste your original code)
            lr_model = LogisticRegression()
            lr_model.fit(X_train, y_train)

            # EBM (paste original EBM training code)
            ebm_model = ExplainableBoostingClassifier()
            ebm_model.fit(X_train, y_train)

            #  Save models
            joblib.dump(lr_model, "models/logistic_regression.pkl")
            joblib.dump(ebm_model, "models/ebm_model.pkl")

            st.success("Models trained & saved successfully")
            st.session_state["lr_model"] = lr_model
            st.session_state["ebm_model"] = ebm_model

        except Exception as e:
            st.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()
