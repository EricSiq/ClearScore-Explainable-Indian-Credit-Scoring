import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier

def main():
    st.title("Train Models")

    if "processed_df" not in st.session_state:
        st.error("⚠ Preprocess data first.")
        return

    df = st.session_state["processed_df"]

    # Identify target column
    target_col = "Approved_Flag"
    if target_col not in df.columns:
        st.error("❌ 'Approved_Flag' not found — ensure preprocessing generated the label.")
        return

    # Determine numeric vs categorical features
    features = [c for c in df.columns if c != target_col]

    num_cols = df[features].select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df[features].select_dtypes(include=['object','category','bool']).columns.tolist()

    st.write(f"Numeric features: {len(num_cols)}")
    st.write(f"Categorical features: {len(cat_cols)}")

    # Optional handling: If GENDER numeric, treat as categorical
    if 'GENDER' in df.columns and 'GENDER' in num_cols:
        num_cols.remove('GENDER')
        cat_cols.append('GENDER')

    # Prepare X, y
    try:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    except Exception as e:
        st.error(f"Error extracting target: {e}")
        return

    if st.button("Train Models"):
        try:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Logistic Regression
            lr_model = LogisticRegression()
            lr_model.fit(X_train, y_train)

            # EBM
            ebm_model = ExplainableBoostingClassifier()
            ebm_model.fit(X_train, y_train)

            # Ensure model folder exists
            os.makedirs("models", exist_ok=True)

            # Save models
            joblib.dump(lr_model, "models/logistic_regression.pkl")
            joblib.dump(ebm_model, "models/ebm_model.pkl")

            st.success("Models trained & saved successfully")
            st.session_state["lr_model"] = lr_model
            st.session_state["ebm_model"] = ebm_model

        except Exception as e:
            st.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()
