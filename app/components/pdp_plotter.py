import streamlit as st
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def plot_pdp(model, X, feature):
    """
    Generate a Partial Dependence Plot (PDP) for a single feature
    and render it in Streamlit with error handling.
    """
    try:
        st.write(f"### 📈 Partial Dependence Plot (Feature: {feature})")

        fig, ax = plt.subplots(figsize=(8, 4))
        PartialDependenceDisplay.from_estimator(model, X, [feature], ax=ax)

        st.pyplot(fig)
        plt.clf()

    except Exception as e:
        st.error(f"❌ Failed to compute PDP for '{feature}': {e}")
