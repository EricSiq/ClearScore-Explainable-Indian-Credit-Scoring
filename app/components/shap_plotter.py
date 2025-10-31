import streamlit as st
import shap
import matplotlib.pyplot as plt

def shap_global_plot(model, X, max_display=20):
    """
    Render a global SHAP summary plot for a trained model.
    Uses shap.Explainer → summary plot → show in Streamlit.
    """
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        st.write("### 🌍 Global SHAP Summary")
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.error(f"❌ Failed to compute global SHAP summary: {e}")


def shap_local_waterfall(model, X, row_index):
    """
    Render local SHAP waterfall for a specific row index.
    """
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        single = shap_values[row_index]

        st.write(f"### 🔍 SHAP Local Explanation (Row {row_index})")
        plt.figure(figsize=(8, 4))
        shap.plots.waterfall(single, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.error(f"❌ SHAP local explanation failed: {e}")


def shap_compare_rows(model, X, row_list):
    """
    Compare SHAP impacts for multiple applicant rows.
    """
    try:
        if len(row_list) < 2:
            st.warning("Select at least two rows to compare.")
            return

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        st.write(f"### 🔁 SHAP Comparison for Rows: {row_list}")
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values[row_list], X.iloc[row_list], show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.error(f"❌ Multi-row SHAP comparison failed: {e}")
