import streamlit as st
import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet


# =======================================================
# ✅ MODEL LOADING
# =======================================================
def load_model(model_filename, session_key):
    """
    Loads model into Streamlit session_state if not already present.
    Looks in /models/<filename>.
    """
    if session_key in st.session_state:
        return st.session_state[session_key]

    path = os.path.join("models", model_filename)
    if os.path.exists(path):
        model = joblib.load(path)
        st.session_state[session_key] = model
        return model

    return None


# =======================================================
# ✅ DATA CHECK HELPERS
# =======================================================
def get_processed_df():
    """Return processed_df from session_state or error."""
    if "processed_df" not in st.session_state:
        st.error("⚠ Data is not preprocessed yet. Run preprocessing first.")
        return None
    return st.session_state["processed_df"]


def get_unseen_df():
    """Return unseen_df from session_state or error."""
    if "unseen_df" not in st.session_state:
        st.error("⚠ Unseen dataset not found. Upload unseen CSV first.")
        return None
    return st.session_state["unseen_df"]


def try_get_Xy(df, target_col="Approved_Flag"):
    """Split dataframe into X, y if target column exists."""
    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' not found.")
        return None, None
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# =======================================================
# ✅ SHAP VISUALIZATION HELPERS
# =======================================================
def plot_global_shap(model, X, max_display=20, title="Global SHAP Summary"):
    """Render global SHAP summary plot safely inside Streamlit."""
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        st.write(f"### {title}")
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X, show=False, max_display=max_display)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"❌ Failed SHAP Global Plot: {e}")


def plot_local_shap(model, X, index):
    """Render local SHAP force / waterfall plot."""
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        single = shap_values[int(index)]
        st.write("### SHAP Waterfall Plot (Local)")

        plt.figure(figsize=(8, 4))
        shap.plots.waterfall(single, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.error(f"❌ Failed SHAP Local Explanation: {e}")


def compare_shap_rows(model, X, selected_rows):
    """
    Compare SHAP impact for multiple applicants.
    selected_rows: list of indices
    """
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values[selected_rows], X.iloc[selected_rows], show=False)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"❌ Failed multi-row SHAP comparison: {e}")


# =======================================================
# ✅ PDP VISUALIZATION
# =======================================================
def plot_pdp(model, X, feature):
    """Create a Partial Dependence Plot in-streamlit."""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        PartialDependenceDisplay.from_estimator(model, X, [feature], ax=ax)
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.error(f"❌ PDP failed: {e}")


# =======================================================
# ✅ SAVE CSV & DOWNLOAD BUTTON
# =======================================================
def save_results_csv(df, filename="scored_output.csv"):
    """
    Saves to /reports and returns path + download button in Streamlit.
    """
    try:
        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", filename)
        df.to_csv(path, index=False)

        st.download_button(
            label="⬇ Download Results as CSV",
            data=df.to_csv(index=False),
            file_name=filename,
            mime="text/csv"
        )

        st.success(f"✅ Results saved: {path}")
        return path
    except Exception as e:
        st.error(f"❌ Could not save CSV: {e}")
        return None


# =======================================================
# ✅ PDF EXPORT
# =======================================================
def save_pdf_report(index, row_data, prob=None):
    """
    Generate a PDF for a selected applicant (explainability summary only).
    """
    try:
        os.makedirs("reports", exist_ok=True)
        filename = f"reports/applicant_{index}_report.pdf"

        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Credit Scoring Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Applicant Index:</b> {index}", styles["BodyText"]))
        story.append(Spacer(1, 6))

        for col, val in row_data.items():
            story.append(Paragraph(f"<b>{col}:</b> {val}", styles["BodyText"]))

        if prob is not None:
            story.append(Spacer(1, 6))
            story.append(
                Paragraph(f"<b>Approval Probability:</b> {prob:.3f}", styles["BodyText"])
            )

        doc.build(story)
        return filename

    except Exception as e:
        st.error(f"❌ Failed to create PDF: {e}")
        return None
