import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet


def load_model(name, session_key):
    if session_key in st.session_state:
        return st.session_state[session_key]

    path = os.path.join("models", name)
    if os.path.exists(path):
        model = joblib.load(path)
        st.session_state[session_key] = model
        return model
    return None


def save_pdf_report(index, result_row, probability=None):
    """Generate PDF explanation for an applicant."""
    os.makedirs("reports", exist_ok=True)

    filename = f"reports/applicant_{index}_report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph(f"<b>Credit Scoring Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Applicant Index:</b> {index}", styles["BodyText"]))
    story.append(Spacer(1, 6))

    for col, val in result_row.items():
        story.append(Paragraph(f"<b>{col}:</b> {val}", styles["BodyText"]))

    if probability is not None:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Approval Probability:</b> {probability:.3f}", styles["BodyText"]))

    doc.build(story)
    return filename


def main():
    st.title("✅ Score New Applicants & Generate Insights")

    # Step 1: Load unseen data
    st.subheader("1️⃣ Load Unseen Dataset")

    unseen_df = None

    if "unseen_df" in st.session_state:
        unseen_df = st.session_state["unseen_df"]
        st.success("✅ Using previously uploaded unseen dataset.")
        st.dataframe(unseen_df.head())

    uploaded_file = st.file_uploader("Upload Unseen CSV", type="csv")
    if uploaded_file:
        try:
            unseen_df = pd.read_csv(uploaded_file)
            st.session_state["unseen_df"] = unseen_df
            st.success("✅ Unseen dataset loaded.")
            st.dataframe(unseen_df.head())
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return

    if unseen_df is None:
        st.warning("Upload unseen dataset first.")
        return

    # Step 2: Load trained models
    st.subheader("2️⃣ Select Model for Predictions")
    lr = load_model("logistic_regression.pkl", "lr_model")
    ebm = load_model("ebm_model.pkl", "ebm_model")

    model_options = {}
    if lr:
        model_options["Logistic Regression"] = lr
    if ebm:
        model_options["Explainable Boosting Machine (EBM)"] = ebm

    if not model_options:
        st.error("❌ No trained models found. Train models first.")
        return

    model_name = st.selectbox("Choose model", list(model_options.keys()))
    model = model_options[model_name]

    st.markdown("---")

    # Step 3: Predict on new applicants
    st.subheader("3️⃣ Predict Approvals for Applicants")

    if st.button("Run Prediction"):
        try:
            predict_df = unseen_df.copy()
            if "Approved_Flag" in predict_df.columns:
                predict_df = predict_df.drop(columns=["Approved_Flag"])

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(predict_df)[:, 1]
                preds = (probs >= 0.5).astype(int)
            else:
                preds = model.predict(predict_df)
                probs = None

            result = predict_df.copy()
            result["Predicted_Approval"] = preds
            if probs is not None:
                result["Approval_Probability"] = probs

            st.session_state["prediction_result"] = result
            st.session_state["predict_df"] = predict_df

            st.success("✅ Prediction complete!")
            st.dataframe(result.head())

            # ✅ Save to CSV
            os.makedirs("reports", exist_ok=True)
            csv_path = "reports/scored_output.csv"
            result.to_csv(csv_path, index=False)
            st.download_button(
                label="⬇ Download Results as CSV",
                data=result.to_csv(index=False),
                file_name="scored_output.csv",
                mime="text/csv"
            )
            st.write(f"✅ Results saved to `{csv_path}`")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    st.markdown("---")

    # Step 4: SHAP, LIME & PDP for individual explanation
    st.subheader("4️⃣ Explain Individual Applicant")

    if "prediction_result" not in st.session_state:
        st.info("Run prediction above first.")
        return

    result = st.session_state["prediction_result"]
    predict_df = st.session_state["predict_df"]

    idx = st.number_input(
        "Select applicant index",
        min_value=0,
        max_value=len(result) - 1,
        value=0,
        step=1,
    )

    instance_data = predict_df.iloc[[idx]]
    instance_full = result.iloc[idx]

    # ✅ SHAP Local
    if st.button("Show SHAP Explanation"):
        try:
            explainer = shap.Explainer(model, predict_df)
            shap_values = explainer(predict_df)
            single = shap_values[int(idx)]

            st.write("✅ SHAP Waterfall Plot")
            plt.figure(figsize=(8, 4))
            shap.plots.waterfall(single, show=False)
            st.pyplot(plt.gcf())
            plt.clf()

        except Exception as e:
            st.error(f"❌ SHAP failed: {e}")

    # ✅ LIME Local
    if st.button("Show LIME Explanation"):
        try:
            mode = "classification" if hasattr(model, "predict_proba") else "regression"
            X_np = predict_df.values
            explainer = LimeTabularExplainer(
                X_np,
                feature_names=predict_df.columns.tolist(),
                class_names=["Reject", "Approve"] if mode == "classification" else None,
                mode=mode,
                discretize_continuous=True
            )
            instance = X_np[int(idx)]
            if mode == "classification":
                exp = explainer.explain_instance(instance, model.predict_proba, num_features=10)
            else:
                exp = explainer.explain_instance(instance, model.predict, num_features=10)

            st.components.v1.html(exp.as_html(), height=420, scrolling=True)

        except Exception as e:
            st.error(f"❌ LIME failed: {e}")

    # ✅ PDP for a chosen feature
    st.markdown("---")
    st.subheader("📈 Partial Dependence Plot (PDP)")
    feature_for_pdp = st.selectbox("Choose a feature", predict_df.columns.tolist())

    if st.button("Show PDP"):
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            PartialDependenceDisplay.from_estimator(model, predict_df, [feature_for_pdp], ax=ax)
            st.pyplot(fig)
            plt.clf()
        except Exception as e:
            st.error(f"❌ PDP failed: {e}")

    # ✅ Multi-instance SHAP Comparison
    st.markdown("---")
    st.subheader("📊 Compare SHAP for Multiple Applicants")

    selected_rows = st.multiselect(
        "Select multiple rows",
        list(range(len(result))),
        default=[0] if len(result) > 0 else []
    )

    if st.button("Compare SHAP"):
        if len(selected_rows) < 2:
            st.warning("Pick at least two applicants to compare.")
        else:
            try:
                explainer = shap.Explainer(model, predict_df)
                shap_values = explainer(predict_df)

                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values[selected_rows], predict_df.iloc[selected_rows], show=False)
                st.pyplot(plt.gcf())
                plt.clf()
            except Exception as e:
                st.error(f"❌ Failed SHAP comparison: {e}")

    # ✅ PDF report for selected applicant
    st.markdown("---")
    st.subheader("🧾 Export Applicant Report to PDF")

    if st.button("Generate PDF Report"):
        try:
            probability = instance_full["Approval_Probability"] if "Approval_Probability" in instance_full else None
            pdf_path = save_pdf_report(idx, instance_full, probability)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="⬇ Download PDF Report",
                    data=pdf_file,
                    file_name=f"applicant_{idx}_report.pdf",
                    mime="application/pdf"
                )
            st.success(f"✅ PDF generated: {pdf_path}")

        except Exception as e:
            st.error(f"❌ PDF generation failed: {e}")


if __name__ == "__main__":
    main()
