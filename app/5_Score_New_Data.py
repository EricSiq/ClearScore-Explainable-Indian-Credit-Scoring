import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# ── Shared components ─────────────────────────────────────────────────────────
from app.components.model_loader import load_model as _load_model_from_disk
from app.components.utils        import get_label_map
from app.components.shap_plotter import get_shap_values, shap_top_features
from app.components.lime_plotter import lime_local_explanation
from app.components.pdp_plotter  import plot_pdp

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_COL  = "Approved_Flag"
MODEL_DIR   = "app/models"
REPORTS_DIR = "reports"
_LABEL_MAP  = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}

# Business meaning of each tier — shown in results table and PDF
TIER_DESCRIPTIONS = {
    "P1": "Excellent — approve, best terms",
    "P2": "Good — approve, standard terms",
    "P3": "Marginal — conditional approval or higher rate",
    "P4": "Poor — reject or secured product only",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _label_map() -> dict:
    return get_label_map()


def _load_model(filename: str, session_key: str):
    return _load_model_from_disk(filename, session_key)


def _transform_unseen(unseen_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the fitted ColumnTransformer from preprocessing to the unseen dataset,
    then align columns to the training feature set.

    The unseen dataset is a 42-column subset of the full 85-column feature space.
    Before transforming, we reindex it to match the columns the ColumnTransformer
    was fitted on, filling any missing columns with 0 (numeric) or the most
    frequent value (categorical). After OHE expansion the result is aligned to
    the 103-column training feature set.
    """
    preprocessor  = st.session_state.get("preprocessor")
    feature_names = st.session_state.get("feature_names")

    if preprocessor is None or feature_names is None:
        st.error(
            "Fitted preprocessor not found in session. "
            "Run **Preprocessing** and **Train Models** first."
        )
        st.stop()

    # Drop target if accidentally present
    df = unseen_df.copy()
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # Reconstruct the full column set the ColumnTransformer was fitted on.
    # We can read this from the transformer's own fitted state.
    all_train_cols = []
    for _, transformer, cols in preprocessor.transformers_:
        all_train_cols.extend(cols)

    # Separate numeric and categorical columns as fitted
    cat_cols_fitted = []
    num_cols_fitted = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            cat_cols_fitted = list(cols)
        elif name == "num":
            num_cols_fitted = list(cols)

    # Reindex to full training column set:
    # - Missing numeric cols → fill with 0 (will be scaled, 0 ≈ median after StandardScaler)
    # - Missing categorical cols → fill with most frequent value seen during fit
    for col in num_cols_fitted:
        if col not in df.columns:
            df[col] = 0.0

    for col in cat_cols_fitted:
        if col not in df.columns:
            # Use the most frequent value from the fitted imputer for this column
            cat_transformer = dict(preprocessor.named_transformers_)["cat"]
            imputer = cat_transformer.named_steps["imp"]
            col_idx = cat_cols_fitted.index(col)
            df[col] = imputer.statistics_[col_idx]

    # Now transform — all expected columns are present
    try:
        X_transformed = preprocessor.transform(df)
    except Exception as e:
        st.error(f"Preprocessing transform failed: {e}")
        st.stop()

    X_unseen = pd.DataFrame(X_transformed, columns=feature_names, dtype=np.float32)

    return X_unseen


# ── Prediction ────────────────────────────────────────────────────────────────

def _run_prediction(model, X_unseen: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    """
    Run multi-class prediction and return a results DataFrame with:
      - Predicted_Class   : P1 / P2 / P3 / P4
      - Tier_Description  : business meaning of the predicted class
      - Prob_P1 … Prob_P4 : per-class probabilities (4 columns)
      - Confidence        : probability of the predicted class
    """
    # predict() returns integer class labels (0–3)
    y_pred_int = model.predict(X_unseen)

    # Map integers back to P1/P2/P3/P4
    y_pred_labels = pd.Series(y_pred_int).map(label_map)

    result = X_unseen.copy()
    result["Predicted_Class"] = y_pred_labels.values

    result["Tier_Description"] = result["Predicted_Class"].map(TIER_DESCRIPTIONS)

    # Per-class probabilities — one column per class
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_unseen)          # shape (n, 4)
        # model.classes_ gives the integer order; map to P-labels
        class_order = [label_map[c] for c in model.classes_]
        for i, cls_label in enumerate(class_order):
            result[f"Prob_{cls_label}"] = proba[:, i].round(4)

        # Confidence = probability of the predicted class
        result["Confidence"] = proba[
            np.arange(len(proba)),
            [list(model.classes_).index(c) for c in y_pred_int]
        ].round(4)

    return result


# ── PDF report ────────────────────────────────────────────────────────────────

def _save_pdf_report(
    idx: int,
    applicant_features: pd.Series,
    predicted_class: str,
    confidence: float,
    per_class_probs: dict,
    shap_top: list | None = None,
) -> str:
    """
    Generate a per-applicant PDF credit report.

    Parameters
    ----------
    idx               : applicant row index
    applicant_features: raw feature values (Series)
    predicted_class   : 'P1' / 'P2' / 'P3' / 'P4'
    confidence        : probability of predicted class
    per_class_probs   : {'P1': 0.05, 'P2': 0.72, ...}
    shap_top          : list of (feature_name, shap_value) tuples, top contributors
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = os.path.join(REPORTS_DIR, f"applicant_{idx}_report.pdf")

    doc    = SimpleDocTemplate(filename, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # ── Header ───────────────────────────────────────────────────────────────
    story.append(Paragraph("Credit Scoring Report", styles["Title"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Applicant Index: <b>{idx}</b>", styles["BodyText"]))
    story.append(Spacer(1, 0.3*cm))

    # ── Decision block ────────────────────────────────────────────────────────
    tier_desc = TIER_DESCRIPTIONS.get(predicted_class, "")
    story.append(Paragraph(
        f"<b>Credit Tier:</b> {predicted_class} — {tier_desc}",
        styles["BodyText"]
    ))
    story.append(Paragraph(
        f"<b>Confidence:</b> {confidence:.1%}",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 0.4*cm))

    # ── Per-class probability table ───────────────────────────────────────────
    story.append(Paragraph("<b>Class Probabilities</b>", styles["Heading3"]))
    prob_data = [["Class", "Probability", "Meaning"]] + [
        [cls, f"{prob:.1%}", TIER_DESCRIPTIONS.get(cls, "")]
        for cls, prob in per_class_probs.items()
    ]
    prob_table = Table(prob_data, colWidths=[2*cm, 3*cm, 10*cm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#dce6f1"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Top SHAP contributors ─────────────────────────────────────────────────
    if shap_top:
        story.append(Paragraph("<b>Top Contributing Factors (SHAP)</b>", styles["Heading3"]))
        shap_data = [["Feature", "SHAP Value", "Direction"]] + [
            [name, f"{val:+.4f}", "↑ Positive" if val > 0 else "↓ Negative"]
            for name, val in shap_top
        ]
        shap_table = Table(shap_data, colWidths=[8*cm, 3*cm, 4*cm])
        shap_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#dce6f1"), colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(shap_table)
        story.append(Spacer(1, 0.4*cm))

    # ── Feature values ────────────────────────────────────────────────────────
    story.append(Paragraph("<b>Applicant Feature Values</b>", styles["Heading3"]))
    feat_items = list(applicant_features.items())
    feat_data  = [["Feature", "Value"]] + [
        [str(k), str(round(v, 4) if isinstance(v, float) else v)]
        for k, v in feat_items
    ]
    feat_table = Table(feat_data, colWidths=[9*cm, 6*cm])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#dce6f1"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(feat_table)

    doc.build(story)
    return filename


# ── SHAP helpers — delegated to shap_plotter component ───────────────────────

def _get_shap_values(model, X: pd.DataFrame, model_key: str):
    """Cache key prefixed with 'unseen_' to separate from train-set SHAP."""
    return get_shap_values(model, X, f"unseen_{model_key}")


def _shap_top_features(shap_values, idx: int, n: int = 10) -> list:
    """Delegate to shared shap_plotter component."""
    return shap_top_features(shap_values, idx, n)


# ── Main page ─────────────────────────────────────────────────────────────────

def main():
    st.title("✅ Score New Applicants")

    label_map = _label_map()

    # ── Step 1: Load unseen data ──────────────────────────────────────────────
    st.subheader("1 — Load Unseen Dataset")

    unseen_df = st.session_state.get("unseen_df")

    if unseen_df is not None:
        st.success("Using previously uploaded unseen dataset.")
        with st.expander("Preview raw data"):
            st.dataframe(unseen_df.head())

    uploaded = st.file_uploader(
        "Upload a new unseen dataset (.xlsx or .csv)",
        type=["xlsx", "csv"],
    )
    if uploaded:
        try:
            if uploaded.name.endswith(".xlsx"):
                unseen_df = pd.read_excel(uploaded)
            else:
                unseen_df = pd.read_csv(uploaded)
            st.session_state["unseen_df"] = unseen_df
            st.success(f"Loaded {len(unseen_df):,} applicants.")
            with st.expander("Preview raw data"):
                st.dataframe(unseen_df.head())
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

    if unseen_df is None:
        st.warning("Upload an unseen dataset to continue.")
        return

    # ── Step 2: Select model ──────────────────────────────────────────────────
    st.subheader("2 — Select Model")

    lr  = _load_model("logistic_regression.pkl", "lr_model")
    ebm = _load_model("ebm_model.pkl",           "ebm_model")

    model_options = {}
    if lr:
        model_options["Logistic Regression"] = lr
    if ebm:
        model_options["Explainable Boosting Machine (EBM)"] = ebm

    if not model_options:
        st.error("No trained models found. Run **Train Models** first.")
        return

    model_key  = st.selectbox("Model", list(model_options.keys()))
    model      = model_options[model_key]

    st.markdown("---")

    # ── Step 3: Predict ───────────────────────────────────────────────────────
    st.subheader("3 — Predict Credit Tiers")

    if st.button("▶ Run Prediction"):
        with st.spinner("Transforming unseen data with fitted preprocessor..."):
            try:
                X_unseen = _transform_unseen(unseen_df)
            except SystemExit:
                return

        with st.spinner("Running predictions..."):
            try:
                result = _run_prediction(model, X_unseen, label_map)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        st.session_state["prediction_result"] = result
        st.session_state["predict_df"]        = X_unseen

        st.success(f"✅ Scored {len(result):,} applicants.")

        # ── Results summary ───────────────────────────────────────────────────
        tier_counts = result["Predicted_Class"].value_counts().reindex(
            ["P1", "P2", "P3", "P4"]
        ).fillna(0).astype(int)

        col1, col2, col3, col4 = st.columns(4)
        for col, tier in zip([col1, col2, col3, col4], ["P1", "P2", "P3", "P4"]):
            col.metric(
                label=f"{tier} — {tier.replace('P','Tier ')}",
                value=int(tier_counts[tier]),
                help=TIER_DESCRIPTIONS[tier],
            )

        # Show result table with key columns first
        display_cols = (
            ["Predicted_Class", "Tier_Description", "Confidence"]
            + [c for c in result.columns if c.startswith("Prob_")]
        )
        st.dataframe(result[display_cols], use_container_width=True)

        # ── CSV download ──────────────────────────────────────────────────────
        os.makedirs(REPORTS_DIR, exist_ok=True)
        csv_path = os.path.join(REPORTS_DIR, "scored_output.csv")
        result.to_csv(csv_path, index=False)

        st.download_button(
            label="⬇ Download Scored CSV",
            data=result.to_csv(index=False),
            file_name="scored_output.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ── Step 4: Individual explanations ──────────────────────────────────────
    st.subheader("4 — Explain Individual Applicant")

    if "prediction_result" not in st.session_state:
        st.info("Run prediction above first.")
        return

    result     = st.session_state["prediction_result"]
    X_unseen   = st.session_state["predict_df"]

    idx = st.number_input(
        "Applicant index",
        min_value=0,
        max_value=len(result) - 1,
        value=0,
        step=1,
    )
    idx = int(idx)

    # Applicant summary card
    row = result.iloc[idx]
    pred_class  = row["Predicted_Class"]
    confidence  = row.get("Confidence", None)
    prob_cols   = {c.replace("Prob_", ""): row[c] for c in result.columns if c.startswith("Prob_")}

    st.info(
        f"**Applicant {idx}** — Predicted: **{pred_class}** "
        f"({TIER_DESCRIPTIONS.get(pred_class, '')})  |  "
        f"Confidence: **{confidence:.1%}**" if confidence is not None
        else f"**Applicant {idx}** — Predicted: **{pred_class}**"
    )

    if prob_cols:
        prob_df = pd.DataFrame(
            {"Class": list(prob_cols.keys()), "Probability": list(prob_cols.values())}
        ).set_index("Class")
        st.bar_chart(prob_df)

    st.markdown("---")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    if st.button("Show SHAP Waterfall"):
        try:
            shap_values = _get_shap_values(model, X_unseen, model_key)
            single = shap_values[idx]
            fig, _ = plt.subplots(figsize=(9, 5))
            shap.plots.waterfall(single, show=False)
            st.pyplot(plt.gcf())
            plt.close("all")
        except Exception as e:
            st.error(f"SHAP failed: {e}")

    # ── LIME ─────────────────────────────────────────────────────────────────
    if st.button("Show LIME Explanation"):
        lime_local_explanation(model, X_unseen, idx, label_map)

    # ── PDP ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Partial Dependence Plot")
    feature_for_pdp = st.selectbox("Feature", X_unseen.columns.tolist())
    if st.button("Show PDP"):
        plot_pdp(model, X_unseen, feature_for_pdp)

    # ── Multi-applicant SHAP comparison ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Compare SHAP Across Applicants")
    selected = st.multiselect(
        "Select applicant indices to compare",
        list(range(len(result))),
        default=[0, 1] if len(result) > 1 else [0],
    )
    if st.button("Compare SHAP"):
        if len(selected) < 2:
            st.warning("Select at least two applicants.")
        else:
            try:
                shap_values = _get_shap_values(model, X_unseen, model_key)
                plt.figure(figsize=(9, 6))
                shap.summary_plot(
                    shap_values[selected],
                    X_unseen.iloc[selected],
                    show=False,
                )
                st.pyplot(plt.gcf())
                plt.close("all")
            except Exception as e:
                st.error(f"SHAP comparison failed: {e}")

    # ── PDF report ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Export PDF Report")

    if st.button("Generate PDF Report"):
        try:
            # Collect SHAP top features if available
            shap_top = None
            cache_key = f"shap_values_unseen_{model_key}"
            if cache_key in st.session_state:
                shap_top = _shap_top_features(
                    st.session_state[cache_key], idx, n=10
                )

            pdf_path = _save_pdf_report(
                idx=idx,
                applicant_features=X_unseen.iloc[idx],
                predicted_class=pred_class,
                confidence=float(confidence) if confidence is not None else 0.0,
                per_class_probs=prob_cols,
                shap_top=shap_top,
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇ Download PDF Report",
                    data=f,
                    file_name=f"applicant_{idx}_report.pdf",
                    mime="application/pdf",
                )
            st.success(f"PDF saved to `{pdf_path}`")

        except Exception as e:
            st.error(f"PDF generation failed: {e}")


if __name__ == "__main__":
    main()
