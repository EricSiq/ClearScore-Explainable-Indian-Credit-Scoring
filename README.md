# XClearScore: Explainable AI Indian Credit Scoring System

XClearScore is an interpretable credit-risk pipeline for CIBIL and internal bank trade-line data. It combines an **Explainable Boosting Machine (EBM)** from InterpretML with **SHAP** and **LIME** to support global, local, and model-native decision review.

## Results

The full-data validation benchmark achieved **95.6% classification accuracy** and **0.982 macro-averaged OvR AUC** on a stratified **10,268-applicant holdout**. The benchmark should be reproduced with the local full-validation profile; Streamlit Community Cloud uses a bounded demonstration profile and must not be represented as this full-data result.

## Model and XAI design

- **Logistic Regression** is the baseline.
- **EBM** is the primary model. Its additive shape functions are part of the prediction function and provide the model-native explanation.
- **SHAP** provides feature-attribution views on a bounded, cached sample in the hosted app.
- **LIME** provides an independently generated local linear approximation for a selected applicant.

The application intentionally computes SHAP and LIME only after the user requests them. This preserves the complete XAI workflow while preventing cold-start or memory failures on Streamlit Community Cloud.

## Workflow

1. Load CIBIL and internal-bank datasets, then merge on `PROSPECTID`.
2. Impute, encode, and scale features while retaining names for auditability.
3. Train Logistic Regression and EBM using a fixed-seed stratified split.
4. Compare holdout metrics and inspect the EBM confusion matrix.
5. Explain predictions with EBM global importance, SHAP attribution, and LIME.
6. Review demographic-parity and equalised-odds diagnostics.
7. Score unseen applicants and export a CSV for analyst review.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Use **Community Cloud demo** for a 10,000-row capped EBM training run and bounded explanations. Use **Local full validation** to reproduce the full-data benchmark.

## Responsible-use boundary

This is a portfolio prototype using case-study data. It does not autonomously approve or reject loans, and its explanation views are not customer-facing adverse-action notices. A production implementation would require governed reason-code mapping, temporal validation, calibration analysis, monitoring, security controls, audit logs, and qualified risk/compliance review.

See [ARCHITECTURE.md](ARCHITECTURE.md), [DECISIONS.md](DECISIONS.md), [BUSINESS_CONTEXT.md](BUSINESS_CONTEXT.md), and [DEPLOY.md](DEPLOY.md).
