# XClearScore architecture

```text
Internal bank data + CIBIL data
              |
      merge on PROSPECTID
              |
 ColumnTransformer: impute, encode, scale
              |
  80/20 stratified split (seed 42)
        |                    |
 Logistic Regression       InterpretML EBM (primary)
        |                    |
    baseline metrics      native shape functions
                              |
                     SHAP + LIME on demand
                              |
        fairness diagnostics / unseen-data scoring
```

The app keeps data, the preprocessor, models, and metrics in Streamlit session state. EBM and Logistic Regression artifacts are also written to the process temporary directory for page navigation during an active session.

## Deployment profiles

The **Community Cloud demo** trains EBM on a capped, stratified 10,000-row sample (`interactions=0`, 64 bins, 4 outer bags) and computes SHAP only over 50 sampled applicants with 20 background rows. LIME uses at most 1,000 background rows. These limits constrain the hosted workload without changing the XAI stack.

The **local full validation** profile uses all 51,336 rows and yields the 10,268-applicant holdout used by the published benchmark. It is intended for a machine with more memory than a free Streamlit instance.

## Auditability

Feature names are preserved after preprocessing. EBM provides model-native global evidence; SHAP attributes feature impact; LIME approximates the local decision boundary. The three views are complementary. They should be linked to governed adverse-action reason codes only after independent validation and compliance review.
