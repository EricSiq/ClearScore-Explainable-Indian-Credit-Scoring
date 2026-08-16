# Architecture decisions

## ADR-001: EBM remains the primary model

XClearScore uses InterpretML's `ExplainableBoostingClassifier` as its primary model and Logistic Regression as a baseline. EBM retains nonlinear credit-risk patterns while exposing additive shape functions that can be inspected directly.

## ADR-002: SHAP and LIME are complementary views

SHAP attribution and LIME local approximation remain part of the application. They run only after an explicit user action and operate on bounded samples in the hosted profile. This prevents expensive computation from blocking application startup while retaining the intended XAI workflow.

## ADR-003: Full benchmark and hosted demo are separate profiles

The 95.6% accuracy / 0.982 macro-AUC benchmark belongs to the full-data local validation run (10,268 holdout applicants). Community Cloud runs a stratified 10,000-row demo and reports its own metrics. The UI and documentation make this distinction explicit.

## ADR-004: Native fairness diagnostics

Demographic parity and equalised-odds gaps are computed with pandas/numpy. They are monitoring signals, not a fairness certification or a basis for automated group-specific lending decisions.
