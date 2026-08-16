# XClearScore business context and responsible-use statement

## Case-study purpose

XClearScore demonstrates the review workflow around a four-tier credit-risk classifier using a bundled Indian-credit-style dataset. It uses an Explainable Boosting Machine with SHAP and LIME to make model evidence visible alongside validation and group diagnostics.

## What a credit analyst needs from a prototype

- A repeatable way to inspect training data and a held-out evaluation.
- A model score that can be decomposed into understandable evidence.
- Clear separation between a model output and an underwriting decision.
- A way to identify group-level monitoring signals that deserve further investigation.

The app supports those activities at a demonstration level. It does not replace policy, manual review, affordability assessment, fraud controls, or institutional risk governance.

## Interpretation safeguards

`P1` to `P4` are model classes in the dataset; they are not automatically approve/reject instructions. Probabilities are classifier outputs, not estimates of a person's repayment probability. Coefficient contributions explain a model score, not why a person has a particular financial history.

Group metrics are descriptive holdout-sample diagnostics. Differences can arise from data quality, sample size, target construction, proxy variables, threshold choices, or genuine performance differences. They should trigger analysis, not automated group-specific treatment.

## Data boundary

The bundled files are suitable for a case study only. Do not upload personal, bureau, or confidential lender data to a public deployment. Real deployment requires a lawful basis for processing, data minimisation, security controls, retention rules, and qualified legal/compliance review.

## Next validation steps

Before using any related approach beyond demonstration: validate on out-of-time data, examine calibration and subgroup uncertainty, assess stability and drift, establish a model-approval process, document policy overrides, and govern any customer-facing reason-code or notice workflow.
