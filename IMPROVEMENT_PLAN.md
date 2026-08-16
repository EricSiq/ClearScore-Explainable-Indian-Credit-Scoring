# CreditLens improvement plan

## Objective

Turn CreditLens into a credible portfolio project: a small, dependable Streamlit application that demonstrates responsible credit-model development rather than a broad collection of expensive demos. The public deployment should open quickly on Streamlit Community Cloud, make its assumptions visible, and never represent a prototype as a production credit-decision service.

## Current-state assessment

The original project has useful ingredients: an Indian-credit data workflow, stratified evaluation, fairness metrics, and a clear business framing. Its weakest design decision is the optional local-language-model "agent". It adds operational complexity, suggests unsupported automation in a high-stakes domain, and encourages a large dependency/runtime footprint without improving the evidence behind an explanation. The app also performs expensive explanation work over large datasets and makes several regulatory claims more strongly than the prototype can support.

## Target product

The deployed app will be a **model review workbench** for a credit-risk analyst. It will:

1. Load data only when a user explicitly chooses the demo or uploads a file.
2. Train one regularised multinomial logistic-regression model on a bounded, stratified sample.
3. Provide exact, reproducible coefficient-contribution explanations for individual predictions.
4. Report model quality and group-level monitoring results with clear caveats.
5. Export scored data as CSV, while deliberately avoiding autonomous recommendations, hidden prompt logic, and on-device LLM downloads.

This is intentionally narrower than a production underwriting platform. That boundary is a strength in a resume project.

## Architecture and deployment changes

| Concern | Decision | Why it improves the project |
| --- | --- | --- |
| Inference | Use a regularised multinomial logistic regression as the deployed model. | Its score is a linear sum of feature contributions, so the local explanation is exact and auditable. |
| Explainability | Replace SHAP/LIME/LLM responses with coefficient contributions and probability outputs. | Removes expensive post-hoc computation and avoids generating unsupported credit rationales. |
| Runtime | Remove InterpretML, SHAP, LIME, Seaborn, and ReportLab from the hosted dependency set. | Reduces build time, memory pressure, and cold-start failure risk on the free tier. |
| Training | Default to a stratified 10,000-row cap; allow a smaller cap in the UI. | Gives a predictable runtime and protects the Community Cloud memory budget. |
| State | Retain Streamlit session state only for the current analyst session. | Honest about prototype limits; no misleading persistence claim. |
| Outputs | Keep CSV download; leave PDF/reporting to a future server-side job. | Avoids a heavyweight runtime dependency and ephemeral-file confusion. |

## XAI and model-risk practice

- Use the model's own coefficients for local explanations. Contributions are `scaled_feature_value x class_coefficient`; describe them as drivers of the selected class score, not causal facts.
- Present confidence as a model probability, not a probability that a person will repay.
- Separate **model explanation** from **adverse-action reason**. A real lender needs policy validation, human review, reason-code governance, and legal/compliance approval before customer-facing use.
- Keep sensitive attributes out of the training feature set where possible; retain them only for monitoring. The preprocessing stage must display exactly which columns are excluded.
- Report group metrics only when group support is adequate, and label them monitoring diagnostics rather than proof of fairness.
- Use a fixed seed, stratified holdout, macro F1, per-class recall, and multiclass OvR AUC. Future work should add temporal validation, calibration, drift monitoring, and an independently governed reason-code mapping.

## UX direction

Use a restrained analyst-workbench visual language: dark navy for navigation, ink/grey text, a single teal accent for positive status, and muted amber/red only for monitoring alerts. Replace marketing copy and emoji-heavy labels with task-oriented names: Overview, Data, Prepare, Train and validate, Decision review, Fairness monitoring, and Portfolio summary. Every page should start with the decision it supports and state its input prerequisites.

## Delivery sequence

1. Slim the deployment dependencies and remove agent execution from navigation.
2. Make training bounded and deterministic; remove EBM training from the hosted path.
3. Implement exact linear contribution views for global and individual explanations.
4. Convert the agent page into deterministic Decision Review.
5. Refresh the home page, README, deployment guide, and architecture notes with precise prototype boundaries.
6. Compile every Python module and run a smoke test of core preprocessing/training helpers.

## Acceptance checks

- `pip install -r requirements.txt` does not need an LLM runtime, InterpretML, SHAP, or LIME.
- The navigation does not expose an agent or model-download workflow.
- A 10,000-row training run produces metrics and an exact local explanation.
- Documentation does not claim RBI approval/alignment or production readiness.
- The app explicitly discloses synthetic/anonymised data, session-only state, fairness limitations, and human-review requirements.

## Future work (intentionally out of scope)

Move artifacts to object storage, introduce an API/service boundary, add audit logs and authentication, perform temporal and calibration validation, map approved reason codes through governance, and have compliance/legal reviewers validate any regulated-decision workflow.
