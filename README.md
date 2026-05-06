# Explainable Indian Credit Scoring System
### An XAI-powered loan approval dashboard with fairness auditing and a Credit Analyst Agent

**Version**: 2.0 | **Status**: Complete  
**Stack**: Python · Streamlit · InterpretML (EBM) · SHAP · LIME · llama.cpp

---

## What This Is

An end-to-end machine learning dashboard for loan approval prediction, built specifically for the Indian credit market. The system ingests internal bank trade-line data and external CIBIL bureau data, trains interpretable models, and provides multi-layered explanations — from global feature importance to per-applicant natural language reasoning.

The primary model is an **Explainable Boosting Machine (EBM)** — a glass-box model that achieves gradient-boosting-level accuracy while producing exact, auditable explanations. This is not post-hoc SHAP on a black box. The model's decision function *is* the explanation.

---

## The Problem

India has ~190 million unbanked adults and a ₹25–30 lakh crore MSME credit gap. Traditional CIBIL scoring fails thin-file borrowers. Existing ML-based credit models are black boxes that:
- Cannot explain individual decisions to customers or regulators
- Have no built-in fairness monitoring (gender, education bias goes undetected)
- Produce outputs that credit officers cannot interrogate or trust

This system addresses all three gaps.

---

## Pipeline

```
Upload Data → Merge (PROSPECTID) → Preprocess → Train → Explain → Fairness Audit → Score → Export
```

| Step | Page | Output |
|------|------|--------|
| Upload internal + external datasets | 1_Data_Upload | `internal_df`, `external_df` in session |
| Merge, encode, scale | 2_Preprocessing | `processed_df`, `feature_names`, `preprocessor` |
| Train LR + EBM, evaluate | 3_Model_Training | `.pkl` models, `model_metrics` |
| SHAP global/local, LIME, PDP | 4_Explainability | Interactive visualizations |
| Score unseen applicants | 5_Score_New_Data | CSV + per-applicant PDF |
| Demographic parity, equalized odds | 6_Fairness_Audit | Fairness heatmap + narrative |
| Natural language Q&A | 7_Credit_Analyst_Agent | LLM-backed explanations |
| ₹ NPA impact framing | 8_Business_Summary | Business PDF report |

---

## Dataset

| Dataset | Rows | Cols | Key Features |
|---------|------|------|-------------|
| Internal Bank | 51,336 | 26 | Trade lines, missed payments, loan type counts |
| External CIBIL | 51,336 | 62 | Delinquency, enquiries, demographics, credit score |
| Unseen (scoring) | 100 | 42 | Subset of merged features, no target |

**Target**: `Approved_Flag` — 4-class ordinal
- P1 (11.3%): Excellent — approve, best terms
- P2 (62.7%): Good — approve, standard terms  
- P3 (14.5%): Marginal — conditional approval
- P4 (11.5%): Poor — reject or secured only

**Merge key**: `PROSPECTID` (present in both datasets)

---

## Models

| Model | Type | Accuracy | Explainability |
|-------|------|----------|----------------|
| Logistic Regression | Baseline | 95.9%* | Coefficients |
| EBM (InterpretML) | Primary | 95.9%+ | Exact shape functions |

*LR accuracy on full 51,336-row dataset, 80/20 stratified split. EBM expected to match or exceed — run training to confirm.

**Why EBM over XGBoost?** EBM's explanations are exact — the shape functions *are* the model. SHAP on XGBoost is a post-hoc approximation. For regulated credit scoring, exact explanations are the correct choice. See [DECISIONS.md](DECISIONS.md#adr-001).

---

## Fairness Audit

The system computes fairness metrics across sensitive demographic attributes, implemented natively (no external fairness library dependency):

- **Demographic Parity Difference** across GENDER, EDUCATION, MARITALSTATUS
- **Equalized Odds Difference** (TPR and FPR parity)
- **Selection Rate by Group** (approval rate per demographic segment)

Thresholds aligned with RBI Model Risk Management Guidelines (2023):
- < 0.05: Acceptable
- 0.05–0.10: Monitor and document
- > 0.10: Mitigation required

---

## Credit Analyst Agent

A chat-style interface where credit officers can ask questions in plain English:

```
"Why was applicant 15 rejected?"
→ SHAP waterfall + LIME → GPT-4o-mini → natural language explanation

"Compare applicants 3 and 7"
→ SHAP delta → NL summary of key differences

"What drives approvals?"
→ Global SHAP → NL feature importance narrative

"Show fairness for GENDER"
→ MetricFrame → NL fairness summary
```

Works without an API key (structured text fallback). Full LLM responses require `OPENAI_API_KEY`.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/AUCML_CreditScoring.git
cd AUCML_CreditScoring

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Optional: LLM Agent
```bash
# Add to .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

---

## Run

```bash
streamlit run app/Home.py
```

Navigate pages using the left sidebar. Run them in order (Upload → Preprocess → Train → ...) as each page depends on session state from the previous.

---

## Tech Stack

| Layer | Library | Version |
|-------|---------|---------|
| Dashboard | Streamlit | 1.56.0 |
| Data | Pandas, NumPy | 2.3.3, 2.3.5 |
| ML | Scikit-learn | 1.8.0 |
| Glass-box model | InterpretML (EBM) | 0.7.8 |
| Post-hoc XAI | SHAP | 0.51.0 |
| Local XAI | LIME | 0.2.0.1 |
| Fairness | Native (pandas/numpy) | — |
| Agent SLM | llama.cpp CLI | binary |
| Visualization | Matplotlib, Seaborn | 3.10.8, 0.13.2 |
| PDF export | ReportLab | 4.4.10 |
| Model persistence | Joblib | 1.5.3 |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [TASKS.md](TASKS.md) | Phase 2 task list with acceptance criteria |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data flow, feature schema |
| [DECISIONS.md](DECISIONS.md) | Architecture Decision Records (ADRs) |
| [BUSINESS_CONTEXT.md](BUSINESS_CONTEXT.md) | Market context, regulatory landscape, NPA framing |
| [DesignDoc.json](DesignDoc.json) | Original design specification |

---

## Regulatory Alignment

This system is designed with the following Indian regulatory frameworks in mind:

- **RBI Model Risk Management Guidelines (2023)**: explainability, validation, audit trail
- **Digital Personal Data Protection Act (2023)**: data minimization, right to explanation
- **RBI Fair Lending Guidelines**: non-discrimination, demographic parity monitoring

---

## Known Limitations

1. Built on synthetic/anonymized data — production deployment requires real bureau data agreements
2. No geographic features (state, city tier) — significant predictor in Indian credit risk
3. No alternative data (UPI, utility, GST) — critical for thin-file borrowers
4. Session state resets on browser refresh — not suitable for multi-user production deployment
5. Temporal validation not implemented — model must be validated on out-of-time samples before production

See [BUSINESS_CONTEXT.md](BUSINESS_CONTEXT.md#dataset-limitations-and-honest-caveats) for full disclosure.

---

## Phase Roadmap

| Phase | Status | Scope |
|-------|--------|-------|
| v0.1 MVP | ✅ Complete | Core pipeline: upload → train → explain → score |
| v2.0 | ✅ Complete | EBM fix, fairness audit, agent interface, business framing, components |
| v3.0 | 📋 Planned | Real data APIs, database persistence, counterfactuals, deployment |
