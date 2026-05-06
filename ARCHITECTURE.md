# ARCHITECTURE.md — Explainable Indian Credit Scoring System
## Technical Architecture Reference

**Version**: 2.0-pre  
**Last Updated**: 2026-05-01  

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Streamlit Multi-Page App                         │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  Upload  │→ │ Preproc  │→ │  Train   │→ │   Explainability │   │
│  │  Page 1  │  │  Page 2  │  │  Page 3  │  │     Page 4       │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
│                                    ↓                ↓               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Business │  │  Agent   │  │ Fairness │  │  Score New Data  │   │
│  │ Summary  │  │  Page 7  │  │  Page 6  │  │     Page 5       │   │
│  │  Page 8  │  └──────────┘  └──────────┘  └──────────────────┘   │
│  └──────────┘                                                       │
│                                                                     │
│  ─────────────────── Shared State (session_state) ──────────────── │
│  internal_df │ external_df │ processed_df │ feature_names          │
│  lr_model    │ ebm_model   │ model_metrics│ label_map              │
│  unseen_df   │ prediction_result          │ chat_history           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Components Layer                            │
│  utils.py │ model_loader.py │ shap_plotter.py │ lime_plotter.py    │
│  pdp_plotter.py │ fairness_auditor.py (new) │ agent_tools.py (new) │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          Storage Layer                              │
│  app/models/logistic_regression.pkl                                 │
│  app/models/ebm_model.pkl                                           │
│  reports/scored_output.csv                                          │
│  reports/applicant_XX_report.pdf                                    │
│  reports/business_summary.pdf                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Training Pipeline

```
Internal_Bank_Dataset.xlsx (51,336 × 26)
        +
External_Cibil_Dataset.xlsx (51,336 × 62)
        ↓
[Merge on PROSPECTID → 51,336 × 87]
        ↓
[Drop: PROSPECTID, Approved_Flag]
        ↓
[ColumnTransformer]
  ├── Numeric (80 cols): SimpleImputer(median) → StandardScaler
  └── Categorical (5 cols): SimpleImputer(most_frequent) → OneHotEncoder
        ↓
[Named DataFrame: ~100+ features after OHE expansion]
        ↓
[Label encode target: P1→0, P2→1, P3→2, P4→3]
        ↓
[train_test_split: 80/20, stratified, random_state=42]
        ↓
┌─────────────────┐    ┌─────────────────────────────────┐
│ LogisticRegres- │    │ ExplainableBoostingClassifier   │
│ sion(multi_     │    │ (InterpretML, glass-box,        │
│ class='ovr',    │    │  natively multi-class)          │
│ max_iter=1000)  │    │                                 │
└─────────────────┘    └─────────────────────────────────┘
        ↓                          ↓
  logistic_regression.pkl      ebm_model.pkl
```

### Inference Pipeline (Score New Data)

```
Unseen_Dataset.xlsx (100 × 42)
        ↓
[Apply same ColumnTransformer (fitted on training data)]
        ↓
[model.predict → P1/P2/P3/P4]
[model.predict_proba → [p_P1, p_P2, p_P3, p_P4]]
        ↓
[SHAP Explainer → per-applicant feature contributions]
[LIME Explainer → local linear approximation]
        ↓
[scored_output.csv + applicant_XX_report.pdf]
```

---

## Feature Schema

### Internal Bank Features (26 total, all numeric)

| Feature | Description | Type |
|---------|-------------|------|
| `PROSPECTID` | Unique applicant identifier | ID (drop) |
| `Total_TL` | Total trade lines ever | int |
| `Tot_Closed_TL` | Total closed trade lines | int |
| `Tot_Active_TL` | Total active trade lines | int |
| `Total_TL_opened_L6M` | Trade lines opened in last 6 months | int |
| `Tot_TL_closed_L6M` | Trade lines closed in last 6 months | int |
| `pct_tl_open_L6M` | % trade lines opened in L6M | float |
| `pct_tl_closed_L6M` | % trade lines closed in L6M | float |
| `pct_active_tl` | % of trade lines currently active | float |
| `pct_closed_tl` | % of trade lines closed | float |
| `Total_TL_opened_L12M` | Trade lines opened in last 12 months | int |
| `Tot_TL_closed_L12M` | Trade lines closed in last 12 months | int |
| `pct_tl_open_L12M` | % trade lines opened in L12M | float |
| `pct_tl_closed_L12M` | % trade lines closed in L12M | float |
| `Tot_Missed_Pmnt` | Total missed payments (all time) | int |
| `Auto_TL` | Auto loan trade lines | int |
| `CC_TL` | Credit card trade lines | int |
| `Consumer_TL` | Consumer loan trade lines | int |
| `Gold_TL` | Gold loan trade lines | int |
| `Home_TL` | Home loan trade lines | int |
| `PL_TL` | Personal loan trade lines | int |
| `Secured_TL` | Secured trade lines | int |
| `Unsecured_TL` | Unsecured trade lines | int |
| `Other_TL` | Other trade lines | int |
| `Age_Oldest_TL` | Age of oldest trade line (months) | int |
| `Age_Newest_TL` | Age of newest trade line (months) | int |

### External CIBIL Features (61 features + target)

**Delinquency Features** (key risk signals):

| Feature | Description |
|---------|-------------|
| `time_since_recent_payment` | Months since last payment |
| `time_since_first_deliquency` | Months since first delinquency |
| `time_since_recent_deliquency` | Months since most recent delinquency |
| `num_times_delinquent` | Total delinquency count |
| `max_delinquency_level` | Worst delinquency level ever |
| `num_deliq_6mts` / `num_deliq_12mts` | Delinquencies in L6M / L12M |
| `num_times_30p_dpd` | Times 30+ days past due |
| `num_times_60p_dpd` | Times 60+ days past due |

**Account Status Features**:

| Feature | Description |
|---------|-------------|
| `num_std` | Standard (performing) accounts |
| `num_sub` | Sub-standard accounts |
| `num_dbt` | Doubtful accounts |
| `num_lss` | Loss accounts |
| `recent_level_of_deliq` | Most recent delinquency level |

**Enquiry Features**:

| Feature | Description |
|---------|-------------|
| `tot_enq` | Total credit enquiries |
| `CC_enq` / `PL_enq` | CC / PL enquiries |
| `enq_L3m` / `enq_L6m` / `enq_L12m` | Enquiries in L3M / L6M / L12M |
| `time_since_recent_enq` | Months since last enquiry |

**Utilization Features**:

| Feature | Description |
|---------|-------------|
| `CC_utilization` | Credit card utilization % |
| `PL_utilization` | Personal loan utilization % |
| `pct_currentBal_all_TL` | % current balance across all TLs |
| `max_unsec_exposure_inPct` | Max unsecured exposure % |

**Demographic Features** (sensitive attributes for fairness):

| Feature | Values | Notes |
|---------|--------|-------|
| `GENDER` | M, F | 88% M, 12% F — imbalanced |
| `EDUCATION` | GRADUATE, 12TH, SSC, UNDER GRADUATE, OTHERS, POST-GRADUATE, PROFESSIONAL | |
| `MARITALSTATUS` | Married, Single | |
| `AGE` | 21–77, mean 33.7 | |
| `NETMONTHLYINCOME` | ₹0–₹25L, mean ₹26,424 | |
| `Time_With_Curr_Empr` | Months with current employer | |

**Product Flags**:

| Feature | Description |
|---------|-------------|
| `CC_Flag` | Has credit card (0/1) |
| `PL_Flag` | Has personal loan (0/1) |
| `HL_Flag` | Has home loan (0/1) |
| `GL_Flag` | Has gold loan (0/1) |
| `last_prod_enq2` | Last product enquired (CC, PL, AL, HL, ConsumerLoan, others) |
| `first_prod_enq2` | First product enquired |
| `Credit_Score` | CIBIL score (469–811, mean 680) |

### Target Variable

| Class | Label | Count | % | Meaning |
|-------|-------|-------|---|---------|
| P1 | 0 | 5,803 | 11.3% | Excellent creditworthiness — approve, best terms |
| P2 | 1 | 32,199 | 62.7% | Good creditworthiness — approve, standard terms |
| P3 | 2 | 7,452 | 14.5% | Marginal — conditional approval or higher rate |
| P4 | 3 | 5,882 | 11.5% | Poor creditworthiness — reject or require collateral |

---

## Model Architecture

### Logistic Regression (Baseline)
- `sklearn.linear_model.LogisticRegression`
- `multi_class='ovr'` (one-vs-rest for 4 classes)
- `solver='lbfgs'`, `max_iter=1000`
- `C=1.0` (default regularization)
- Role: interpretable baseline, coefficient-based feature importance

### Explainable Boosting Machine (Primary)
- `interpret.glassbox.ExplainableBoostingClassifier`
- Glass-box model: inherently interpretable, not post-hoc explained
- Learns pairwise feature interactions automatically
- Produces per-feature shape functions (not just importance scores)
- Natively supports multi-class classification
- Role: primary production model, highest accuracy + full interpretability

### Why EBM over XGBoost/LightGBM
EBM achieves near-gradient-boosting accuracy while being fully interpretable by design. For a regulated credit scoring context, this is the correct choice — the model's decisions can be audited without relying on post-hoc approximations (SHAP on a black-box model is an approximation; EBM's own explanations are exact).

---

## XAI Layer

### Global Explanations
- **EBM Global Importance**: exact feature contribution scores from the model itself
- **SHAP Summary Plot**: mean |SHAP value| per feature across all training samples
- Both should show identical top features (cross-validation of explanation quality)

### Local Explanations
- **SHAP Waterfall**: per-applicant feature contributions (additive, sums to prediction)
- **LIME**: local linear approximation around the applicant's feature space
- **EBM Local**: exact local explanation from EBM's additive structure

### Counterfactual (Phase 3)
- DiCE (Diverse Counterfactual Explanations) — "what would need to change for approval?"
- Not in Phase 2 scope

---

## Fairness Framework

### Metrics (fairlearn)
- **Demographic Parity Difference**: |P(Ŷ=approve|A) − P(Ŷ=approve|B)|
- **Equalized Odds Difference**: max(|TPR_A − TPR_B|, |FPR_A − FPR_B|)
- **Selection Rate by Group**: approval rate per demographic segment

### Sensitive Attributes
- Primary: `GENDER` (M vs F)
- Secondary: `EDUCATION` (7 levels, grouped to high/low for binary fairness metrics)
- Tertiary: `MARITALSTATUS` (Married vs Single)

### Thresholds (RBI-aligned)
- Demographic Parity Difference < 0.05: acceptable
- 0.05–0.10: monitor and document
- > 0.10: requires mitigation before deployment

---

## Agent Architecture (Page 7)

```
User Input (st.chat_input)
        ↓
Intent Classifier
  ├── "explain applicant {N}" → explain_applicant(N)
  ├── "compare {N} and {M}"  → compare_applicants(N, M)
  ├── "what drives approvals" → global_importance()
  └── "fairness for {attr}"  → fairness_summary(attr)
        ↓
Tool Execution (SHAP/LIME/MetricFrame)
        ↓
Context Builder (structured JSON → prompt)
        ↓
LLM Call (OpenAI gpt-4o-mini, ~200 tokens output)
        ↓
Streamed Response (st.write_stream)
```

### Fallback (no API key)
Structured text template filled with computed values — no LLM dependency for core functionality.

---

## Session State Schema

```python
st.session_state = {
    # Data
    "internal_df": pd.DataFrame,          # raw internal dataset
    "external_df": pd.DataFrame,          # raw external dataset
    "processed_df": pd.DataFrame,         # merged + encoded + scaled
    "unseen_df": pd.DataFrame,            # raw unseen dataset
    "feature_names": List[str],           # post-encoding column names
    "label_map": Dict[int, str],          # {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}
    "preprocessor": ColumnTransformer,    # fitted transformer (for unseen data)

    # Models
    "lr_model": LogisticRegression,
    "ebm_model": ExplainableBoostingClassifier,

    # Evaluation
    "model_metrics": {
        "lr": {"f1_macro": float, "auc_ovr": float},
        "ebm": {"f1_macro": float, "auc_ovr": float},
        "n_train": int,
        "n_test": int,
        "class_counts": Dict[str, int],
        "confusion_matrix_lr": np.ndarray,
        "confusion_matrix_ebm": np.ndarray,
    },

    # Scoring
    "prediction_result": pd.DataFrame,   # unseen + predictions + probabilities
    "predict_df": pd.DataFrame,          # unseen features only (post-transform)

    # Agent
    "chat_history": List[Dict],          # [{role, content}, ...]
    "shap_values_cache": Dict,           # {model_name: shap.Explanation}
}
```

---

## File Structure (Phase 2 Target)

```
AUCML_CreditScoring/
│
├── app/
│   ├── Home.py
│   ├── 1_Data_Upload.py
│   ├── 2_Preprocessing.py          ← TASK-01: add ColumnTransformer
│   ├── 3_Model_Training.py         ← TASK-02, TASK-03, TASK-04
│   ├── 4_Explainability.py         ← TASK-02: fix feature names
│   ├── 5_Score_New_Data.py         ← TASK-03: fix multi-class scoring
│   ├── 6_Fairness_Audit.py         ← TASK-05: NEW
│   ├── 7_Credit_Analyst_Agent.py   ← TASK-06: NEW
│   ├── 8_Business_Summary.py       ← TASK-07: NEW
│   │
│   └── components/
│       ├── utils.py                ← TASK-09: single source of truth
│       ├── model_loader.py
│       ├── shap_plotter.py
│       ├── lime_plotter.py
│       ├── pdp_plotter.py
│       ├── fairness_auditor.py     ← TASK-05: NEW
│       └── agent_tools.py          ← TASK-06: NEW
│
├── app/models/
│   ├── logistic_regression.pkl
│   └── ebm_model.pkl
│
├── reports/
│   ├── scored_output.csv
│   ├── applicant_XX_report.pdf
│   └── business_summary.pdf
│
├── Datasets/
│   ├── Internal_Bank_Dataset.xlsx
│   ├── External_Cibil_Dataset.xlsx
│   └── Unseen_Dataset.xlsx
│
├── Notebooks/
│   ├── Init_Project_Notebook.ipynb
│   └── AUCML_dashboard.ipynb
│
├── .kiro/steering/project-context.md
├── streamlit_app.py                ← TASK-08: fix nav
├── requirements.txt                ← TASK-10: update
├── TASKS.md
├── ARCHITECTURE.md
├── DECISIONS.md
└── README.md
```
