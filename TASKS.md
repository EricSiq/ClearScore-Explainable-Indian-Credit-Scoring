# TASKS.md — Explainable Indian Credit Scoring System
## Phase 2: Production-Ready Upgrade

**Version**: 2.0  
**Status**: ✅ All tasks complete  
**Last Updated**: 2026-05-06  
**Owner**: Engineering  

---

## Context

The current system (v0.1-MVP) demonstrates the core pipeline: ingest → merge → preprocess → train → explain → score. Phase 2 upgrades it from a working prototype to a presentation-ready, production-credible system. The four upgrade pillars are:

1. **Fix EBM feature naming** — replace internal indices with real column names
2. **Add fairness audit** — demographic parity and equalized odds across GENDER and EDUCATION
3. **Add Credit Analyst Agent interface** — LLM-backed natural language explanation layer
4. **Add business impact summary** — translate model metrics into ₹ NPA exposure framing

Each task below is atomic, independently testable, and ordered by dependency.

---

## TASK-01 — Fix Categorical Encoding in Preprocessing

**Priority**: P0 (blocker for all downstream tasks)  
**Effort**: ~2 hours  
**File**: `app/2_Preprocessing.py`  
**Status**: [x] Complete

### Problem
The preprocessing page applies `SimpleImputer` and `StandardScaler` but has no `OneHotEncoder` for categorical columns. The merged dataset has 5 categorical columns: `MARITALSTATUS`, `EDUCATION`, `GENDER`, `last_prod_enq2`, `first_prod_enq2`. Passing these raw strings to sklearn or EBM causes a `ValueError` at training time.

### Acceptance Criteria
- [x] `ColumnTransformer` pipeline with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical cols
- [x] `SimpleImputer(strategy='median')` + `StandardScaler()` for numeric cols
- [x] Encoded feature names preserved and stored in `st.session_state["feature_names"]`
- [x] `processed_df` in session_state is a clean numeric DataFrame with named columns
- [x] `PROSPECTID` dropped before training (identifier, not a feature)
- [x] Target column `Approved_Flag` preserved as-is (P1/P2/P3/P4 strings → label encoded 0/1/2/3)

### Verified Output (dry-run against real datasets)
- Merged shape: 51,336 × 87 → X_processed: 51,336 × 103 (80 numeric + 23 OHE-expanded categorical)
- Feature names: real column names throughout (`Total_TL`, `GENDER_M`, `EDUCATION_GRADUATE`, etc.)
- Zero NaN in X_processed or y
- y distribution: {P1: 5,803 | P2: 32,199 | P3: 7,452 | P4: 5,882}

### Implementation Notes
```python
# Categorical cols in merged dataset
CAT_COLS = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
# Drop identifier
DROP_COLS = ['PROSPECTID']
# Target
TARGET_COL = 'Approved_Flag'
# Target encoding: P1=0, P2=1, P3=2, P4=3 (ordinal, worst to best creditworthiness)
TARGET_MAP = {'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3}
```

Use `ColumnTransformer(remainder='passthrough')` and reconstruct a named DataFrame from the output using `get_feature_names_out()`.

---

## TASK-02 — Fix EBM Feature Name Display

**Priority**: P0 (visual correctness, hiring manager will see this)  
**Effort**: ~30 minutes  
**File**: `app/3_Model_Training.py`, `app/4_Explainability.py`  
**Status**: [x] Complete

### Problem
EBM internally re-indexes features after `ColumnTransformer` preprocessing. Global importance charts show `feature_0080`, `feature_0059` etc. instead of real names like `Tot_Missed_Pmnt`, `CC_utilization`, `num_times_delinquent`.

### Root Cause
`ExplainableBoostingClassifier` receives a numpy array from the pipeline, losing column names. The fix is to pass a named DataFrame (not array) to EBM, or to explicitly set `feature_names` on the EBM after fitting.

### Acceptance Criteria
- [x] EBM trained on a named pandas DataFrame (not raw numpy array)
- [x] `ebm_model.feature_names_in_` matches actual column names post-encoding
- [x] Global importance chart in Page 4 shows real feature names
- [x] SHAP summary plot shows real feature names
- [x] Feature names stored in `st.session_state["feature_names"]` for use across pages

### Verified Output (dry-run on 2,000-row sample)
- `ebm.feature_names_in_` set: ✅
- First 5: `['Total_TL', 'Tot_Closed_TL', 'Tot_Active_TL', 'Total_TL_opened_L6M', 'Tot_TL_closed_L6M']`
- Last 5: `['first_prod_enq2_CC', 'first_prod_enq2_ConsumerLoan', 'first_prod_enq2_HL', 'first_prod_enq2_PL', 'first_prod_enq2_others']`
- Generic `feature_XXXX` indices: **0** (was the entire feature list before this fix)

### Implementation Notes
After `ColumnTransformer.fit_transform()`, reconstruct DataFrame:
```python
feature_names = preprocessor.get_feature_names_out()
X_processed = pd.DataFrame(X_transformed, columns=feature_names)
# Then train EBM on X_processed directly
ebm_model = ExplainableBoostingClassifier(feature_names=feature_names.tolist())
ebm_model.fit(X_processed, y)
```

---

## TASK-03 — Fix Multi-Class Target Handling

**Priority**: P0 (model correctness)  
**Effort**: ~1 hour  
**File**: `app/3_Model_Training.py`, `app/5_Score_New_Data.py`  
**Status**: [x] Complete

### Problem
`Approved_Flag` is a 4-class ordinal target (P1, P2, P3, P4), not binary. Current code treats it as binary (threshold at 0.5 on `predict_proba[:, 1]`). This produces incorrect predictions and misleading probabilities.

### Acceptance Criteria
- [x] Target label-encoded: `{'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3}` (done in Page 2)
- [x] `LogisticRegression(solver='lbfgs', max_iter=1000)` — `multi_class` param removed (deprecated/removed in sklearn 1.5+; lbfgs uses multinomial automatically)
- [x] `ExplainableBoostingClassifier` used as-is (natively supports multi-class)
- [x] Scoring page shows predicted class label (P1/P2/P3/P4) and per-class probabilities
- [x] Reverse mapping applied to display: `{0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}`
- [x] Class label mapping stored in `st.session_state["label_map"]`

### Additional fix: unseen data transform
The unseen dataset (100 × 42) is a subset of the 85-column training feature space. Passing it raw to `preprocessor.transform()` raised `ValueError: columns are missing`. Fixed by reindexing the unseen DataFrame to the full training column set before transforming — missing numeric columns filled with 0, missing categorical columns filled with the fitted imputer's `statistics_` (most frequent value).

### Verified Output
- Unseen transformed shape: 100 × 103 ✅
- `predict_proba` shape: 100 × 4 ✅  
- Class order: `['P1', 'P2', 'P3', 'P4']` ✅
- No binary `[:, 1]` or `>= 0.5` logic anywhere ✅

### Business Context
P1 = highest creditworthiness (approve, best terms)  
P2 = good creditworthiness (approve, standard terms)  
P3 = marginal (approve with conditions or higher rate)  
P4 = poor creditworthiness (reject or require collateral)

---

## TASK-04 — Add Model Evaluation Metrics Page

**Priority**: P1 (required for credibility)  
**Effort**: ~2 hours  
**File**: `app/3_Model_Training.py`  
**Status**: [x] Complete

### Acceptance Criteria
- [x] After training, display classification report (precision, recall, F1 per class P1–P4)
- [x] Confusion matrix heatmap (seaborn, annotated with count + row %)
- [x] Macro-averaged F1 and weighted F1 displayed as headline metrics
- [x] ROC-AUC (one-vs-rest, macro-averaged) for both LR and EBM
- [x] Train vs test split sizes shown (80/20, stratified)
- [x] Class distribution in train/test shown to confirm stratification
- [x] Metrics stored in `st.session_state["model_metrics"]` for use in business summary

### Verified Output (5,000-row sample, LR)
| Metric | Value |
|--------|-------|
| Macro F1 | 0.924 |
| Weighted F1 | 0.939 |
| AUC (OvR, macro) | 0.980 |
| P1 AUC | 0.999 |
| P2 AUC | 0.993 |
| P3 AUC | 0.929 |
| P4 AUC | 0.999 |

Stratification: max class-proportion diff between train/test = **0.03%** (effectively zero).

### Business Impact Framing (for Task-07)
Store alongside metrics:
```python
st.session_state["model_metrics"] = {
    "lr": {"f1_macro": ..., "auc_ovr": ..., "n_test": ...},
    "ebm": {"f1_macro": ..., "auc_ovr": ..., "n_test": ...},
    "n_train": ...,
    "n_test": ...,
    "class_counts": {...}
}
```

---

## TASK-05 — Add Fairness Audit Page

**Priority**: P1 (key differentiator, RBI responsible AI alignment)  
**Effort**: ~3 hours  
**File**: `app/6_Fairness_Audit.py` (new page)  
**Status**: [x] Complete

### Acceptance Criteria
- [x] New page `app/6_Fairness_Audit.py` added to sidebar
- [x] Sensitive attributes selectable: GENDER, EDUCATION, MARITALSTATUS
- [x] Fairness metrics computed (native numpy/pandas — no fairlearn dependency)
- [x] Heatmap: approval rate (selection rate) by demographic group × predicted class
- [x] Bar chart: demographic parity difference per group
- [x] Equalized odds difference displayed with traffic-light colouring (green/amber/red)
- [x] Narrative summary with real numbers and recommended actions
- [x] `aif360` removed from `requirements.txt`
- [x] `requirements.txt` pinned with exact versions

### Note on fairlearn
`fairlearn` requires `scipy` which requires a Fortran compiler — build fails on this Windows environment (Python 3.14). All three fairness metrics are implemented from first principles in ~30 lines of pandas/numpy. The math is identical to fairlearn's implementation. This is documented in `DECISIONS.md` ADR-002.

### Verified Findings (LR model, full 51K dataset)
| Attribute | DPD | Status | EOD | Status |
|-----------|-----|--------|-----|--------|
| GENDER | 0.0105 | ✅ Acceptable | 0.0500 | ⚠️ Monitor |
| EDUCATION | 0.0825 | ⚠️ Monitor | 0.1545 | 🚨 Action Required |
| MARITALSTATUS | 0.0921 | ⚠️ Monitor | 0.0330 | ✅ Acceptable |

Key finding: **EDUCATION** has an EOD of 0.154 — the model's error rates differ significantly across education levels. This is a genuine regulatory concern and a strong talking point.

### Regulatory Context
RBI's "Guidelines on Model Risk Management" (2023) and the proposed Digital India Act both emphasize non-discrimination in automated credit decisions. This page directly addresses that requirement and is a strong talking point in any fintech hiring conversation.

---

## TASK-06 — Add Credit Analyst Agent Interface

**Priority**: P1 (product differentiation, mirrors Acuity-style agent architecture)  
**Effort**: ~4 hours  
**File**: `app/7_Credit_Analyst_Agent.py` (new page)  
**Status**: [x] Complete

### Acceptance Criteria
- [x] Chat-style UI using `st.chat_input` and `st.chat_message`
- [x] Conversation history maintained in `st.session_state["chat_history"]`
- [x] Intent detection for 5 query types:
  - Explain individual applicant (SHAP → NL)
  - Compare two applicants (SHAP delta → NL)
  - Global feature importance (SHAP summary → NL)
  - Fairness query (native metrics → NL)
  - Model performance (metrics dict → NL)
- [x] SLM integration via llama.cpp CLI binary (`subprocess.run`) — no Python build step
- [x] Model path configured via sidebar text input
- [x] Graceful fallback: structured template responses with real SHAP numbers when binary/model absent
- [x] SHAP values computed once and cached in `session_state["agent_shap_{model_key}"]`
- [x] Intent classifier: 14/14 test cases pass

### SLM Architecture (llama.cpp subprocess, not Python bindings)
`llama-cpp-python` requires cmake + C++ compiler — both absent on this machine.
The agent calls `llama-cli` directly via `subprocess.run()`. Zero new Python dependencies.

**Recommended models (download one GGUF file):**
| Model | Size | Notes |
|-------|------|-------|
| Phi-3-mini-4k-instruct Q4_K_M | 2.2 GB | Best quality for credit explanations |
| Qwen2.5-1.5B-Instruct Q4_K_M | 1.0 GB | Good balance of speed/quality |
| Llama-3.2-1B-Instruct Q4_K_M | 0.7 GB | Fastest on CPU |

**Binary install:** Download pre-built `llama-cli` from https://github.com/ggerganov/llama.cpp/releases

### Fallback behaviour
The page is fully functional without any SLM. Template responses use real SHAP values, real predictions, and real fairness metrics — they're structured and accurate, just not fluent prose. The SLM adds natural language quality on top of correct data.

### Intent patterns verified (14/14)
```
explain applicant 42          → explain_applicant  {idx: 42}
why was applicant 15 rejected → explain_applicant  {idx: 15}
compare applicants 5 and 12   → compare_applicants {idx_a: 5, idx_b: 12}
compare applicant 3 vs 7      → compare_applicants {idx_a: 3, idx_b: 7}
what features drive approvals → global_importance  {}
which factors matter most     → global_importance  {}
global shap importance        → global_importance  {}
fairness for GENDER           → fairness           {attr: GENDER}
show bias for education       → fairness           {attr: EDUCATION}
how accurate is the model     → model_performance  {}
what is the f1 score          → model_performance  {}
help                          → help               {}
what can you do               → help               {}
random gibberish xyz          → unknown            {}
```

---

## TASK-07 — Add Business Impact Summary

**Priority**: P1 (client-facing framing)  
**Effort**: ~2 hours  
**File**: `app/8_Business_Summary.py` (new page)  
**Status**: [x] Complete

### Acceptance Criteria
- [x] Business summary displayed as a dedicated page (Page 8)
- [x] Metrics translated to business language with real numbers:
  - "9,848 of 10,268 applicants correctly classified (95.9% accuracy)"
  - "Approved (P1+P2): 7,842 (76.4% approval rate)"
  - "High-risk flagged (P4): 1,182 applicants"
  - "NPA exposure avoided: ₹12.92 Cr (P4 × 40% + P3 × 15% × ₹2L avg loan)"
- [x] Configurable assumption sliders (avg loan, P4 default rate, P3 default rate)
- [x] Comparison table: EBM vs Logistic Regression on all business metrics
- [x] Risk tier breakdown: donut charts (Approved / Marginal / Rejected)
- [x] NPA impact grouped bar chart (avoided vs residual exposure)
- [x] Correct classifications per tier bar chart
- [x] Downloadable one-page PDF business summary via ReportLab
- [x] Page added to sidebar navigation

### Verified Output (LR model, full 51K dataset, default assumptions)
| Business Metric | Value |
|----------------|-------|
| Test set size | 10,268 applicants |
| Correctly classified | 9,848 (95.9%) |
| Approved (P1+P2) | 7,842 (76.4%) |
| Marginal (P3) | 1,244 (12.1%) |
| Rejected (P4) | 1,182 (11.5%) |
| NPA avoided | ₹12.92 Cr |
| Residual NPA exposure | ₹0.00 Cr (0 P4 incorrectly approved) |

### NPA Calculation Logic
```python
# Correctly rejected P4 applicants (true positives for rejection)
tp_p4 = confusion_matrix[3, 3]
npa_p4 = tp_p4 * avg_loan * p4_default_rate

# Correctly flagged P3 applicants (conditional approval avoids partial NPA)
tp_p3 = confusion_matrix[2, 2]
npa_p3 = tp_p3 * avg_loan * p3_default_rate

npa_total = npa_p4 + npa_p3

# Residual exposure: P4 incorrectly approved as P1 or P2
fn_p4 = confusion_matrix[3, 0] + confusion_matrix[3, 1]
npa_exposure = fn_p4 * avg_loan * p4_default_rate
```

---

## TASK-08 — Fix Streamlit Navigation and File References

**Priority**: P0 (app won't run correctly without this)  
**Effort**: ~15 minutes  
**File**: `streamlit_app.py`  
**Status**: [x] Complete (done as part of TASK-05)

### Acceptance Criteria
- [x] `streamlit_app.py` updated with correct filenames (`1_Data_Upload.py` not `01_Data_Upload.py`)
- [x] Page 6 (Fairness Audit) added to sidebar navigation
- [x] Page 7 (Credit Analyst Agent) added to sidebar navigation
- [x] Page 8 (Business Summary) added to sidebar navigation
- [x] Sidebar page order: Home → Upload → Preprocess → Train → Explain → Score → Fairness → Agent → Business Summary

---

## TASK-09 — Wire Components Folder

**Priority**: P2 (code quality, maintainability)  
**Effort**: ~1 hour  
**Files**: All `app/*.py` pages  
**Status**: [x] Complete

### Acceptance Criteria
- [x] All pages import from `app/components/` instead of duplicating logic
- [x] `utils.py` is the single source of truth for `get_label_map`, `get_X_y`, `get_processed_df`, `get_unseen_df`
- [x] `model_loader.py` is the single source of truth for `load_model`
- [x] `shap_plotter.py` used by Pages 4, 5, and 7 (`get_shap_values`, `shap_top_features`)
- [x] `lime_plotter.py` used by Pages 4 and 5 (`lime_local_explanation`)
- [x] `pdp_plotter.py` used by Pages 4 and 5 (`plot_pdp`)
- [x] No duplicate function definitions across pages (7/7 verified)
- [x] `app/components/__init__.py` added so the package imports cleanly

### What was extracted vs kept inline

| Function | Extracted to | Pages that import it |
|----------|-------------|---------------------|
| `load_model` | `model_loader.py` | 4, 5, 7 |
| `get_label_map` | `utils.py` | 4, 5, 6, 7, 8 |
| `get_X_y` | `utils.py` | 4, 7 |
| `get_processed_df` | `utils.py` | (available, used via get_X_y) |
| `get_shap_values` | `shap_plotter.py` | 4, 5, 7 |
| `shap_top_features` | `shap_plotter.py` | 5, 7 |
| `lime_local_explanation` | `lime_plotter.py` | 4, 5 |
| `plot_pdp` | `pdp_plotter.py` | 4, 5 |

Page-specific functions (`_transform_unseen`, `_run_prediction`, `_save_pdf_report`, `_compute_business_metrics`, `_classify_intent`, `_dispatch`, fairness metric functions) remain inline — they are not shared across pages and extracting them would add indirection without reducing duplication.

---

## TASK-10 — Update requirements.txt

**Priority**: P0 (environment correctness)  
**Effort**: ~10 minutes  
**File**: `requirements.txt`  
**Status**: [x] Complete (revised — versions corrected to match actual installed environment)

### Problem with previous version
The requirements.txt had pinned versions that were months-old guesses and did not match what was actually installed. `lime` was not installed at all. `openai` was pinned to 1.30.0 but 2.31.0 is installed. `scikit-learn` was pinned to 1.5.0 but 1.8.0 is installed (and 1.5.0 would have broken — `multi_class` param was removed in 1.5).

### Final state — verified against actual installed environment

| Package | Pinned version | Role |
|---------|---------------|------|
| streamlit | 1.56.0 | Dashboard framework |
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.3.5 | Numerical operations |
| openpyxl | 3.1.5 | Excel file reading |
| scikit-learn | 1.8.0 | ML pipeline, LR, metrics |
| joblib | 1.5.3 | Model serialisation |
| interpret | 0.7.8 | EBM (ExplainableBoostingClassifier) |
| shap | 0.51.0 | SHAP explainability |
| lime | 0.2.0.1 | LIME local explanations |
| matplotlib | 3.10.8 | Visualisation |
| seaborn | 0.13.2 | Heatmaps, confusion matrices |
| reportlab | 4.4.10 | PDF generation |

### Intentionally omitted (with reasons)
- `openai` — commented out; agent works in template mode without it
- `fairlearn` — scipy build fails on Python 3.14/Windows; metrics implemented natively
- `aif360` — broken install; replaced by native implementation
- `llama-cpp-python` — requires cmake/C++; agent uses llama-cli binary via subprocess

### Verified
All 12 packages confirmed importable in the current environment.

---

## Dependency Graph

```
TASK-10 (requirements)
    ↓
TASK-01 (categorical encoding)  ← TASK-08 (nav fix, parallel)
    ↓
TASK-02 (EBM feature names)
    ↓
TASK-03 (multi-class target)
    ↓
TASK-04 (evaluation metrics)
    ↓
TASK-05 (fairness audit)    TASK-06 (agent)    TASK-07 (business summary)
    ↓                           ↓                   ↓
TASK-09 (wire components — can run in parallel after TASK-04)
```

---

## Definition of Done (per task)

A task is complete when:
1. Code changes are implemented and saved
2. The Streamlit page renders without error
3. The specific acceptance criteria above are all checked
4. No regressions in upstream pages (run full pipeline: Upload → Preprocess → Train → Explain → Score)

---

## Out of Scope (Phase 2)

The following are noted for Phase 3 and should not be implemented now:
- Real-time API data ingestion (RBI, NPCI, CPCB)
- Database persistence layer (replace session_state with SQLite or Redis)
- User authentication and role-based access
- Counterfactual explanation engine (DiCE)
- Model retraining feedback loop
- Deployment to Streamlit Cloud / AWS
