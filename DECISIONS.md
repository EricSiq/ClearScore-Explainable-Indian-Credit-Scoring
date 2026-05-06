# DECISIONS.md — Architecture Decision Records
## Explainable Indian Credit Scoring System

**Format**: ADR (Architecture Decision Record)  
**Last Updated**: 2026-05-01  

---

## ADR-001: EBM as Primary Model (not XGBoost/LightGBM)

**Status**: Accepted  
**Date**: 2026-05-01  

### Context
The system needs a high-accuracy classifier that is also explainable. The standard industry approach is to use a black-box model (XGBoost, LightGBM) and apply post-hoc SHAP explanations. An alternative is to use a glass-box model that is inherently interpretable.

### Decision
Use `ExplainableBoostingClassifier` (EBM) from InterpretML as the primary model, with Logistic Regression as a baseline.

### Rationale
- EBM achieves accuracy comparable to gradient boosting on tabular data (validated in multiple benchmarks including the original InterpretML paper, Nori et al. 2019)
- EBM's explanations are **exact**, not approximations. SHAP on XGBoost is a post-hoc approximation; EBM's own shape functions are the model itself
- For regulated credit scoring, exact explanations are preferable — a regulator can audit the model's decision function directly
- EBM natively supports multi-class classification (P1/P2/P3/P4)
- EBM learns pairwise feature interactions automatically, capturing non-linear credit risk patterns

### Trade-offs
- EBM training is slower than LightGBM (~3–5x on this dataset size)
- EBM is less widely known in industry than XGBoost — requires explanation in presentations
- SHAP integration with EBM requires `shap.Explainer` (TreeExplainer path), which works correctly

### Rejected Alternatives
- **XGBoost + SHAP**: Higher accuracy ceiling but black-box; SHAP is approximate
- **Decision Tree**: Fully interpretable but insufficient accuracy for 4-class credit scoring
- **Random Forest**: Better accuracy than DT but explanations are averaged, less precise

---

## ADR-002: Native Fairness Metrics over fairlearn/aif360

**Status**: Accepted  
**Date**: 2026-05-05 (supersedes 2026-05-01 draft)  

### Context
The design doc specifies `aif360` for bias detection. `aif360` is listed in `requirements.txt` but has never been successfully imported (the notebook shows `ModuleNotFoundError`). A fairness library must be chosen for the Phase 2 fairness audit page.

Two external libraries were evaluated:
- `aif360` (IBM): broken install on Python 3.14 / Windows — requires specific scikit-learn versions and C++ build tools
- `fairlearn` (Microsoft): requires `scipy`, which requires a Fortran compiler — build fails on this environment

### Decision
Implement all three fairness metrics from first principles using numpy/pandas. Remove both `aif360` and `fairlearn` from `requirements.txt`.

### Metrics implemented natively
```python
# Demographic Parity Difference
# max(P(Ŷ=approved | group)) − min(P(Ŷ=approved | group))
dpd = y_pred_bin.groupby(sensitive).mean()
dpd_value = dpd.max() - dpd.min()

# Equalized Odds Difference
# max(|ΔTPR|, |ΔFPR|) across groups
tpr_by_group = df[df.y_true==1].groupby('group')['y_pred'].mean()
fpr_by_group = df[df.y_true==0].groupby('group')['y_pred'].mean()
eod_value = max(tpr_by_group.max() - tpr_by_group.min(),
                fpr_by_group.max() - fpr_by_group.min())

# Selection Rate by group
sr = y_pred_bin.groupby(sensitive).mean()
```

### Rationale
- The three metrics are each 3–5 lines of pandas groupby — no library abstraction adds value
- Removing scipy/fairlearn eliminates a heavy build dependency with no functional loss
- The implementations are mathematically identical to fairlearn's `MetricFrame` output (verified against fairlearn docs)
- Zero new dependencies added for the entire fairness audit page

### Verified against real data (51K dataset, LR model)
| Attribute | DPD | EOD |
|-----------|-----|-----|
| GENDER | 0.0105 (✅ Acceptable) | 0.0500 (⚠️ Monitor) |
| EDUCATION | 0.0825 (⚠️ Monitor) | 0.1545 (🚨 Action Required) |
| MARITALSTATUS | 0.0921 (⚠️ Monitor) | 0.0330 (✅ Acceptable) |

### Trade-offs
- No access to fairlearn's `ThresholdOptimizer` for bias mitigation (Phase 3 concern, not Phase 2)
- If fairlearn becomes installable in a future environment, migration is trivial — the metric definitions are identical

### Rejected Alternatives
- **aif360**: Installation broken on Python 3.14 / Windows
- **fairlearn**: scipy build fails (missing Fortran compiler)
- **Both**: Unnecessary dependency weight for metrics implementable in ~30 lines

---

## ADR-003: 4-Class Ordinal Target (P1/P2/P3/P4) not Binary

**Status**: Accepted  
**Date**: 2026-05-01  

### Context
The current implementation treats `Approved_Flag` as binary (approve/reject). The actual target has 4 classes: P1, P2, P3, P4 — representing creditworthiness tiers, not a binary decision.

### Decision
Treat the problem as 4-class multi-class classification. Label encode: P1→0, P2→1, P3→2, P4→3.

### Rationale
- The 4-class framing is more informative and actionable for a credit officer
- P1 and P2 applicants can be offered different loan terms (P1 gets lower interest rate)
- P3 applicants can be offered conditional approval (higher rate, lower limit, collateral)
- P4 applicants are rejected — but the distinction from P3 matters for appeals
- Collapsing to binary (P1+P2 = approve, P3+P4 = reject) loses information and is a downstream business decision, not a modeling decision
- Both LR (with `multi_class='ovr'`) and EBM natively support 4-class classification

### Trade-offs
- 4-class evaluation is more complex (confusion matrix is 4×4, AUC is macro-averaged OvR)
- Class imbalance: P2 dominates (62.7%), P1 and P4 are minority classes
- May need class weighting (`class_weight='balanced'`) if P1/P4 recall is poor

### Rejected Alternatives
- **Binary (approve/reject)**: Loses tier information, less useful for credit officers
- **Regression on Credit_Score**: Credit_Score is a derived feature, not the target; using it as target would be circular

---

## ADR-004: llama.cpp CLI (subprocess) for Agent SLM — no Python bindings

**Status**: Accepted  
**Date**: 2026-05-05 (supersedes 2026-05-01 draft)  

### Context
The Credit Analyst Agent (Task-06) needs a language model to convert structured SHAP/LIME output into natural language explanations. The original design specified OpenAI `gpt-4o-mini`. The user requirement is a local SLM via llama.cpp to avoid API keys, internet dependency, and cost.

Two integration approaches were evaluated:
- `llama-cpp-python` (Python bindings): requires cmake + C++ compiler to build from source — both absent on this machine (Python 3.14 / Windows, no cmake, no MSVC in PATH)
- `llama-cli` binary via `subprocess`: standalone pre-built executable, no build step, works on any platform

### Decision
Call the `llama-cli` binary directly via `subprocess.run()`. No Python package dependency. The app falls back to deterministic template responses when the binary is not found.

### Recommended models (GGUF format, CPU-runnable)
| Model | Size | RAM needed | Quality |
|-------|------|-----------|---------|
| Phi-3-mini-4k-instruct Q4_K_M | 2.2 GB | ~3 GB | Best for credit explanations |
| Qwen2.5-1.5B-Instruct Q4_K_M | 1.0 GB | ~2 GB | Good, faster on CPU |
| Llama-3.2-1B-Instruct Q4_K_M | 0.7 GB | ~1.5 GB | Fastest, lower quality |

All three are instruction-tuned and follow the structured prompt format used in the agent.

### Integration pattern
```python
cmd = [
    "llama-cli",
    "--model",       model_path,
    "--prompt",      prompt,
    "--n-predict",   "300",
    "--temp",        "0.3",
    "--ctx-size",    "2048",
    "--threads",     str(os.cpu_count() // 2),
    "--no-display-prompt",
    "--log-disable",
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
```

### Fallback behaviour
When `llama-cli` is not in PATH or no model file is specified, the agent uses deterministic template responses filled with real SHAP numbers. The page is fully functional without any SLM — the SLM adds fluency, not correctness.

### Rationale
- Zero new Python dependencies — `subprocess` is stdlib
- Pre-built binaries available for Windows/Linux/Mac from the llama.cpp releases page
- CPU-only inference works on any machine with ~2 GB RAM
- Temperature 0.3 gives consistent, professional-sounding output without hallucination risk
- 2-minute timeout prevents the UI from hanging on slow hardware

### Trade-offs
- First inference is slow on CPU (~15–30 seconds for Phi-3-mini) — acceptable for a demo
- Non-deterministic output (mitigated by low temperature)
- User must manually download the binary and a GGUF model file

### Rejected Alternatives
- **OpenAI gpt-4o-mini**: Requires API key and internet — not suitable for offline demo
- **llama-cpp-python**: Build fails (no cmake/MSVC on this machine)
- **Ollama**: Requires a separate server process — adds operational complexity
- **Hardcoded templates only**: Works but loses the "agent" quality that differentiates the product

---

## ADR-005: Streamlit Session State as Persistence Layer

**Status**: Accepted for Phase 2, Revisit in Phase 3  
**Date**: 2026-05-01  

### Context
The app uses `st.session_state` to pass data between pages. This resets on browser refresh and doesn't support multiple concurrent users.

### Decision
Keep `st.session_state` for Phase 2. Document the limitation explicitly.

### Rationale
- For a demo/presentation context, session state is sufficient
- Adding a database (SQLite, Redis) would significantly increase setup complexity
- The hiring manager demo is single-user, single-session
- Phase 3 can introduce a proper persistence layer when multi-user or production deployment is needed

### Trade-offs
- Data lost on refresh — user must re-run the pipeline
- Not suitable for production multi-user deployment
- Cannot share sessions between users

### Phase 3 Migration Path
Replace `st.session_state` with:
- SQLite for model artifacts and processed data (via `sqlalchemy`)
- Redis for session caching (via `redis-py`)
- Or: move to a proper backend API (FastAPI) with Streamlit as frontend only

---

## ADR-006: ColumnTransformer with Named Output for Feature Preservation

**Status**: Accepted  
**Date**: 2026-05-01  

### Context
The preprocessing pipeline must preserve feature names through `ColumnTransformer` so that EBM, SHAP, and LIME all display real column names instead of `feature_0080` indices.

### Decision
Use `ColumnTransformer(verbose_feature_names_out=False)` and reconstruct a named DataFrame using `get_feature_names_out()` after fitting.

### Rationale
- `verbose_feature_names_out=False` produces clean names like `CC_utilization` instead of `num__CC_utilization`
- Reconstructing a named DataFrame before passing to EBM ensures `feature_names_in_` is set correctly
- SHAP's `Explainer` uses `feature_names_in_` automatically when the input is a DataFrame
- This is a one-time fix that resolves the `feature_0080` display issue across all pages

### Implementation
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

X_transformed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out().tolist()
X_processed = pd.DataFrame(X_transformed, columns=feature_names)
```
