import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from app.components.model_loader import load_model as _load_model_from_disk
from app.components.utils        import get_label_map, get_X_y
from app.components.shap_plotter import get_shap_values, shap_top_features

TARGET_COL = "Approved_Flag"
MODEL_DIR  = "app/models"
_LABEL_MAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}

TIER_DESCRIPTIONS = {
    "P1": "Excellent creditworthiness — approve, best terms",
    "P2": "Good creditworthiness — approve, standard terms",
    "P3": "Marginal — conditional approval or higher rate",
    "P4": "Poor creditworthiness — reject or secured product only",
}

_LLAMA_BIN_CANDIDATES = ["llama-cli", "llama-cli.exe", "./llama-cli", "./llama-cli.exe"]

RECOMMENDED_MODELS = {
    "Phi-3-mini-4k-instruct Q4_K_M (~2.2 GB)":
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
    "Qwen2.5-1.5B-Instruct Q4_K_M (~1.0 GB)":
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Llama-3.2-1B-Instruct Q4_K_M (~0.7 GB)":
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
}

_INTENT_PATTERNS = [
    (r"(?:explain|why|reason|tell me about).*?applicant\s+(\d+)",
     "explain_applicant", lambda m: {"idx": int(m.group(1))}),
    (r"compare\s+applicants?\s+(\d+)\s+(?:and|vs\.?|with)\s+(\d+)",
     "compare_applicants", lambda m: {"idx_a": int(m.group(1)), "idx_b": int(m.group(2))}),
    (r"(?:what|which)\s+(?:features?|factors?)\s+(?:drive|matter|important|affect)",
     "global_importance", lambda m: {}),
    (r"(?:global|overall)\s+(?:importance|shap|features?)",
     "global_importance", lambda m: {}),
    (r"(?:fairness|bias|fair|discriminat)\s+(?:for\s+)?(\w+)?",
     "fairness", lambda m: {"attr": m.group(1) or "GENDER"}),
    (r"(?:how\s+(?:good|accurate|well)|performance|metrics?|f1|auc|accuracy)",
     "model_performance", lambda m: {}),
    (r"(?:help|what can you do|commands?|examples?)",
     "help", lambda m: {}),
]


def _label_map() -> dict:
    return get_label_map()


def _load_model(filename: str, session_key: str):
    return _load_model_from_disk(filename, session_key)


def _get_X_y():
    df = st.session_state.get("processed_df")
    if df is None:
        return None, None
    feature_names = st.session_state.get("feature_names", [c for c in df.columns if c != TARGET_COL])
    return df[feature_names], df.get(TARGET_COL)


def _get_shap_values(model, X: pd.DataFrame, model_key: str):
    return get_shap_values(model, X, f"agent_{model_key}")


def _top_shap_features(shap_values, idx: int, n: int = 5):
    return shap_top_features(shap_values, idx, n)


def _shap_delta(shap_values, idx_a: int, idx_b: int, n: int = 5):
    sv_a = shap_values[idx_a]
    sv_b = shap_values[idx_b]
    vals_a = sv_a.values.mean(axis=1) if sv_a.values.ndim == 2 else sv_a.values
    vals_b = sv_b.values.mean(axis=1) if sv_b.values.ndim == 2 else sv_b.values
    delta  = vals_b - vals_a
    top_idx = np.argsort(np.abs(delta))[::-1][:n]
    return [(sv_a.feature_names[i], float(vals_a[i]), float(vals_b[i])) for i in top_idx]


def _find_llama_binary():
    for candidate in _LLAMA_BIN_CANDIDATES:
        if shutil.which(candidate):
            return candidate
        if Path(candidate).exists():
            return candidate
    return None


def _run_llama(prompt: str, model_path: str, n_predict: int = 300, temperature: float = 0.3) -> str:
    binary = _find_llama_binary()
    if binary is None:
        return ""
    cmd = [binary, "--model", model_path, "--prompt", prompt,
           "--n-predict", str(n_predict), "--temp", str(temperature),
           "--ctx-size", "2048", "--threads", str(max(1, os.cpu_count() // 2)),
           "--no-display-prompt", "--log-disable"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        if prompt.strip() in output:
            output = output.replace(prompt.strip(), "").strip()
        return output
    except subprocess.TimeoutExpired:
        return "[Generation timed out. Try a smaller model.]"
    except Exception as e:
        return f"[llama.cpp error: {e}]"


def _build_explain_prompt(idx, pred_class, confidence, top_features, label_map):
    feat_lines = "\n".join(
        f"  {i+1}. {name}: {val:+.4f} ({'increases' if val > 0 else 'decreases'} approval chance)"
        for i, (name, val) in enumerate(top_features)
    )
    return textwrap.dedent(f"""
        You are a credit analyst AI for an Indian bank. Explain the following credit decision
        in plain English. Be concise (3-4 sentences). End with one actionable recommendation.

        Applicant index: {idx}
        Predicted credit tier: {pred_class} — {TIER_DESCRIPTIONS.get(pred_class, '')}
        Confidence: {confidence:.1%}

        Top SHAP feature contributions (positive = supports approval):
        {feat_lines}

        Format your response as:
        DECISION: [one sentence]
        REASONS: [2-3 bullet points]
        RECOMMENDATION: [one actionable sentence]
    """).strip()


def _build_compare_prompt(idx_a, pred_a, conf_a, idx_b, pred_b, conf_b, delta_features):
    delta_lines = "\n".join(
        f"  {name}: applicant {idx_a}={va:+.4f}, applicant {idx_b}={vb:+.4f} (delta={vb-va:+.4f})"
        for name, va, vb in delta_features
    )
    return textwrap.dedent(f"""
        You are a credit analyst AI. Compare two loan applicants concisely (4-5 sentences).

        Applicant {idx_a}: tier {pred_a} ({conf_a:.1%} confidence)
        Applicant {idx_b}: tier {pred_b} ({conf_b:.1%} confidence)

        Largest SHAP differences:
        {delta_lines}

        Explain why applicant {idx_b} received a {'better' if pred_b < pred_a else 'worse'} tier.
    """).strip()


def _build_global_prompt(top_global):
    feat_lines = "\n".join(f"  {i+1}. {name} (mean |SHAP| = {val:.4f})" for i, (name, val) in enumerate(top_global))
    return textwrap.dedent(f"""
        You are a credit analyst AI. Explain what drives credit approval decisions
        in 3-4 plain English sentences for a non-technical audience.

        Top 10 features by mean absolute SHAP value:
        {feat_lines}
    """).strip()


def _build_fairness_prompt(attr, dpd, eod, sr_dict):
    sr_lines = "\n".join(f"  {g}: {r:.1%}" for g, r in sorted(sr_dict.items(), key=lambda x: -x[1]))
    return textwrap.dedent(f"""
        You are a credit analyst AI explaining fairness metrics to a compliance officer.
        Be concise (3-4 sentences).

        Attribute: {attr}
        Demographic Parity Difference: {dpd:.4f}
        Equalized Odds Difference: {eod:.4f}
        Approval rates: {sr_lines}
        Thresholds: <0.05 acceptable, 0.05-0.10 monitor, >0.10 action required.
    """).strip()


def _build_performance_prompt(metrics):
    lr  = metrics.get("lr", {})
    ebm = metrics.get("ebm", {})
    return textwrap.dedent(f"""
        You are a credit analyst AI explaining model performance to a bank manager.
        Write 3-4 plain English sentences.

        Test applicants: {metrics.get('n_test', 'N/A'):,}
        Logistic Regression: Macro F1={lr.get('f1_macro',0):.3f}, AUC={lr.get('auc_ovr',0):.3f}
        EBM: Macro F1={ebm.get('f1_macro',0):.3f}, AUC={ebm.get('auc_ovr',0):.3f}
    """).strip()


def _fallback_explain(idx, pred_class, confidence, top_features):
    tier_desc = TIER_DESCRIPTIONS.get(pred_class, "")
    top3 = top_features[:3]
    reasons = "\n".join(
        f"  - **{name}**: SHAP {val:+.4f} ({'positive' if val > 0 else 'negative'} impact)"
        for name, val in top3
    )
    rec = top3[0][0] if top3 else "credit utilization"
    return (f"**DECISION:** Applicant {idx} classified as **{pred_class}** ({tier_desc}) "
            f"with {confidence:.1%} confidence.\n\n**REASONS:**\n{reasons}\n\n"
            f"**RECOMMENDATION:** Focus on improving `{rec}`.")


def _fallback_compare(idx_a, pred_a, conf_a, idx_b, pred_b, conf_b, delta_features):
    lines = "\n".join(f"  - **{n}**: {idx_a}={va:+.4f}, {idx_b}={vb:+.4f}" for n, va, vb in delta_features[:3])
    return (f"**Applicant {idx_a}**: {pred_a} ({conf_a:.1%})\n"
            f"**Applicant {idx_b}**: {pred_b} ({conf_b:.1%})\n\n**Key differences:**\n{lines}")


def _fallback_global(top_global):
    lines = "\n".join(f"  {i+1}. **{n}** (mean |SHAP| = {v:.4f})" for i, (n, v) in enumerate(top_global[:10]))
    return f"**Top features driving credit decisions:**\n{lines}"


def _fallback_fairness(attr, dpd, eod, sr_dict):
    lines = "\n".join(f"  - **{g}**: {r:.1%}" for g, r in sorted(sr_dict.items(), key=lambda x: -x[1]))
    dpd_s = "Acceptable" if dpd < 0.05 else ("Monitor" if dpd < 0.10 else "Action Required")
    eod_s = "Acceptable" if eod < 0.05 else ("Monitor" if eod < 0.10 else "Action Required")
    return (f"**Fairness — {attr}:**\n\n{lines}\n\n"
            f"**DPD:** {dpd:.4f} — {dpd_s}\n**EOD:** {eod:.4f} — {eod_s}")


def _fallback_performance(metrics):
    lr  = metrics.get("lr", {})
    ebm = metrics.get("ebm", {})
    return (f"**Model performance on {metrics.get('n_test','N/A'):,} test applicants:**\n\n"
            f"| Model | Macro F1 | AUC (OvR) |\n|-------|----------|----------|\n"
            f"| Logistic Regression | {lr.get('f1_macro',0):.3f} | {lr.get('auc_ovr',0):.3f} |\n"
            f"| EBM | {ebm.get('f1_macro',0):.3f} | {ebm.get('auc_ovr',0):.3f} |")


def _help_text():
    return textwrap.dedent("""
        **Example queries:**
        - `explain applicant 42`
        - `compare applicants 5 and 12`
        - `what features drive approvals?`
        - `fairness for GENDER`
        - `how accurate is the model?`
    """).strip()


def _classify_intent(query: str):
    q = query.lower().strip()
    for pattern, intent, extractor in _INTENT_PATTERNS:
        m = re.search(pattern, q)
        if m:
            return intent, extractor(m)
    return "unknown", {}


def _dispatch(intent, args, model, model_key, X, y, label_map, model_path, metrics, external_df, X_test, y_test):
    use_slm = (model_path is not None) and Path(model_path).exists() and (_find_llama_binary() is not None)

    if intent == "explain_applicant":
        idx = args.get("idx", 0)
        if idx >= len(X):
            return f"Applicant index {idx} is out of range (dataset has {len(X)} rows)."
        shap_values  = _get_shap_values(model, X, model_key)
        top_features = _top_shap_features(shap_values, idx, n=5)
        y_pred_int   = int(model.predict(X.iloc[[idx]])[0])
        pred_class   = label_map.get(y_pred_int, str(y_pred_int))
        proba        = model.predict_proba(X.iloc[[idx]])[0]
        confidence   = float(proba[list(model.classes_).index(y_pred_int)])
        if use_slm:
            r = _run_llama(_build_explain_prompt(idx, pred_class, confidence, top_features, label_map), model_path)
            if r:
                return r
        return _fallback_explain(idx, pred_class, confidence, top_features)

    elif intent == "compare_applicants":
        idx_a, idx_b = args.get("idx_a", 0), args.get("idx_b", 1)
        n = len(X)
        if idx_a >= n or idx_b >= n:
            return f"Index out of range. Dataset has {n} rows (0-{n-1})."
        shap_values = _get_shap_values(model, X, model_key)
        delta = _shap_delta(shap_values, idx_a, idx_b, n=5)
        def _pred_info(i):
            yi = int(model.predict(X.iloc[[i]])[0])
            p  = model.predict_proba(X.iloc[[i]])[0]
            return label_map.get(yi, str(yi)), float(p[list(model.classes_).index(yi)])
        pred_a, conf_a = _pred_info(idx_a)
        pred_b, conf_b = _pred_info(idx_b)
        if use_slm:
            r = _run_llama(_build_compare_prompt(idx_a, pred_a, conf_a, idx_b, pred_b, conf_b, delta), model_path)
            if r:
                return r
        return _fallback_compare(idx_a, pred_a, conf_a, idx_b, pred_b, conf_b, delta)

    elif intent == "global_importance":
        shap_values = _get_shap_values(model, X, model_key)
        sv_vals = shap_values.values
        importance = np.abs(sv_vals).mean(axis=(0, 2)) if sv_vals.ndim == 3 else np.abs(sv_vals).mean(axis=0)
        top_idx    = np.argsort(importance)[::-1][:10]
        top_global = [(shap_values.feature_names[i], float(importance[i])) for i in top_idx]
        if use_slm:
            r = _run_llama(_build_global_prompt(top_global), model_path)
            if r:
                return r
        return _fallback_global(top_global)

    elif intent == "fairness":
        attr = args.get("attr", "GENDER").upper()
        if external_df is None or X_test is None or y_test is None:
            return "Fairness analysis requires the full pipeline (Upload -> Preprocess -> Train)."
        if attr not in external_df.columns:
            available = [c for c in ["GENDER", "EDUCATION", "MARITALSTATUS"] if c in external_df.columns]
            return f"Attribute `{attr}` not found. Available: {available}"
        try:
            sensitive = external_df.loc[X_test.index, attr].reset_index(drop=True)
        except KeyError:
            sensitive = external_df[attr].iloc[:len(X_test)].reset_index(drop=True)
        y_pred_raw = model.predict(X_test)
        y_pred_bin = pd.Series(y_pred_raw).isin({0, 1}).astype(int)
        y_true_bin = y_test.reset_index(drop=True).isin({0, 1}).astype(int)
        sr  = y_pred_bin.groupby(sensitive.values).mean()
        dpd = float(sr.max() - sr.min())
        df_tmp = pd.DataFrame({"y_true": y_true_bin.values, "y_pred": y_pred_bin.values, "group": sensitive.values})
        tpr = df_tmp[df_tmp.y_true == 1].groupby("group")["y_pred"].mean()
        fpr = df_tmp[df_tmp.y_true == 0].groupby("group")["y_pred"].mean()
        eod = float(max(tpr.max()-tpr.min() if len(tpr)>1 else 0, fpr.max()-fpr.min() if len(fpr)>1 else 0))
        if use_slm:
            r = _run_llama(_build_fairness_prompt(attr, dpd, eod, sr.to_dict()), model_path)
            if r:
                return r
        return _fallback_fairness(attr, dpd, eod, sr.to_dict())

    elif intent == "model_performance":
        if metrics is None:
            return "No model metrics found. Run Train Models first."
        if use_slm:
            r = _run_llama(_build_performance_prompt(metrics), model_path)
            if r:
                return r
        return _fallback_performance(metrics)

    elif intent == "help":
        return _help_text()

    else:
        return ("I didn't understand that query. Try:\n"
                "- `explain applicant 42`\n- `compare applicants 5 and 12`\n"
                "- `what features drive approvals?`\n- `fairness for GENDER`\n- `help`")


def main():
    st.title("Credit Analyst Agent")
    st.caption("Ask questions in plain English. Responses are grounded in SHAP feature attributions from the trained model.")

    # ── Example conversation shown on first load ──────────────────────────────
    with st.expander("See example conversation — what this agent can do", expanded=True):
        st.markdown(
            "**User:** explain applicant 42\n\n"
            "**Agent:** Applicant 42 was classified as **P4** (poor creditworthiness) "
            "with 87% confidence. The three strongest negative factors were:\n"
            "- `num_times_delinquent`: 4 occurrences — the single strongest predictor of credit risk\n"
            "- `CC_utilization`: 91% — sustained high utilization signals financial stress\n"
            "- `Tot_Missed_Pmnt`: 6 — persistent payment failures across trade lines\n\n"
            "**RECOMMENDATION:** To improve from P4 to P3, the applicant needs zero new "
            "delinquencies for 12 months and CC utilization reduced below 60%.\n\n"
            "---\n\n"
            "**User:** fairness for GENDER\n\n"
            "**Agent:** Approval rates — M: 76.3%, F: 77.3%. "
            "Demographic Parity Difference: 0.0105 (**Acceptable** — below RBI threshold of 0.05). "
            "The model does not disadvantage female applicants on approval rate. "
            "Equalized Odds Difference: 0.0500 (**Monitor**) — female applicants have a slightly "
            "higher false positive rate (16.0% vs 10.9%). This means more non-creditworthy female "
            "applicants are incorrectly approved. Warrants monitoring but no immediate mitigation required.\n\n"
            "---\n\n"
            "**User:** what features drive approvals?\n\n"
            "**Agent:** The top predictors (by mean |SHAP| across all applicants) are: "
            "`num_times_delinquent`, `CC_utilization`, `Tot_Missed_Pmnt`, `time_since_recent_deliquency`, "
            "`pct_active_tl`. Delinquency-related features dominate — consistent with "
            "standard credit risk theory. `CC_utilization` captures current financial stress. "
            "`pct_active_tl` (fraction of trade lines still active) reflects credit portfolio health."
        )

    # ── Quick query buttons ───────────────────────────────────────────────────
    st.markdown("**Quick queries** (click to send):")
    qcol1, qcol2, qcol3 = st.columns(3)
    if qcol1.button("Explain applicant 0", use_container_width=True):
        st.session_state["_prefill_query"] = "explain applicant 0"
        st.rerun()
    if qcol2.button("What features drive approvals?", use_container_width=True):
        st.session_state["_prefill_query"] = "what features drive approvals?"
        st.rerun()
    if qcol3.button("Fairness for EDUCATION", use_container_width=True):
        st.session_state["_prefill_query"] = "fairness for EDUCATION"
        st.rerun()

    st.markdown("---")

    with st.sidebar:
        st.markdown("### SLM Configuration")
        binary_found = _find_llama_binary()
        if binary_found:
            st.success(f"llama-cli found: `{binary_found}`")
        else:
            st.warning("llama-cli not found in PATH")
            with st.expander("How to install llama.cpp"):
                st.markdown(
                    "Download the pre-built binary from the "
                    "[llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases) "
                    "and place `llama-cli` in your PATH or project root."
                )

        model_path = st.text_input("GGUF model path",
                                   value=st.session_state.get("agent_model_path", ""),
                                   placeholder="/path/to/model.gguf")
        if model_path:
            st.session_state["agent_model_path"] = model_path
            if Path(model_path).exists():
                size_gb = Path(model_path).stat().st_size / 1e9
                st.success(f"Model found ({size_gb:.1f} GB)")
            else:
                st.error("File not found")
                model_path = None

        with st.expander("Recommended models"):
            for name, url in RECOMMENDED_MODELS.items():
                st.markdown(f"**{name}**")
                st.code(f"wget {url}", language="bash")

        st.markdown("---")
        if binary_found and model_path and Path(model_path).exists():
            st.success("SLM active (llama.cpp)")
        else:
            st.info("Template mode (no SLM)")
            st.caption("Structured template responses with real SHAP numbers.")

    X, y = get_X_y()
    if X is None:
        st.error(
            "No preprocessed data found. "
            "Run the full pipeline: **Home > Launch Demo** (or Upload Data > Preprocess > Train Models) > then return here."
        )
        return

    lr  = _load_model("logistic_regression.pkl", "lr_model")
    ebm = _load_model("ebm_model.pkl",           "ebm_model")

    model_options = {}
    if lr:
        model_options["Logistic Regression"] = ("lr",  lr)
    if ebm:
        model_options["EBM"]                 = ("ebm", ebm)
    if not model_options:
        st.error(
            "No trained models found. "
            "Run **3 · Train Models** first, then return here."
        )
        return

    model_display_key = st.selectbox("Model", list(model_options.keys()), key="agent_model_select")
    model_key, model  = model_options[model_display_key]

    label_map   = _label_map()
    metrics     = st.session_state.get("model_metrics")
    external_df = st.session_state.get("external_df")
    X_test      = st.session_state.get("X_test")
    y_test      = st.session_state.get("y_test")
    model_path  = st.session_state.get("agent_model_path") or None

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle prefill from quick-query buttons
    prefill = st.session_state.pop("_prefill_query", None)

    if query := (prefill or st.chat_input("Ask about an applicant, feature importance, or fairness...")):
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state["chat_history"].append({"role": "user", "content": query})

        intent, args = _classify_intent(query)
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                response = _dispatch(intent, args, model, model_key, X, y, label_map,
                                     model_path, metrics, external_df, X_test, y_test)
            st.markdown(response)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})

    if st.session_state["chat_history"]:
        if st.button("Clear conversation"):
            st.session_state["chat_history"] = []
            st.rerun()

    st.markdown("---")
    st.info("Proceed to Business Summary for NPA impact framing.")


main()
