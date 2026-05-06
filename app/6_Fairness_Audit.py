"""
Page 6 — Fairness Audit

Computes and visualises demographic fairness metrics for the trained credit
scoring model across sensitive attributes: GENDER, EDUCATION, MARITALSTATUS.

Metrics implemented from first principles (no fairlearn/aif360 dependency):
  - Selection Rate by group  : P(Ŷ = approved | group)
  - Demographic Parity Diff  : max(selection_rate) − min(selection_rate)
  - Equalized Odds Diff      : max(|ΔTPR|, |ΔFPR|) across groups

Thresholds aligned with RBI Model Risk Management Guidelines (2023):
  < 0.05  → Acceptable   (green)
  0.05–0.10 → Monitor    (amber)
  > 0.10  → Action req.  (red)

Session state consumed
──────────────────────
  model_metrics   : dict written by Page 3 — contains y_pred, y_proba per model
  X_test          : pd.DataFrame — test feature matrix (index aligned to raw df)
  y_test          : pd.Series   — true integer labels (0–3)
  external_df     : pd.DataFrame — raw external dataset (contains sensitive cols)
  label_map       : dict {0:'P1', 1:'P2', 2:'P3', 3:'P4'}
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Shared components ─────────────────────────────────────────────────────────
from app.components.utils import get_label_map

# ── Constants ─────────────────────────────────────────────────────────────────
_LABEL_MAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}

# Sensitive attributes available in the raw external dataset
SENSITIVE_ATTRS = ["GENDER", "EDUCATION", "MARITALSTATUS"]

# "Approved" = P1 or P2 (integer labels 0 or 1)
APPROVED_CLASSES = {0, 1}

# RBI-aligned fairness thresholds
THRESHOLD_GREEN = 0.05
THRESHOLD_AMBER = 0.10

# Colour palette consistent with Page 3
CLASS_COLOURS = {
    "P1": "#2ecc71",
    "P2": "#3498db",
    "P3": "#f39c12",
    "P4": "#e74c3c",
}


# ── Session state helpers ─────────────────────────────────────────────────────

def _label_map() -> dict:
    return get_label_map()


def _get_required_state():
    """
    Validate and return all session state objects this page needs.
    Returns None on any missing dependency with a clear error message.
    """
    missing = []

    if "model_metrics" not in st.session_state:
        missing.append("`model_metrics` — run **Train Models** first")
    if "X_test" not in st.session_state:
        missing.append("`X_test` — run **Train Models** first")
    if "y_test" not in st.session_state:
        missing.append("`y_test` — run **Train Models** first")
    if "external_df" not in st.session_state:
        missing.append("`external_df` — run **Upload Data** first")

    if missing:
        st.error("Missing required data:\n" + "\n".join(f"  • {m}" for m in missing))
        return None

    return (
        st.session_state["model_metrics"],
        st.session_state["X_test"],
        st.session_state["y_test"],
        st.session_state["external_df"],
        _label_map(),
    )


def _recover_sensitive_attrs(X_test: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recover original string values of sensitive attributes for the test rows.

    X_test has the same index as the original merged DataFrame (preserved through
    train_test_split because we never reset_index on X before splitting).
    We use that index to slice the raw external_df.
    """
    available = [c for c in SENSITIVE_ATTRS if c in external_df.columns]
    if not available:
        st.error(
            "Sensitive attribute columns not found in the external dataset. "
            f"Expected: {SENSITIVE_ATTRS}"
        )
        return pd.DataFrame()

    # Align on index — external_df was loaded before any index manipulation
    try:
        sensitive = external_df.loc[X_test.index, available].copy()
        sensitive.index = range(len(sensitive))   # reset for clean groupby
        return sensitive
    except KeyError:
        # Index mismatch — fall back to positional alignment
        st.warning(
            "Index mismatch between test set and raw data. "
            "Using positional alignment — verify results carefully."
        )
        n = len(X_test)
        return external_df[available].iloc[:n].reset_index(drop=True)


# ── Fairness metric functions ─────────────────────────────────────────────────

def _binary_approval(y: pd.Series) -> pd.Series:
    """Map integer class labels to binary: 1 = approved (P1/P2), 0 = not approved."""
    return y.isin(APPROVED_CLASSES).astype(int)


def selection_rate_by_group(y_pred_bin: pd.Series, sensitive: pd.Series) -> pd.Series:
    """P(Ŷ = approved | group) for each group."""
    return (
        y_pred_bin
        .groupby(sensitive.values)
        .mean()
        .rename("Selection Rate")
        .sort_index()
    )


def demographic_parity_difference(y_pred_bin: pd.Series, sensitive: pd.Series) -> float:
    """max(selection_rate) − min(selection_rate) across groups."""
    rates = y_pred_bin.groupby(sensitive.values).mean()
    return float(rates.max() - rates.min())


def equalized_odds_difference(
    y_true_bin: pd.Series,
    y_pred_bin: pd.Series,
    sensitive: pd.Series,
) -> float:
    """
    max(|ΔTPR|, |ΔFPR|) across groups.
    TPR = P(Ŷ=1 | Y=1, group)  — true positive rate (sensitivity)
    FPR = P(Ŷ=1 | Y=0, group)  — false positive rate (1 − specificity)
    """
    df_tmp = pd.DataFrame({
        "y_true": y_true_bin.values,
        "y_pred": y_pred_bin.values,
        "group":  sensitive.values,
    })

    positives = df_tmp[df_tmp["y_true"] == 1]
    negatives = df_tmp[df_tmp["y_true"] == 0]

    tpr_by_group = positives.groupby("group")["y_pred"].mean()
    fpr_by_group = negatives.groupby("group")["y_pred"].mean()

    tpr_diff = float(tpr_by_group.max() - tpr_by_group.min()) if len(tpr_by_group) > 1 else 0.0
    fpr_diff = float(fpr_by_group.max() - fpr_by_group.min()) if len(fpr_by_group) > 1 else 0.0

    return max(tpr_diff, fpr_diff)


def per_class_rate_by_group(
    y_pred: pd.Series,
    sensitive: pd.Series,
    label_map: dict,
) -> pd.DataFrame:
    """
    For each group, compute the fraction of applicants predicted into each class.
    Returns a DataFrame: rows = groups, columns = P1/P2/P3/P4.
    """
    df_tmp = pd.DataFrame({"pred": y_pred.values, "group": sensitive.values})
    result = {}
    for cls_int, cls_label in sorted(label_map.items()):
        result[cls_label] = (df_tmp["pred"] == cls_int).groupby(df_tmp["group"]).mean()
    return pd.DataFrame(result)


def tpr_fpr_by_group(
    y_true_bin: pd.Series,
    y_pred_bin: pd.Series,
    sensitive: pd.Series,
) -> pd.DataFrame:
    """Return TPR and FPR per group as a DataFrame."""
    df_tmp = pd.DataFrame({
        "y_true": y_true_bin.values,
        "y_pred": y_pred_bin.values,
        "group":  sensitive.values,
    })
    positives = df_tmp[df_tmp["y_true"] == 1]
    negatives = df_tmp[df_tmp["y_true"] == 0]

    tpr = positives.groupby("group")["y_pred"].mean().rename("TPR")
    fpr = negatives.groupby("group")["y_pred"].mean().rename("FPR")
    return pd.concat([tpr, fpr], axis=1).round(4)


# ── Traffic-light display ─────────────────────────────────────────────────────

def _traffic_light_metric(label: str, value: float, help_text: str = ""):
    """
    Display a metric card with traffic-light background colour based on
    RBI fairness thresholds.
    """
    if value < THRESHOLD_GREEN:
        colour = "#d4edda"   # green
        status = "✅ Acceptable"
    elif value < THRESHOLD_AMBER:
        colour = "#fff3cd"   # amber
        status = "⚠️ Monitor"
    else:
        colour = "#f8d7da"   # red
        status = "🚨 Action Required"

    st.markdown(
        f"""
        <div style="
            background-color:{colour};
            border-radius:8px;
            padding:12px 16px;
            margin-bottom:8px;
        ">
            <div style="font-size:13px;color:#555;">{label}</div>
            <div style="font-size:28px;font-weight:bold;">{value:.4f}</div>
            <div style="font-size:12px;">{status}</div>
            <div style="font-size:11px;color:#777;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _plot_selection_rate_bar(
    sr: pd.Series,
    attr: str,
    dpd: float,
) -> plt.Figure:
    """Horizontal bar chart of selection rate per group."""
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(sr) * 0.55)))

    colours = ["#2ecc71" if v >= sr.mean() else "#e74c3c" for v in sr.values]
    bars = ax.barh(sr.index.astype(str), sr.values, color=colours, edgecolor="white")

    # Reference line at overall mean
    ax.axvline(sr.mean(), color="#555", linestyle="--", linewidth=1,
               label=f"Overall mean: {sr.mean():.1%}")

    # Annotate bars
    for bar, val in zip(bars, sr.values):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=9,
        )

    ax.set_xlabel("Approval Rate (P1 + P2)", fontsize=10)
    ax.set_title(f"Selection Rate by {attr}\n(DPD = {dpd:.4f})", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def _plot_per_class_heatmap(
    pcr: pd.DataFrame,
    attr: str,
) -> plt.Figure:
    """
    Heatmap: rows = demographic groups, columns = P1/P2/P3/P4.
    Cell values = fraction of group predicted into that class.
    """
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(pcr) * 0.55)))
    sns.heatmap(
        pcr,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=pcr.values.max(),
        cbar_kws={"format": mticker.PercentFormatter(xmax=1)},
    )
    ax.set_title(
        f"Predicted Class Distribution by {attr}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel(attr, fontsize=10)
    plt.tight_layout()
    return fig


def _plot_tpr_fpr(tpr_fpr_df: pd.DataFrame, attr: str) -> plt.Figure:
    """Grouped bar chart of TPR and FPR per demographic group."""
    groups = tpr_fpr_df.index.astype(str).tolist()
    x      = np.arange(len(groups))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(groups) * 0.55)))
    ax.bar(x - width/2, tpr_fpr_df["TPR"], width, label="TPR (Sensitivity)",
           color="#3498db", alpha=0.85)
    ax.bar(x + width/2, tpr_fpr_df["FPR"], width, label="FPR (1 − Specificity)",
           color="#e74c3c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title(f"TPR & FPR by {attr}", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ── Narrative generator ───────────────────────────────────────────────────────

def _generate_narrative(
    attr: str,
    sr: pd.Series,
    dpd: float,
    eod: float,
    label_map: dict,
) -> str:
    """
    Generate a plain-English fairness summary for one sensitive attribute.
    No LLM required — template-based with real numbers.
    """
    highest_group = sr.idxmax()
    lowest_group  = sr.idxmin()
    highest_rate  = sr.max()
    lowest_rate   = sr.min()

    if dpd < THRESHOLD_GREEN:
        dpd_verdict = "within acceptable limits"
    elif dpd < THRESHOLD_AMBER:
        dpd_verdict = "at a level that warrants monitoring"
    else:
        dpd_verdict = "above the action-required threshold — mitigation recommended"

    lines = [
        f"**{attr} Fairness Summary**",
        "",
        f"Applicants in the **{highest_group}** group are approved at the highest rate "
        f"(**{highest_rate:.1%}**), while **{lowest_group}** applicants are approved at "
        f"**{lowest_rate:.1%}** — a gap of **{dpd:.1%}** (Demographic Parity Difference).",
        "",
        f"This gap is {dpd_verdict}.",
        "",
        f"The Equalized Odds Difference is **{eod:.4f}**, measuring whether the model's "
        f"error rates (false approvals and false rejections) are consistent across groups.",
    ]

    if dpd >= THRESHOLD_AMBER:
        lines += [
            "",
            "**Recommended actions:**",
            f"- Investigate whether `{attr}` is a direct or proxy feature in the model",
            "- Consider reweighting training samples to balance approval rates",
            "- Review whether the approval threshold should be group-specific",
            "- Document this finding for RBI model risk management audit trail",
        ]

    return "\n".join(lines)


# ── Main page ─────────────────────────────────────────────────────────────────

def main():
    st.title("⚖️ Fairness Audit")
    st.caption(
        "Demographic fairness analysis across sensitive attributes. "
        "Thresholds aligned with RBI Model Risk Management Guidelines (2023)."
    )

    # ── Threshold legend ──────────────────────────────────────────────────────
    with st.expander("Threshold reference"):
        col1, col2, col3 = st.columns(3)
        col1.markdown("🟢 **< 0.05** — Acceptable")
        col2.markdown("🟡 **0.05 – 0.10** — Monitor & document")
        col3.markdown("🔴 **> 0.10** — Action required before deployment")
        st.caption(
            "Source: RBI Guidelines on Model Risk Management (2023), "
            "Digital India Act (proposed), EU AI Act Article 10 (reference)."
        )

    st.markdown("---")

    # ── Load dependencies ─────────────────────────────────────────────────────
    state = _get_required_state()
    if state is None:
        return

    model_metrics, X_test, y_test, external_df, label_map = state

    # ── Model selector ────────────────────────────────────────────────────────
    available_models = [k for k in ["lr", "ebm"] if k in model_metrics]
    if not available_models:
        st.error("No model metrics found. Run **Train Models** first.")
        return

    model_display = {"lr": "Logistic Regression", "ebm": "EBM"}
    model_key = st.selectbox(
        "Model to audit",
        available_models,
        format_func=lambda k: model_display.get(k, k),
    )

    y_pred_raw = model_metrics[model_key]["y_pred"]   # numpy array of int labels 0–3
    y_pred     = pd.Series(y_pred_raw, index=X_test.index)
    y_true     = y_test.copy()

    # Binary approval series (aligned index)
    y_pred_bin = _binary_approval(y_pred)
    y_true_bin = _binary_approval(y_true)

    # ── Recover sensitive attributes ──────────────────────────────────────────
    sensitive_df = _recover_sensitive_attrs(X_test, external_df)
    if sensitive_df.empty:
        return

    available_attrs = sensitive_df.columns.tolist()

    # ── Attribute selector ────────────────────────────────────────────────────
    attr = st.selectbox("Sensitive attribute", available_attrs)
    sensitive_col = sensitive_df[attr].reset_index(drop=True)

    # Reset prediction series index to match sensitive_col
    y_pred_bin_r = y_pred_bin.reset_index(drop=True)
    y_true_bin_r = y_true_bin.reset_index(drop=True)
    y_pred_r     = y_pred.reset_index(drop=True)

    st.markdown("---")

    # ── Compute metrics ───────────────────────────────────────────────────────
    sr  = selection_rate_by_group(y_pred_bin_r, sensitive_col)
    dpd = demographic_parity_difference(y_pred_bin_r, sensitive_col)
    eod = equalized_odds_difference(y_true_bin_r, y_pred_bin_r, sensitive_col)
    pcr = per_class_rate_by_group(y_pred_r, sensitive_col, label_map)
    tpr_fpr_df = tpr_fpr_by_group(y_true_bin_r, y_pred_bin_r, sensitive_col)

    # ── Section 1: Headline metrics ───────────────────────────────────────────
    st.subheader(f"1 — Headline Fairness Metrics ({attr})")

    m1, m2 = st.columns(2)
    with m1:
        _traffic_light_metric(
            "Demographic Parity Difference",
            dpd,
            help_text="max(approval_rate) − min(approval_rate) across groups. "
                      "Measures whether the model approves different groups at equal rates.",
        )
    with m2:
        _traffic_light_metric(
            "Equalized Odds Difference",
            eod,
            help_text="max(|ΔTPR|, |ΔFPR|) across groups. "
                      "Measures whether error rates are consistent across groups.",
        )

    st.markdown("---")

    # ── Section 2: Selection rate bar chart ───────────────────────────────────
    st.subheader(f"2 — Approval Rate by {attr}")

    fig_sr = _plot_selection_rate_bar(sr, attr, dpd)
    st.pyplot(fig_sr)
    plt.close(fig_sr)

    # Tabular view
    sr_df = sr.to_frame()
    sr_df["Count"] = sensitive_col.value_counts().reindex(sr.index).fillna(0).astype(int)
    sr_df["Selection Rate"] = sr_df["Selection Rate"].map("{:.1%}".format)
    st.dataframe(sr_df, use_container_width=True)

    st.markdown("---")

    # ── Section 3: Per-class heatmap ──────────────────────────────────────────
    st.subheader(f"3 — Predicted Class Distribution by {attr}")
    st.caption(
        "Each row shows what fraction of that demographic group was predicted "
        "into each credit tier. Uniform rows indicate no demographic bias in tier assignment."
    )

    fig_hm = _plot_per_class_heatmap(pcr, attr)
    st.pyplot(fig_hm)
    plt.close(fig_hm)

    with st.expander("Raw values"):
        st.dataframe(
            pcr.style.format("{:.1%}").background_gradient(cmap="YlOrRd", axis=None),
            use_container_width=True,
        )

    st.markdown("---")

    # ── Section 4: TPR / FPR breakdown ───────────────────────────────────────
    st.subheader(f"4 — Error Rate Parity by {attr}")
    st.caption(
        "TPR (True Positive Rate): fraction of genuinely creditworthy applicants correctly approved. "
        "FPR (False Positive Rate): fraction of non-creditworthy applicants incorrectly approved. "
        "Both should be similar across groups for equalized odds."
    )

    fig_tpr = _plot_tpr_fpr(tpr_fpr_df, attr)
    st.pyplot(fig_tpr)
    plt.close(fig_tpr)

    with st.expander("Raw TPR / FPR values"):
        st.dataframe(
            tpr_fpr_df.style.format("{:.1%}").background_gradient(cmap="Blues"),
            use_container_width=True,
        )

    st.markdown("---")

    # ── Section 5: Narrative summary ─────────────────────────────────────────
    st.subheader(f"5 — Narrative Summary")

    narrative = _generate_narrative(attr, sr, dpd, eod, label_map)
    st.markdown(narrative)

    st.markdown("---")

    # ── Section 6: All-attributes overview ───────────────────────────────────
    st.subheader("6 — All Attributes Overview")
    st.caption("Quick scan across all sensitive attributes to identify which need attention.")

    overview_rows = []
    for a in available_attrs:
        sc   = sensitive_df[a].reset_index(drop=True)
        _dpd = demographic_parity_difference(y_pred_bin_r, sc)
        _eod = equalized_odds_difference(y_true_bin_r, y_pred_bin_r, sc)

        if _dpd < THRESHOLD_GREEN:
            dpd_status = "✅ Acceptable"
        elif _dpd < THRESHOLD_AMBER:
            dpd_status = "⚠️ Monitor"
        else:
            dpd_status = "🚨 Action Required"

        if _eod < THRESHOLD_GREEN:
            eod_status = "✅ Acceptable"
        elif _eod < THRESHOLD_AMBER:
            eod_status = "⚠️ Monitor"
        else:
            eod_status = "🚨 Action Required"

        overview_rows.append({
            "Attribute":  a,
            "DPD":        round(_dpd, 4),
            "DPD Status": dpd_status,
            "EOD":        round(_eod, 4),
            "EOD Status": eod_status,
        })

    overview_df = pd.DataFrame(overview_rows).set_index("Attribute")
    st.dataframe(overview_df, use_container_width=True)

    st.info(
        "➡ Proceed to **Credit Analyst Agent** for natural language explanations, "
        "or **Business Summary** for ₹ NPA impact framing."
    )


if __name__ == "__main__":
    main()
