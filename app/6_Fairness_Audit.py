import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from app.components.utils import get_label_map

_LABEL_MAP      = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
SENSITIVE_ATTRS = ["GENDER", "EDUCATION", "MARITALSTATUS"]
APPROVED_CLASSES = {0, 1}
THRESHOLD_GREEN  = 0.05
THRESHOLD_AMBER  = 0.10
CLASS_COLOURS    = {"P1": "#2ecc71", "P2": "#3498db", "P3": "#f39c12", "P4": "#e74c3c"}


def _label_map() -> dict:
    return get_label_map()


def _get_required_state():
    missing = []
    if "model_metrics" not in st.session_state:
        missing.append("`model_metrics` (run Train Models first)")
    if "X_test" not in st.session_state:
        missing.append("`X_test` (run Train Models first)")
    if "y_test" not in st.session_state:
        missing.append("`y_test` (run Train Models first)")
    if "external_df" not in st.session_state:
        missing.append("`external_df` (run Upload Data first)")
    if missing:
        st.error("Missing required data:\n" + "\n".join(f"  - {m}" for m in missing))
        return None
    return (
        st.session_state["model_metrics"],
        st.session_state["X_test"],
        st.session_state["y_test"],
        st.session_state["external_df"],
        _label_map(),
    )


def _recover_sensitive_attrs(X_test: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in SENSITIVE_ATTRS if c in external_df.columns]
    if not available:
        st.error(f"Sensitive attribute columns not found. Expected: {SENSITIVE_ATTRS}")
        return pd.DataFrame()
    try:
        sensitive = external_df.loc[X_test.index, available].copy()
        sensitive.index = range(len(sensitive))
        return sensitive
    except KeyError:
        st.warning("Index mismatch. Using positional alignment.")
        return external_df[available].iloc[:len(X_test)].reset_index(drop=True)


def _binary_approval(y: pd.Series) -> pd.Series:
    return y.isin(APPROVED_CLASSES).astype(int)


def selection_rate_by_group(y_pred_bin: pd.Series, sensitive: pd.Series) -> pd.Series:
    return y_pred_bin.groupby(sensitive.values).mean().rename("Selection Rate").sort_index()


def demographic_parity_difference(y_pred_bin: pd.Series, sensitive: pd.Series) -> float:
    rates = y_pred_bin.groupby(sensitive.values).mean()
    return float(rates.max() - rates.min())


def equalized_odds_difference(y_true_bin: pd.Series, y_pred_bin: pd.Series, sensitive: pd.Series) -> float:
    df_tmp = pd.DataFrame({"y_true": y_true_bin.values, "y_pred": y_pred_bin.values, "group": sensitive.values})
    pos = df_tmp[df_tmp["y_true"] == 1]
    neg = df_tmp[df_tmp["y_true"] == 0]
    tpr = pos.groupby("group")["y_pred"].mean()
    fpr = neg.groupby("group")["y_pred"].mean()
    tpr_diff = float(tpr.max() - tpr.min()) if len(tpr) > 1 else 0.0
    fpr_diff = float(fpr.max() - fpr.min()) if len(fpr) > 1 else 0.0
    return max(tpr_diff, fpr_diff)


def per_class_rate_by_group(y_pred: pd.Series, sensitive: pd.Series, label_map: dict) -> pd.DataFrame:
    df_tmp = pd.DataFrame({"pred": y_pred.values, "group": sensitive.values})
    return pd.DataFrame({
        lbl: (df_tmp["pred"] == ci).groupby(df_tmp["group"]).mean()
        for ci, lbl in sorted(label_map.items())
    })


def tpr_fpr_by_group(y_true_bin: pd.Series, y_pred_bin: pd.Series, sensitive: pd.Series) -> pd.DataFrame:
    df_tmp = pd.DataFrame({"y_true": y_true_bin.values, "y_pred": y_pred_bin.values, "group": sensitive.values})
    tpr = df_tmp[df_tmp["y_true"] == 1].groupby("group")["y_pred"].mean().rename("TPR")
    fpr = df_tmp[df_tmp["y_true"] == 0].groupby("group")["y_pred"].mean().rename("FPR")
    return pd.concat([tpr, fpr], axis=1).round(4)


def _traffic_light_metric(label: str, value: float, help_text: str = ""):
    if value < THRESHOLD_GREEN:
        colour, status = "#d4edda", "Acceptable"
    elif value < THRESHOLD_AMBER:
        colour, status = "#fff3cd", "Monitor"
    else:
        colour, status = "#f8d7da", "Action Required"
    st.markdown(
        f'<div style="background-color:{colour};border-radius:8px;padding:12px 16px;margin-bottom:8px;">'
        f'<div style="font-size:13px;color:#555;">{label}</div>'
        f'<div style="font-size:28px;font-weight:bold;">{value:.4f}</div>'
        f'<div style="font-size:12px;">{status}</div>'
        f'<div style="font-size:11px;color:#777;">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


def _plot_selection_rate_bar(sr: pd.Series, attr: str, dpd: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(sr) * 0.55)))
    colours = ["#2ecc71" if v >= sr.mean() else "#e74c3c" for v in sr.values]
    bars = ax.barh(sr.index.astype(str), sr.values, color=colours, edgecolor="white")
    ax.axvline(sr.mean(), color="#555", linestyle="--", linewidth=1, label=f"Overall mean: {sr.mean():.1%}")
    for bar, val in zip(bars, sr.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", fontsize=9)
    ax.set_xlabel("Approval Rate (P1 + P2)", fontsize=10)
    ax.set_title(f"Selection Rate by {attr}\n(DPD = {dpd:.4f})", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def _plot_per_class_heatmap(pcr: pd.DataFrame, attr: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(pcr) * 0.55)))
    sns.heatmap(pcr, annot=True, fmt=".1%", cmap="YlOrRd", linewidths=0.5, ax=ax,
                vmin=0, vmax=pcr.values.max(),
                cbar_kws={"format": mticker.PercentFormatter(xmax=1)})
    ax.set_title(f"Predicted Class Distribution by {attr}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel(attr, fontsize=10)
    plt.tight_layout()
    return fig


def _plot_tpr_fpr(tpr_fpr_df: pd.DataFrame, attr: str) -> plt.Figure:
    groups = tpr_fpr_df.index.astype(str).tolist()
    x = np.arange(len(groups)); width = 0.35
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(groups) * 0.55)))
    ax.bar(x - width/2, tpr_fpr_df["TPR"], width, label="TPR (Sensitivity)", color="#3498db", alpha=0.85)
    ax.bar(x + width/2, tpr_fpr_df["FPR"], width, label="FPR (1 - Specificity)", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title(f"TPR & FPR by {attr}", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def _generate_narrative(attr: str, sr: pd.Series, dpd: float, eod: float, label_map: dict) -> str:
    highest_group = sr.idxmax()
    lowest_group  = sr.idxmin()
    if dpd < THRESHOLD_GREEN:
        dpd_verdict = "within acceptable limits"
    elif dpd < THRESHOLD_AMBER:
        dpd_verdict = "at a level that warrants monitoring"
    else:
        dpd_verdict = "above the action threshold, mitigation recommended"

    lines = [
        f"**{attr} Fairness Summary**", "",
        f"Applicants in the **{highest_group}** group are approved at the highest rate "
        f"(**{sr.max():.1%}**), while **{lowest_group}** applicants are approved at "
        f"**{sr.min():.1%}** (gap of **{dpd:.1%}**, Demographic Parity Difference).", "",
        f"This gap is {dpd_verdict}.", "",
        f"The Equalized Odds Difference is **{eod:.4f}**, measuring whether the model's "
        f"error rates are consistent across groups.",
    ]
    if dpd >= THRESHOLD_AMBER:
        lines += [
            "", "**Recommended actions:**",
            f"- Check whether `{attr}` acts as a proxy for other features in the model",
            "- Consider reweighting training samples to balance approval rates",
            "- Review whether the approval threshold should differ by group",
            "- Document this finding in the RBI model risk management register",
        ]
    return "\n".join(lines)


def main():
    st.title("Fairness Audit")
    st.caption("CreditLens. Demographic fairness analysis. RBI MRM Guidelines (2023).")

    with st.expander("Threshold reference"):
        col1, col2, col3 = st.columns(3)
        col1.markdown("**Below 0.05** - Acceptable")
        col2.markdown("**0.05 to 0.10** - Monitor and document")
        col3.markdown("**Above 0.10** - Action required before deployment")
        st.caption("Source: RBI Guidelines on Model Risk Management (2023).")

    st.markdown("---")

    state = _get_required_state()
    if state is None:
        return

    model_metrics, X_test, y_test, external_df, label_map = state

    available_models = [k for k in ["lr", "ebm"] if k in model_metrics]
    if not available_models:
        st.error("No model metrics found. Run Train Models first.")
        return

    model_display = {"lr": "Logistic Regression", "ebm": "EBM"}
    model_key = st.selectbox("Model to audit", available_models,
                             format_func=lambda k: model_display.get(k, k))

    y_pred_raw = model_metrics[model_key]["y_pred"]
    y_pred     = pd.Series(y_pred_raw, index=X_test.index)
    y_true     = y_test.copy()
    y_pred_bin = _binary_approval(y_pred)
    y_true_bin = _binary_approval(y_true)

    sensitive_df = _recover_sensitive_attrs(X_test, external_df)
    if sensitive_df.empty:
        return

    available_attrs = sensitive_df.columns.tolist()
    attr = st.selectbox("Sensitive attribute", available_attrs)
    sensitive_col = sensitive_df[attr].reset_index(drop=True)

    y_pred_bin_r = y_pred_bin.reset_index(drop=True)
    y_true_bin_r = y_true_bin.reset_index(drop=True)
    y_pred_r     = y_pred.reset_index(drop=True)

    st.markdown("---")

    sr  = selection_rate_by_group(y_pred_bin_r, sensitive_col)
    dpd = demographic_parity_difference(y_pred_bin_r, sensitive_col)
    eod = equalized_odds_difference(y_true_bin_r, y_pred_bin_r, sensitive_col)
    pcr = per_class_rate_by_group(y_pred_r, sensitive_col, label_map)
    tpr_fpr_df = tpr_fpr_by_group(y_true_bin_r, y_pred_bin_r, sensitive_col)

    st.subheader(f"Fairness Metrics: {attr}")
    m1, m2 = st.columns(2)
    with m1:
        _traffic_light_metric("Demographic Parity Difference", dpd,
            help_text="max(approval_rate) - min(approval_rate) across groups.")
    with m2:
        _traffic_light_metric("Equalized Odds Difference", eod,
            help_text="max(|DTPR|, |DFPR|) across groups.")

    st.markdown("---")
    st.subheader(f"Approval Rate by {attr}")
    fig_sr = _plot_selection_rate_bar(sr, attr, dpd)
    st.pyplot(fig_sr); plt.close(fig_sr)
    sr_df = sr.to_frame()
    sr_df["Count"] = sensitive_col.value_counts().reindex(sr.index).fillna(0).astype(int)
    sr_df["Selection Rate"] = sr_df["Selection Rate"].map("{:.1%}".format)
    st.dataframe(sr_df, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Class Distribution by {attr}")
    st.caption("Each row shows what fraction of that demographic group was predicted into each credit tier.")
    fig_hm = _plot_per_class_heatmap(pcr, attr)
    st.pyplot(fig_hm); plt.close(fig_hm)
    with st.expander("Raw values"):
        st.dataframe(pcr.round(3), use_container_width=True)

    st.markdown("---")
    st.subheader(f"Error Rate Parity by {attr}")
    st.caption("TPR and FPR should be similar across groups for equalized odds.")
    fig_tpr = _plot_tpr_fpr(tpr_fpr_df, attr)
    st.pyplot(fig_tpr); plt.close(fig_tpr)
    with st.expander("Raw TPR / FPR values"):
        st.dataframe(tpr_fpr_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Summary")
    st.markdown(_generate_narrative(attr, sr, dpd, eod, label_map))

    # ── Action panel — shown only when thresholds are breached ────────────────
    if dpd >= THRESHOLD_AMBER or eod >= THRESHOLD_AMBER:
        st.markdown("---")
        st.subheader("Mitigation Options")
        st.caption(
            "One or more metrics exceed the RBI MRM monitoring threshold. "
            "The following mitigation strategies are standard practice for regulated credit models."
        )
        with st.expander("Option 1: Post-processing with group-specific thresholds (no retraining)"):
            st.markdown(
                "Set different approval probability thresholds per demographic group "
                "to equalize TPR and FPR.\n\n"
                "**Pros:** No retraining. Immediate effect. Fully auditable.\n\n"
                "**Cons:** Slightly lower overall accuracy (typically 1–2%). "
                "Requires justification in model documentation.\n\n"
                "**Tool:** `fairlearn.postprocessing.ThresholdOptimizer` "
                "(when scipy build issues are resolved in the environment)."
            )
        with st.expander("Option 2: Reweight training data by group (retraining required)"):
            st.markdown(
                "Assign higher training weights to underrepresented demographic groups "
                "so the model learns more balanced error rates.\n\n"
                "**Pros:** Addresses the root cause in the model itself.\n\n"
                "**Cons:** Requires full retraining. May reduce accuracy on majority group.\n\n"
                "**Implementation:** `sklearn.utils.class_weight.compute_sample_weight` "
                "with sensitive attribute as the class label."
            )
        with st.expander("Option 3: Feature audit for proxy discrimination"):
            st.markdown(
                f"If `{attr}` is correlated with features the model uses directly, "
                "removing or transforming those features may reduce bias without "
                "explicitly using demographic information.\n\n"
                "**For EDUCATION specifically:** `EDUCATION` correlates with `NETMONTHLYINCOME`. "
                "If income is a direct model feature, education-level bias may be a proxy effect "
                "rather than the model directly discriminating.\n\n"
                "**RBI requirement:** Document the root cause analysis and chosen mitigation "
                "strategy in the Model Risk Management register before deployment."
            )

    st.markdown("---")
    st.subheader("All Attributes Overview")
    st.caption("Quick scan across all sensitive attributes.")
    overview_rows = []
    for a in available_attrs:
        sc   = sensitive_df[a].reset_index(drop=True)
        _dpd = demographic_parity_difference(y_pred_bin_r, sc)
        _eod = equalized_odds_difference(y_true_bin_r, y_pred_bin_r, sc)
        dpd_status = "Acceptable" if _dpd < THRESHOLD_GREEN else ("Monitor" if _dpd < THRESHOLD_AMBER else "Action Required")
        eod_status = "Acceptable" if _eod < THRESHOLD_GREEN else ("Monitor" if _eod < THRESHOLD_AMBER else "Action Required")
        overview_rows.append({"Attribute": a, "DPD": round(_dpd, 4), "DPD Status": dpd_status,
                               "EOD": round(_eod, 4), "EOD Status": eod_status})
    st.dataframe(pd.DataFrame(overview_rows).set_index("Attribute"), use_container_width=True)
    st.info("Proceed to Credit Analyst Agent or Business Summary.")


main()
