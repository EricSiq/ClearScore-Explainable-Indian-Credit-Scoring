"""Group-level monitoring diagnostics for the current holdout sample."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

SENSITIVE_ATTRIBUTES = ["GENDER", "EDUCATION", "MARITALSTATUS"]


def _status(value):
    if value < 0.05:
        return "Low gap"
    if value < 0.10:
        return "Review"
    return "Investigate"


def main():
    st.title("Fairness monitoring")
    st.caption("Holdout-sample diagnostics; not a certification of fairness or regulatory compliance.")
    metrics = st.session_state.get("model_metrics")
    external = st.session_state.get("external_df")
    X_test = st.session_state.get("X_test")
    y_test = st.session_state.get("y_test")
    if not all(item is not None for item in [metrics, external, X_test, y_test]):
        st.error("Run data preparation and validation before reviewing monitoring metrics.")
        return
    available = [column for column in SENSITIVE_ATTRIBUTES if column in external.columns]
    if not available:
        st.error("No configured monitoring attributes are present in the uploaded data.")
        return
    model_key = st.selectbox("Model to monitor", ["ebm", "lr"], format_func=lambda key: "EBM (primary)" if key == "ebm" else "Logistic Regression (baseline)")
    attribute = st.selectbox("Monitoring attribute", available)
    try:
        groups = external.loc[X_test.index, attribute].reset_index(drop=True)
    except KeyError:
        groups = external[attribute].iloc[:len(X_test)].reset_index(drop=True)
    predicted = pd.Series(metrics[model_key]["y_pred"]).reset_index(drop=True)
    observed = y_test.reset_index(drop=True)
    approved = predicted.isin([0, 1]).astype(int)
    actual_approved = observed.isin([0, 1]).astype(int)
    summary = pd.DataFrame({"group": groups, "predicted_approval": approved, "actual_approval": actual_approved})
    rates = summary.groupby("group")["predicted_approval"].agg(["mean", "size"]).rename(columns={"mean": "selection_rate", "size": "sample_size"})
    dpd = float(rates["selection_rate"].max() - rates["selection_rate"].min())
    conditioned = summary.groupby(["group", "actual_approval"])["predicted_approval"].mean().unstack()
    tpr_gap = float(conditioned.get(1, pd.Series([0])).max() - conditioned.get(1, pd.Series([0])).min())
    fpr_gap = float(conditioned.get(0, pd.Series([0])).max() - conditioned.get(0, pd.Series([0])).min())
    eod = max(tpr_gap, fpr_gap)
    a, b = st.columns(2)
    a.metric("Selection-rate gap", f"{dpd:.3f}", _status(dpd))
    b.metric("Equalised-odds gap", f"{eod:.3f}", _status(eod))
    st.write("A gap is the maximum difference between observed groups in this holdout. Small groups can make these estimates unstable.")
    display = rates.assign(selection_rate=lambda x: (x.selection_rate * 100).round(1))
    st.dataframe(display.rename(columns={"selection_rate": "selection rate (%)"}), use_container_width=True)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(display.index.astype(str), display["selection_rate"], color="#0f766e")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Predicted approval rate (%)")
    ax.set_title(f"Selection rate by {attribute}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.warning("Use these diagnostics to trigger investigation: verify data quality, group sample sizes, proxy features, calibration, and policy impacts. Do not use them to apply group-specific credit thresholds in this prototype.")


main()
