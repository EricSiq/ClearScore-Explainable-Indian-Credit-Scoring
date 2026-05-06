"""
Page 8 — Business Impact Summary

Translates model evaluation metrics into business language for credit risk
managers and bank executives. No ML jargon — only outcomes that matter:
approval rates, NPA exposure avoided, and risk tier segmentation.

Session state consumed
──────────────────────
  model_metrics : dict written by Page 3
    lr / ebm    : confusion_matrix, y_pred, f1_macro, f1_weighted, auc_ovr
    n_test      : int
    class_counts: dict
  label_map     : {0:'P1', 1:'P2', 2:'P3', 3:'P4'}
"""

import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER

# ── Shared components ─────────────────────────────────────────────────────────
from app.components.utils import get_label_map

# ── Constants ─────────────────────────────────────────────────────────────────
_LABEL_MAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}

TIER_DESCRIPTIONS = {
    "P1": "Excellent creditworthiness",
    "P2": "Good creditworthiness",
    "P3": "Marginal — conditional approval",
    "P4": "Poor creditworthiness — reject",
}
TIER_ACTIONS = {
    "P1": "Approve — best rate",
    "P2": "Approve — standard rate",
    "P3": "Conditional / higher rate",
    "P4": "Reject / secured only",
}

# Default NPA assumptions (overridden by sliders)
DEFAULT_AVG_LOAN   = 200_000   # ₹2 lakh
DEFAULT_P4_DEFAULT = 0.40      # 40% default rate for P4
DEFAULT_P3_DEFAULT = 0.15      # 15% default rate for P3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_map() -> dict:
    return get_label_map()


def _get_metrics():
    if "model_metrics" not in st.session_state:
        st.error("No model metrics found. Run **Train Models** first.")
        return None
    return st.session_state["model_metrics"]


def _fmt_cr(rupees: float) -> str:
    """Format rupees as ₹X.XX Cr."""
    return f"₹{rupees / 1e7:,.2f} Cr"


def _fmt_lakh(rupees: float) -> str:
    return f"₹{rupees / 1e5:,.0f}L"


# ── Business metric computation ───────────────────────────────────────────────

def _compute_business_metrics(
    cm: np.ndarray,
    y_pred: np.ndarray,
    n_test: int,
    avg_loan: int,
    p4_rate: float,
    p3_rate: float,
) -> dict:
    """
    Derive all business-facing numbers from the confusion matrix and predictions.

    NPA avoided  = correctly rejected high-risk applicants × avg_loan × default_rate
    NPA exposure = incorrectly approved P4 applicants × avg_loan × p4_rate

    Confusion matrix layout (rows=actual, cols=predicted):
      [0,0]=P1  [1,1]=P2  [2,2]=P3  [3,3]=P4  (diagonal = correct)
    """
    # Tier counts from predictions
    approved = int((y_pred <= 1).sum())   # P1 + P2
    marginal = int((y_pred == 2).sum())   # P3
    rejected = int((y_pred == 3).sum())   # P4

    # Diagonal = correctly classified per tier
    tp_p1 = int(cm[0, 0])
    tp_p2 = int(cm[1, 1])
    tp_p3 = int(cm[2, 2])
    tp_p4 = int(cm[3, 3])
    total_correct = tp_p1 + tp_p2 + tp_p3 + tp_p4
    accuracy = total_correct / n_test

    # NPA avoided: correctly rejected high-risk applicants
    npa_p4    = tp_p4 * avg_loan * p4_rate
    npa_p3    = tp_p3 * avg_loan * p3_rate
    npa_total = npa_p4 + npa_p3

    # Residual exposure: P4 applicants incorrectly approved as P1 or P2
    fn_p4        = int(cm[3, :2].sum())
    npa_exposure = fn_p4 * avg_loan * p4_rate

    return {
        "n_test":        n_test,
        "approved":      approved,
        "marginal":      marginal,
        "rejected":      rejected,
        "approval_rate": approved / n_test,
        "tp_p1": tp_p1, "tp_p2": tp_p2, "tp_p3": tp_p3, "tp_p4": tp_p4,
        "total_correct": total_correct,
        "accuracy":      accuracy,
        "npa_p4":        npa_p4,
        "npa_p3":        npa_p3,
        "npa_total":     npa_total,
        "fn_p4":         fn_p4,
        "npa_exposure":  npa_exposure,
    }


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _plot_tier_donut(bm: dict, title: str) -> plt.Figure:
    """Donut chart of predicted tier distribution (Approved / Marginal / Rejected)."""
    sizes  = [bm["approved"], bm["marginal"], bm["rejected"]]
    labels = ["Approved (P1+P2)", "Marginal (P3)", "Rejected (P4)"]
    clrs   = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=clrs,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def _plot_npa_bar(bm_lr: dict, bm_ebm: dict) -> plt.Figure:
    """Grouped bar: NPA avoided vs residual exposure for LR and EBM."""
    models   = ["Logistic Regression", "EBM"]
    avoided  = [bm_lr["npa_total"] / 1e7,   bm_ebm["npa_total"] / 1e7]
    exposure = [bm_lr["npa_exposure"] / 1e7, bm_ebm["npa_exposure"] / 1e7]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 4))
    b1 = ax.bar(x - width/2, avoided,  width, label="NPA Avoided (₹ Cr)",
                color="#2ecc71", alpha=0.9)
    b2 = ax.bar(x + width/2, exposure, width, label="Residual NPA Exposure (₹ Cr)",
                color="#e74c3c", alpha=0.9)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.05,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("₹ Crore", fontsize=10)
    ax.set_title("NPA Impact Comparison — LR vs EBM", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def _plot_correct_by_tier(bm_lr: dict, bm_ebm: dict) -> plt.Figure:
    """Grouped bar: correct classifications per tier for LR and EBM."""
    tiers  = ["P1", "P2", "P3", "P4"]
    lr_tp  = [bm_lr[f"tp_{t.lower()}"]  for t in tiers]
    ebm_tp = [bm_ebm[f"tp_{t.lower()}"] for t in tiers]

    x     = np.arange(len(tiers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x - width/2, lr_tp,  width, label="Logistic Regression",
           color="#3498db", alpha=0.85)
    ax.bar(x + width/2, ebm_tp, width, label="EBM",
           color="#2ecc71", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Correctly classified", fontsize=10)
    ax.set_title("Correct Classifications by Tier", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ── PDF generation ────────────────────────────────────────────────────────────

def _build_pdf(
    bm_lr: dict,
    bm_ebm: dict,
    lr_meta: dict,
    ebm_meta: dict,
    avg_loan: int,
    p4_rate: float,
    p3_rate: float,
) -> bytes:
    """
    Build a one-page A4 PDF business summary.
    Returns bytes for st.download_button.
    """
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"],
        fontSize=15, spaceAfter=4, alignment=TA_CENTER,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["BodyText"],
        fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=11, spaceBefore=10, spaceAfter=4,
    )
    fn_style = ParagraphStyle(
        "Footnote", parent=styles["BodyText"],
        fontSize=8, textColor=colors.HexColor("#555555"),
    )

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Credit Scoring Model — Business Impact Summary", title_style))
    story.append(Paragraph(
        f"Test set: {bm_ebm['n_test']:,} applicants  |  "
        f"Avg loan: {_fmt_lakh(avg_loan)}  |  "
        f"P4 default rate: {p4_rate:.0%}  |  P3 default rate: {p3_rate:.0%}",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1f4e79"), spaceAfter=6))

    # ── EBM headline table ────────────────────────────────────────────────────
    story.append(Paragraph("EBM Model — Headline Outcomes", h2_style))
    headline_data = [
        ["Metric", "Value", "Business Meaning"],
        ["Applicants scored",
         f"{bm_ebm['n_test']:,}",
         "Total test-set applicants evaluated"],
        ["Correctly classified",
         f"{bm_ebm['total_correct']:,} ({bm_ebm['accuracy']:.1%})",
         "Applicants placed in the correct credit tier"],
        ["Approved (P1+P2)",
         f"{bm_ebm['approved']:,} ({bm_ebm['approval_rate']:.1%})",
         "Creditworthy applicants approved for loans"],
        ["Marginal flagged (P3)",
         f"{bm_ebm['marginal']:,} ({bm_ebm['marginal']/bm_ebm['n_test']:.1%})",
         "Conditional approval — higher rate or collateral required"],
        ["High-risk rejected (P4)",
         f"{bm_ebm['rejected']:,} ({bm_ebm['rejected']/bm_ebm['n_test']:.1%})",
         "Poor creditworthiness — loan declined"],
        ["NPA exposure avoided",
         _fmt_cr(bm_ebm["npa_total"]),
         f"Estimated bad debt prevented (P4 x {p4_rate:.0%} + P3 x {p3_rate:.0%})"],
        ["Residual NPA exposure",
         _fmt_cr(bm_ebm["npa_exposure"]),
         f"P4 applicants incorrectly approved ({bm_ebm['fn_p4']} cases)"],
    ]
    ht = Table(headline_data, colWidths=[4.5*cm, 3.5*cm, 9*cm])
    ht.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#dce6f1"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.grey),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ]))
    story.append(ht)
    story.append(Spacer(1, 0.4*cm))

    # ── Model comparison table ────────────────────────────────────────────────
    story.append(Paragraph("Model Comparison — LR vs EBM", h2_style))
    comp_data = [
        ["Metric", "Logistic Regression", "EBM", "Better"],
        ["Macro F1",
         f"{lr_meta['f1_macro']:.3f}",
         f"{ebm_meta['f1_macro']:.3f}",
         "EBM" if ebm_meta["f1_macro"] > lr_meta["f1_macro"] else "LR"],
        ["AUC (OvR)",
         f"{lr_meta['auc_ovr']:.3f}",
         f"{ebm_meta['auc_ovr']:.3f}",
         "EBM" if ebm_meta["auc_ovr"] > lr_meta["auc_ovr"] else "LR"],
        ["Overall Accuracy",
         f"{bm_lr['accuracy']:.1%}",
         f"{bm_ebm['accuracy']:.1%}",
         "EBM" if bm_ebm["accuracy"] > bm_lr["accuracy"] else "LR"],
        ["NPA Avoided",
         _fmt_cr(bm_lr["npa_total"]),
         _fmt_cr(bm_ebm["npa_total"]),
         "EBM" if bm_ebm["npa_total"] > bm_lr["npa_total"] else "LR"],
        ["Residual NPA Exposure",
         _fmt_cr(bm_lr["npa_exposure"]),
         _fmt_cr(bm_ebm["npa_exposure"]),
         "EBM" if bm_ebm["npa_exposure"] < bm_lr["npa_exposure"] else "LR"],
        ["P4 Incorrectly Approved",
         str(bm_lr["fn_p4"]),
         str(bm_ebm["fn_p4"]),
         "EBM" if bm_ebm["fn_p4"] < bm_lr["fn_p4"] else "LR"],
    ]
    ct = Table(comp_data, colWidths=[5*cm, 3.5*cm, 3.5*cm, 5*cm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#dce6f1"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.grey),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ]))
    story.append(ct)
    story.append(Spacer(1, 0.3*cm))

    # ── Footnote ──────────────────────────────────────────────────────────────
    story.append(Paragraph(
        f"Assumptions: Average loan {_fmt_lakh(avg_loan)}, "
        f"P4 default rate {p4_rate:.0%}, P3 default rate {p3_rate:.0%}. "
        f"NPA avoided = correctly classified high-risk applicants x avg loan x default rate. "
        f"Figures are illustrative estimates for a portfolio of this size.",
        fn_style,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Main page ─────────────────────────────────────────────────────────────────

def main():
    st.title("₹ Business Impact Summary")
    st.caption(
        "Model performance translated into credit risk business language. "
        "All ₹ figures are estimates based on configurable assumptions."
    )

    metrics = _get_metrics()
    if metrics is None:
        return

    label_map = _label_map()

    if "lr" not in metrics or "ebm" not in metrics:
        st.error("Both LR and EBM metrics required. Re-run **Train Models**.")
        return

    lr_meta  = metrics["lr"]
    ebm_meta = metrics["ebm"]
    n_test   = metrics["n_test"]

    # ── Assumption sliders ────────────────────────────────────────────────────
    st.subheader("Assumptions")
    st.caption("Adjust to match your portfolio. Defaults are conservative industry estimates.")

    col1, col2, col3 = st.columns(3)
    avg_loan = col1.slider(
        "Average loan size (₹)",
        min_value=50_000, max_value=1_000_000,
        value=DEFAULT_AVG_LOAN, step=50_000,
        help="Average disbursed loan amount per applicant",
    )
    p4_rate = col2.slider(
        "P4 default rate",
        min_value=0.10, max_value=0.70,
        value=DEFAULT_P4_DEFAULT, step=0.05,
        format="%.0f%%",
        help="Fraction of P4 applicants expected to default if approved",
    )
    p3_rate = col3.slider(
        "P3 default rate",
        min_value=0.05, max_value=0.40,
        value=DEFAULT_P3_DEFAULT, step=0.05,
        format="%.0f%%",
        help="Fraction of P3 applicants expected to default if approved",
    )

    st.markdown("---")

    # ── Compute ───────────────────────────────────────────────────────────────
    bm_lr  = _compute_business_metrics(
        lr_meta["confusion_matrix"],  lr_meta["y_pred"],
        n_test, avg_loan, p4_rate, p3_rate,
    )
    bm_ebm = _compute_business_metrics(
        ebm_meta["confusion_matrix"], ebm_meta["y_pred"],
        n_test, avg_loan, p4_rate, p3_rate,
    )

    # ── Section 1: Headline KPI cards (EBM) ───────────────────────────────────
    st.subheader("1 — EBM Model: Headline Outcomes")
    st.caption(f"Based on {n_test:,} test-set applicants (20% holdout, stratified)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Correctly Classified",
        f"{bm_ebm['total_correct']:,}",
        f"{bm_ebm['accuracy']:.1%} accuracy",
        help="Applicants placed in the correct credit tier",
    )
    c2.metric(
        "Approved (P1+P2)",
        f"{bm_ebm['approved']:,}",
        f"{bm_ebm['approval_rate']:.1%} approval rate",
        help="Creditworthy applicants correctly approved",
    )
    c3.metric(
        "NPA Exposure Avoided",
        _fmt_cr(bm_ebm["npa_total"]),
        f"{bm_ebm['tp_p4']:,} high-risk correctly rejected",
        help=f"P4 × {p4_rate:.0%} + P3 × {p3_rate:.0%} default rate × avg loan",
    )
    c4.metric(
        "Residual NPA Risk",
        _fmt_cr(bm_ebm["npa_exposure"]),
        f"{bm_ebm['fn_p4']} P4 incorrectly approved",
        delta_color="inverse",
        help="P4 applicants the model incorrectly approved — potential bad debt",
    )

    st.markdown("---")

    # ── Section 2: Risk tier breakdown ────────────────────────────────────────
    st.subheader("2 — Risk Tier Distribution")
    st.caption("How each model segments the test-set applicant population")

    left, right = st.columns(2)
    with left:
        fig_d_lr = _plot_tier_donut(bm_lr, "Logistic Regression")
        st.pyplot(fig_d_lr)
        plt.close(fig_d_lr)
    with right:
        fig_d_ebm = _plot_tier_donut(bm_ebm, "EBM")
        st.pyplot(fig_d_ebm)
        plt.close(fig_d_ebm)

    # Tier breakdown table
    tier_rows = []
    for tier, cls_int in [("P1", 0), ("P2", 1), ("P3", 2), ("P4", 3)]:
        lr_n  = int((lr_meta["y_pred"]  == cls_int).sum())
        ebm_n = int((ebm_meta["y_pred"] == cls_int).sum())
        tier_rows.append({
            "Tier":        tier,
            "Description": TIER_DESCRIPTIONS[tier],
            "Action":      TIER_ACTIONS[tier],
            "LR Count":    lr_n,
            "LR %":        f"{lr_n / n_test:.1%}",
            "EBM Count":   ebm_n,
            "EBM %":       f"{ebm_n / n_test:.1%}",
        })
    tier_df = pd.DataFrame(tier_rows).set_index("Tier")
    st.dataframe(tier_df, use_container_width=True)

    st.markdown("---")

    # ── Section 3: NPA impact comparison ──────────────────────────────────────
    st.subheader("3 — NPA Impact: LR vs EBM")
    st.caption(
        f"NPA avoided = correctly classified high-risk applicants × "
        f"avg loan ({_fmt_lakh(avg_loan)}) × default rate. "
        f"Residual exposure = incorrectly approved P4 applicants."
    )

    fig_npa = _plot_npa_bar(bm_lr, bm_ebm)
    st.pyplot(fig_npa)
    plt.close(fig_npa)

    st.markdown("---")

    # ── Section 4: Correct classifications per tier ───────────────────────────
    st.subheader("4 — Correct Classifications by Tier")
    st.caption("How many applicants in each tier were correctly identified by each model")

    fig_acc = _plot_correct_by_tier(bm_lr, bm_ebm)
    st.pyplot(fig_acc)
    plt.close(fig_acc)

    st.markdown("---")

    # ── Section 5: Full comparison table ──────────────────────────────────────
    st.subheader("5 — Full Model Comparison")

    comp_rows = [
        {"Metric": "Macro F1",
         "Logistic Regression": f"{lr_meta['f1_macro']:.3f}",
         "EBM": f"{ebm_meta['f1_macro']:.3f}"},
        {"Metric": "Weighted F1",
         "Logistic Regression": f"{lr_meta['f1_weighted']:.3f}",
         "EBM": f"{ebm_meta['f1_weighted']:.3f}"},
        {"Metric": "AUC (OvR)",
         "Logistic Regression": f"{lr_meta['auc_ovr']:.3f}",
         "EBM": f"{ebm_meta['auc_ovr']:.3f}"},
        {"Metric": "Overall Accuracy",
         "Logistic Regression": f"{bm_lr['accuracy']:.1%}",
         "EBM": f"{bm_ebm['accuracy']:.1%}"},
        {"Metric": "Correctly Classified",
         "Logistic Regression": f"{bm_lr['total_correct']:,}",
         "EBM": f"{bm_ebm['total_correct']:,}"},
        {"Metric": "Approved (P1+P2)",
         "Logistic Regression": f"{bm_lr['approved']:,} ({bm_lr['approval_rate']:.1%})",
         "EBM": f"{bm_ebm['approved']:,} ({bm_ebm['approval_rate']:.1%})"},
        {"Metric": "NPA Avoided",
         "Logistic Regression": _fmt_cr(bm_lr["npa_total"]),
         "EBM": _fmt_cr(bm_ebm["npa_total"])},
        {"Metric": "Residual NPA Exposure",
         "Logistic Regression": _fmt_cr(bm_lr["npa_exposure"]),
         "EBM": _fmt_cr(bm_ebm["npa_exposure"])},
        {"Metric": "P4 Incorrectly Approved",
         "Logistic Regression": str(bm_lr["fn_p4"]),
         "EBM": str(bm_ebm["fn_p4"])},
    ]
    comp_df = pd.DataFrame(comp_rows).set_index("Metric")
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("---")

    # ── Section 6: PDF export ─────────────────────────────────────────────────
    st.subheader("6 — Export PDF Report")
    st.caption(
        "One-page business summary suitable for a credit committee "
        "or management presentation."
    )

    if st.button("Generate PDF Business Summary"):
        with st.spinner("Building PDF..."):
            try:
                pdf_bytes = _build_pdf(
                    bm_lr, bm_ebm, lr_meta, ebm_meta,
                    avg_loan, p4_rate, p3_rate,
                )
                st.download_button(
                    label="⬇ Download Business Summary PDF",
                    data=pdf_bytes,
                    file_name="credit_scoring_business_summary.pdf",
                    mime="application/pdf",
                )
                st.success("PDF ready for download.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    st.info("✅ Pipeline complete — all 8 pages are active.")


if __name__ == "__main__":
    main()
