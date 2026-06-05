import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER

from app.components.utils import get_label_map

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

DEFAULT_AVG_LOAN   = 200_000
DEFAULT_P4_DEFAULT = 0.40
DEFAULT_P3_DEFAULT = 0.15


def _label_map() -> dict:
    return get_label_map()


def _get_metrics():
    if "model_metrics" not in st.session_state:
        st.error("No model metrics found. Run Train Models first.")
        return None
    return st.session_state["model_metrics"]


def _fmt_cr(rupees: float) -> str:
    return f"Rs {rupees / 1e7:,.2f} Cr"


def _fmt_lakh(rupees: float) -> str:
    return f"Rs {rupees / 1e5:,.0f}L"


def _compute_business_metrics(cm, y_pred, n_test, avg_loan, p4_rate, p3_rate):
    approved = int((y_pred <= 1).sum())
    marginal = int((y_pred == 2).sum())
    rejected = int((y_pred == 3).sum())
    tp_p1, tp_p2, tp_p3, tp_p4 = int(cm[0,0]), int(cm[1,1]), int(cm[2,2]), int(cm[3,3])
    total_correct = tp_p1 + tp_p2 + tp_p3 + tp_p4
    npa_p4    = tp_p4 * avg_loan * p4_rate
    npa_p3    = tp_p3 * avg_loan * p3_rate
    npa_total = npa_p4 + npa_p3
    fn_p4     = int(cm[3, :2].sum())
    return {
        "n_test": n_test, "approved": approved, "marginal": marginal, "rejected": rejected,
        "approval_rate": approved / n_test,
        "tp_p1": tp_p1, "tp_p2": tp_p2, "tp_p3": tp_p3, "tp_p4": tp_p4,
        "total_correct": total_correct, "accuracy": total_correct / n_test,
        "npa_p4": npa_p4, "npa_p3": npa_p3, "npa_total": npa_total,
        "fn_p4": fn_p4, "npa_exposure": fn_p4 * avg_loan * p4_rate,
    }


def _plot_tier_donut(bm: dict, title: str) -> plt.Figure:
    sizes  = [bm["approved"], bm["marginal"], bm["rejected"]]
    labels = ["Approved (P1+P2)", "Marginal (P3)", "Rejected (P4)"]
    clrs   = ["#2ecc71", "#f39c12", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=clrs, autopct="%1.1f%%",
                                      startangle=90, wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
                                      textprops={"fontsize": 9})
    for at in autotexts:
        at.set_fontsize(8); at.set_fontweight("bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def _plot_npa_bar(bm_lr: dict, bm_ebm: dict) -> plt.Figure:
    models   = ["Logistic Regression", "EBM"]
    avoided  = [bm_lr["npa_total"] / 1e7,   bm_ebm["npa_total"] / 1e7]
    exposure = [bm_lr["npa_exposure"] / 1e7, bm_ebm["npa_exposure"] / 1e7]
    x = np.arange(len(models)); width = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 4))
    b1 = ax.bar(x - width/2, avoided,  width, label="NPA Avoided (Rs Cr)",  color="#2ecc71", alpha=0.9)
    b2 = ax.bar(x + width/2, exposure, width, label="Residual NPA (Rs Cr)", color="#e74c3c", alpha=0.9)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Rs Crore", fontsize=10)
    ax.set_title("NPA Impact Comparison — LR vs EBM", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def _plot_correct_by_tier(bm_lr: dict, bm_ebm: dict) -> plt.Figure:
    tiers  = ["P1", "P2", "P3", "P4"]
    lr_tp  = [bm_lr[f"tp_{t.lower()}"]  for t in tiers]
    ebm_tp = [bm_ebm[f"tp_{t.lower()}"] for t in tiers]
    x = np.arange(len(tiers)); width = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x - width/2, lr_tp,  width, label="Logistic Regression", color="#3498db", alpha=0.85)
    ax.bar(x + width/2, ebm_tp, width, label="EBM",                 color="#2ecc71", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(tiers)
    ax.set_ylabel("Correctly classified", fontsize=10)
    ax.set_title("Correct Classifications by Tier", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def _build_pdf(bm_lr, bm_ebm, lr_meta, ebm_meta, avg_loan, p4_rate, p3_rate) -> bytes:
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=15, spaceAfter=4, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("Sub", parent=styles["BodyText"], fontSize=9, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#555555"))
    h2_style    = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=4)
    fn_style    = ParagraphStyle("Footnote", parent=styles["BodyText"], fontSize=8,
                                 textColor=colors.HexColor("#555555"))

    story.append(Paragraph("Credit Scoring Model — Business Impact Summary", title_style))
    story.append(Paragraph(
        f"Test set: {bm_ebm['n_test']:,} applicants  |  Avg loan: {_fmt_lakh(avg_loan)}  |  "
        f"P4 default rate: {p4_rate:.0%}  |  P3 default rate: {p3_rate:.0%}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1f4e79"), spaceAfter=6))

    story.append(Paragraph("EBM Model — Headline Outcomes", h2_style))
    headline_data = [
        ["Metric", "Value", "Business Meaning"],
        ["Applicants scored",       f"{bm_ebm['n_test']:,}",                              "Total test-set applicants evaluated"],
        ["Correctly classified",    f"{bm_ebm['total_correct']:,} ({bm_ebm['accuracy']:.1%})", "Applicants placed in the correct credit tier"],
        ["Approved (P1+P2)",        f"{bm_ebm['approved']:,} ({bm_ebm['approval_rate']:.1%})", "Creditworthy applicants approved"],
        ["Marginal flagged (P3)",   f"{bm_ebm['marginal']:,} ({bm_ebm['marginal']/bm_ebm['n_test']:.1%})", "Conditional approval"],
        ["High-risk rejected (P4)", f"{bm_ebm['rejected']:,} ({bm_ebm['rejected']/bm_ebm['n_test']:.1%})", "Loan declined"],
        ["NPA exposure avoided",    _fmt_cr(bm_ebm["npa_total"]),                          f"P4 x {p4_rate:.0%} + P3 x {p3_rate:.0%}"],
        ["Residual NPA exposure",   _fmt_cr(bm_ebm["npa_exposure"]),                       f"P4 incorrectly approved ({bm_ebm['fn_p4']} cases)"],
    ]
    ht = Table(headline_data, colWidths=[4.5*cm, 3.5*cm, 9*cm])
    ht.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
        ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#dce6f1"), colors.white]),
        ("GRID",           (0,0), (-1,-1), 0.4, colors.grey),
        ("TOPPADDING",     (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
    ]))
    story.append(ht)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Model Comparison — LR vs EBM", h2_style))
    comp_data = [
        ["Metric", "Logistic Regression", "EBM", "Better"],
        ["Macro F1",            f"{lr_meta['f1_macro']:.3f}",  f"{ebm_meta['f1_macro']:.3f}",  "EBM" if ebm_meta["f1_macro"] > lr_meta["f1_macro"] else "LR"],
        ["AUC (OvR)",           f"{lr_meta['auc_ovr']:.3f}",   f"{ebm_meta['auc_ovr']:.3f}",   "EBM" if ebm_meta["auc_ovr"] > lr_meta["auc_ovr"] else "LR"],
        ["Overall Accuracy",    f"{bm_lr['accuracy']:.1%}",    f"{bm_ebm['accuracy']:.1%}",    "EBM" if bm_ebm["accuracy"] > bm_lr["accuracy"] else "LR"],
        ["NPA Avoided",         _fmt_cr(bm_lr["npa_total"]),   _fmt_cr(bm_ebm["npa_total"]),   "EBM" if bm_ebm["npa_total"] > bm_lr["npa_total"] else "LR"],
        ["Residual NPA",        _fmt_cr(bm_lr["npa_exposure"]),_fmt_cr(bm_ebm["npa_exposure"]),"EBM" if bm_ebm["npa_exposure"] < bm_lr["npa_exposure"] else "LR"],
        ["P4 Incorrectly Approved", str(bm_lr["fn_p4"]),       str(bm_ebm["fn_p4"]),           "EBM" if bm_ebm["fn_p4"] < bm_lr["fn_p4"] else "LR"],
    ]
    ct = Table(comp_data, colWidths=[5*cm, 3.5*cm, 3.5*cm, 5*cm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
        ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#dce6f1"), colors.white]),
        ("GRID",           (0,0), (-1,-1), 0.4, colors.grey),
        ("TOPPADDING",     (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
    ]))
    story.append(ct)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Assumptions: Average loan {_fmt_lakh(avg_loan)}, P4 default rate {p4_rate:.0%}, "
        f"P3 default rate {p3_rate:.0%}. Figures are illustrative estimates.", fn_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def main():
    st.title("Business Impact Summary")
    st.caption("CreditLens · Model performance in credit risk business language")

    metrics = _get_metrics()
    if metrics is None:
        return

    label_map = _label_map()

    if "lr" not in metrics or "ebm" not in metrics:
        st.error("Both LR and EBM metrics required. Re-run Train Models.")
        return

    lr_meta  = metrics["lr"]
    ebm_meta = metrics["ebm"]
    n_test   = metrics["n_test"]

    st.subheader("Assumptions")
    st.caption("Adjust to match your portfolio.")
    with st.expander("Where these defaults come from"):
        st.markdown(
            "- **P4 default rate 40%**: Conservative estimate for subprime unsecured personal "
            "loans in India. CRISIL data and RBI Financial Stability Reports (2022–2024) show "
            "30–50% historical NPA roll-rates for this segment.\n"
            "- **P3 default rate 15%**: Sub-standard (watch-list) loans show 10–20% roll-rate "
            "to NPA. P3 applicants are offered conditional approval (higher rate, lower limit) "
            "which partially mitigates this risk.\n"
            "- **Average loan Rs 2L**: Median unsecured personal loan ticket size for mid-market "
            "NBFCs and digital lenders in India (2023–2024 RBI data)."
        )
    col1, col2, col3 = st.columns(3)
    avg_loan = col1.slider("Average loan size (Rs)", min_value=50_000, max_value=1_000_000,
                           value=DEFAULT_AVG_LOAN, step=50_000)
    p4_rate  = col2.slider("P4 default rate", min_value=0.10, max_value=0.70,
                           value=DEFAULT_P4_DEFAULT, step=0.05, format="%.0f%%")
    p3_rate  = col3.slider("P3 default rate", min_value=0.05, max_value=0.40,
                           value=DEFAULT_P3_DEFAULT, step=0.05, format="%.0f%%")

    st.markdown("---")

    bm_lr  = _compute_business_metrics(lr_meta["confusion_matrix"],  lr_meta["y_pred"],  n_test, avg_loan, p4_rate, p3_rate)
    bm_ebm = _compute_business_metrics(ebm_meta["confusion_matrix"], ebm_meta["y_pred"], n_test, avg_loan, p4_rate, p3_rate)

    # ── Model recommendation ──────────────────────────────────────────────────
    ebm_wins = sum([
        bm_ebm["accuracy"]     > bm_lr["accuracy"],
        bm_ebm["npa_total"]    >= bm_lr["npa_total"],
        bm_ebm["npa_exposure"] <= bm_lr["npa_exposure"],
        ebm_meta["f1_macro"]   > lr_meta["f1_macro"],
        ebm_meta["auc_ovr"]    > lr_meta["auc_ovr"],
    ])
    if ebm_wins >= 4:
        st.success(
            f"**Recommended model: EBM.** "
            f"Outperforms Logistic Regression on {ebm_wins}/5 business metrics — "
            f"accuracy ({bm_ebm['accuracy']:.1%} vs {bm_lr['accuracy']:.1%}), "
            f"NPA avoided ({_fmt_cr(bm_ebm['npa_total'])} vs {_fmt_cr(bm_lr['npa_total'])}), "
            f"and AUC ({ebm_meta['auc_ovr']:.3f} vs {lr_meta['auc_ovr']:.3f}). "
            f"EBM is also preferred for regulatory compliance — its shape functions provide "
            f"exact, auditable explanations that satisfy RBI MRM documentation requirements "
            f"without relying on post-hoc SHAP approximations."
        )
    else:
        st.info(
            f"**Models are comparable** on this sample. "
            f"EBM won {ebm_wins}/5 metrics. Consider full 100% training set for a definitive comparison."
        )

    st.subheader("1 — EBM Model: Headline Outcomes")
    st.caption(f"Based on {n_test:,} test-set applicants (20% holdout, stratified)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Correctly Classified", f"{bm_ebm['total_correct']:,}", f"{bm_ebm['accuracy']:.1%} accuracy")
    c2.metric("Approved (P1+P2)",     f"{bm_ebm['approved']:,}",     f"{bm_ebm['approval_rate']:.1%} approval rate")
    c3.metric("NPA Exposure Avoided", _fmt_cr(bm_ebm["npa_total"]),  f"{bm_ebm['tp_p4']:,} high-risk correctly rejected")
    c4.metric("Residual NPA Risk",    _fmt_cr(bm_ebm["npa_exposure"]), f"{bm_ebm['fn_p4']} P4 incorrectly approved",
              delta_color="inverse")

    st.markdown("---")
    st.subheader("2 — Risk Tier Distribution")
    left, right = st.columns(2)
    with left:
        fig_d_lr = _plot_tier_donut(bm_lr, "Logistic Regression")
        st.pyplot(fig_d_lr); plt.close(fig_d_lr)
    with right:
        fig_d_ebm = _plot_tier_donut(bm_ebm, "EBM")
        st.pyplot(fig_d_ebm); plt.close(fig_d_ebm)

    tier_rows = []
    for tier, cls_int in [("P1",0),("P2",1),("P3",2),("P4",3)]:
        lr_n  = int((lr_meta["y_pred"]  == cls_int).sum())
        ebm_n = int((ebm_meta["y_pred"] == cls_int).sum())
        tier_rows.append({"Tier": tier, "Description": TIER_DESCRIPTIONS[tier], "Action": TIER_ACTIONS[tier],
                          "LR Count": lr_n, "LR %": f"{lr_n/n_test:.1%}",
                          "EBM Count": ebm_n, "EBM %": f"{ebm_n/n_test:.1%}"})
    st.dataframe(pd.DataFrame(tier_rows).set_index("Tier"), use_container_width=True)

    st.markdown("---")
    st.subheader("3 — NPA Impact: LR vs EBM")
    fig_npa = _plot_npa_bar(bm_lr, bm_ebm)
    st.pyplot(fig_npa); plt.close(fig_npa)

    st.markdown("---")
    st.subheader("4 — Correct Classifications by Tier")
    fig_acc = _plot_correct_by_tier(bm_lr, bm_ebm)
    st.pyplot(fig_acc); plt.close(fig_acc)

    st.markdown("---")
    st.subheader("5 — Full Model Comparison")
    comp_rows = [
        {"Metric": "Macro F1",              "LR": f"{lr_meta['f1_macro']:.3f}",    "EBM": f"{ebm_meta['f1_macro']:.3f}"},
        {"Metric": "Weighted F1",           "LR": f"{lr_meta['f1_weighted']:.3f}", "EBM": f"{ebm_meta['f1_weighted']:.3f}"},
        {"Metric": "AUC (OvR)",             "LR": f"{lr_meta['auc_ovr']:.3f}",     "EBM": f"{ebm_meta['auc_ovr']:.3f}"},
        {"Metric": "Overall Accuracy",      "LR": f"{bm_lr['accuracy']:.1%}",      "EBM": f"{bm_ebm['accuracy']:.1%}"},
        {"Metric": "Correctly Classified",  "LR": f"{bm_lr['total_correct']:,}",   "EBM": f"{bm_ebm['total_correct']:,}"},
        {"Metric": "Approved (P1+P2)",      "LR": f"{bm_lr['approved']:,} ({bm_lr['approval_rate']:.1%})",
                                            "EBM": f"{bm_ebm['approved']:,} ({bm_ebm['approval_rate']:.1%})"},
        {"Metric": "NPA Avoided",           "LR": _fmt_cr(bm_lr["npa_total"]),     "EBM": _fmt_cr(bm_ebm["npa_total"])},
        {"Metric": "Residual NPA Exposure", "LR": _fmt_cr(bm_lr["npa_exposure"]),  "EBM": _fmt_cr(bm_ebm["npa_exposure"])},
        {"Metric": "P4 Incorrectly Approved","LR": str(bm_lr["fn_p4"]),            "EBM": str(bm_ebm["fn_p4"])},
    ]
    st.dataframe(pd.DataFrame(comp_rows).set_index("Metric"), use_container_width=True)

    st.markdown("---")
    st.subheader("6 — Export PDF Report")
    if st.button("Generate PDF Business Summary"):
        with st.spinner("Building PDF..."):
            try:
                pdf_bytes = _build_pdf(bm_lr, bm_ebm, lr_meta, ebm_meta, avg_loan, p4_rate, p3_rate)
                st.download_button(label="Download Business Summary PDF", data=pdf_bytes,
                                   file_name="credit_scoring_business_summary.pdf", mime="application/pdf")
                st.success("PDF ready for download.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    st.info("Pipeline complete — all 8 pages are active.")


main()
