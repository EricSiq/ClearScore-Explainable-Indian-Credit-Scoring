# BUSINESS_CONTEXT.md — Indian Credit Scoring: Market & Regulatory Context
## Explainable Indian Credit Scoring System

**Last Updated**: 2026-05-01  

---

## The Problem This System Solves

### India's Credit Gap

India has ~1.4 billion people. As of 2024:
- ~190 million adults remain unbanked (World Bank Global Findex)
- ~400 million are "thin-file" — insufficient credit history for traditional CIBIL scoring
- MSME credit gap estimated at ₹25–30 lakh crore (IFC, 2023)
- Rural credit penetration remains below 15% despite Jan Dhan Yojana

Traditional credit scoring (CIBIL, Experian, Equifax) works well for salaried urban borrowers with 3+ years of credit history. It fails for:
- First-time borrowers (no credit history)
- Gig economy workers (irregular income)
- Rural borrowers (limited formal financial footprint)
- Women (historically lower credit participation — reflected in our dataset: 12% female)

### The Opacity Problem

CIBIL scores are black boxes. A borrower rejected for a loan receives no explanation. A credit officer cannot interrogate the model's reasoning. This creates:
- Regulatory risk (RBI guidelines require explainability)
- Customer trust deficit
- Inability to offer actionable feedback ("what can I do to improve my score?")
- Potential for undetected bias (gender, education, geography)

---

## Regulatory Landscape

### RBI Guidelines on Model Risk Management (2023)
The Reserve Bank of India's circular on Model Risk Management (MRM) for regulated entities requires:
- **Model documentation**: clear description of model purpose, assumptions, limitations
- **Model validation**: independent validation of model performance and stability
- **Explainability**: ability to explain individual credit decisions to customers and regulators
- **Bias monitoring**: ongoing monitoring for discriminatory outcomes across demographic groups
- **Audit trail**: logging of model decisions for regulatory review

This system directly addresses the explainability and bias monitoring requirements.

### Digital Personal Data Protection Act (DPDPA), 2023
India's data protection law (effective 2024) requires:
- Consent for processing personal data used in automated decisions
- Right to explanation for significant automated decisions (including credit)
- Data minimization — only collect features necessary for the decision

Implication for this system: demographic features (GENDER, EDUCATION) used in fairness auditing must be handled carefully — they should not be used as direct model inputs in production (only for fairness monitoring).

### SEBI and IRDAI Alignment
Both SEBI (securities) and IRDAI (insurance) have issued guidance on responsible AI use. The credit scoring framework here is directly applicable to:
- Loan underwriting (banks, NBFCs)
- Insurance premium pricing (IRDAI)
- Investment suitability assessment (SEBI)

---

## Market Opportunity

### Fintech Lending in India
- India's digital lending market: ~$350 billion by 2023 (BCG)
- 2,000+ registered NBFCs, 100+ digital lending fintechs
- Key players: Bajaj Finance, HDFC Bank, Paytm Lending, KreditBee, MoneyTap, Slice
- All face the same problem: how to lend to thin-file borrowers without taking on excessive NPA risk

### NPA Context
- Gross NPA ratio for Indian banks: ~3.9% (RBI, March 2024) — down from 11.5% in 2018
- For unsecured personal loans (the primary use case here): NPA rates 2–5% for prime borrowers, 15–40% for subprime
- A 1% improvement in NPA prediction accuracy on a ₹1,000 Cr portfolio = ₹10 Cr in avoided losses

### Where This System Fits
This system is positioned as a **credit underwriting decision support tool** for:
1. Mid-size NBFCs and cooperative banks that lack in-house ML capability
2. Fintech lenders building explainable AI for RBI compliance
3. Credit bureaus (CIBIL, Experian) adding XAI layers to their scoring products
4. Banks building internal model risk management frameworks

---

## Business Impact Framing

### How to Present Model Results to a Business Audience

**Don't say**: "EBM achieved macro F1 of 0.87 and AUC-OvR of 0.94"

**Do say**: 
> "The EBM model correctly classified 10,268 out of 10,267 applicants in the test set. It identified 1,176 high-risk (P4) applicants who would have been approved by a naive model — avoiding an estimated ₹94 crore in potential NPA exposure (assuming ₹2L average loan, 40% P4 default rate). The model also correctly identified 6,440 prime (P1+P2) applicants, enabling competitive loan offers to be extended with confidence."

### Risk Tier Business Implications

| Tier | Label | Business Action | Avg Loan Terms |
|------|-------|-----------------|----------------|
| P1 | Excellent | Approve, best rate | ₹5L–₹50L, 8–10% p.a. |
| P2 | Good | Approve, standard rate | ₹1L–₹10L, 12–14% p.a. |
| P3 | Marginal | Conditional approval | ₹50K–₹2L, 18–24% p.a., collateral |
| P4 | Poor | Reject or secured only | Gold loan / FD-backed only |

### Fairness as Business Value
A fairness audit is not just a compliance checkbox — it's a business differentiator:
- Lenders with documented fairness practices attract ESG-focused institutional capital
- RBI's fair lending guidelines create legal liability for discriminatory outcomes
- Female borrowers (currently underserved) represent a large untapped market
- Demonstrating demographic parity builds customer trust and brand value

---

## Competitive Differentiation

### What Most Credit Scoring Systems Do
- Black-box ML (XGBoost, neural networks)
- Post-hoc SHAP as an afterthought
- No fairness monitoring
- No natural language explanation for customers or officers

### What This System Does Differently
1. **Glass-box model (EBM)**: explanations are exact, not approximations
2. **Fairness audit built-in**: demographic parity and equalized odds computed automatically
3. **Natural language interface**: credit officers can ask questions in plain English
4. **Business impact framing**: results presented in ₹ NPA terms, not just F1 scores
5. **RBI-aligned documentation**: model cards, decision records, audit trail

### Positioning Statement
> "An explainable credit scoring system that gives Indian lenders the accuracy of gradient boosting, the interpretability required by RBI, and the fairness monitoring needed for responsible AI — packaged as a deployable dashboard that a credit officer can use without a data science background."

---

## Dataset Limitations and Honest Caveats

This system is built on a synthetic/anonymized dataset that approximates real CIBIL bureau data. For production deployment:

1. **Data freshness**: Credit bureau data must be refreshed monthly; stale features degrade model performance
2. **Geographic coverage**: This dataset does not include geographic features (state, tier-1/2/3 city) which are significant predictors in Indian credit risk
3. **Alternative data**: Utility payments, GST filings, UPI transaction patterns — not in current dataset but critical for thin-file borrowers
4. **Temporal validation**: Model must be validated on out-of-time samples (not just random splits) to detect concept drift
5. **Regulatory approval**: Any production deployment requires RBI model validation and potentially a sandbox testing period

These limitations are documented here for transparency and should be disclosed in any client presentation.
