"""Deterministic analyst decision-review page; deliberately no LLM runtime."""

import streamlit as st

from app.components.linear_explanations import local_contributions
from app.components.model_loader import load_model
from app.components.utils import get_X_y, get_label_map


def main():
    st.title("Decision review")
    st.caption("Evidence first: inspect one model output and its exact contribution record.")
    X, _ = get_X_y()
    model = load_model("logistic_regression.pkl", "lr_model")
    if X is None or model is None:
        st.error("Prepare data and run validation before opening decision review.")
        return

    labels = get_label_map()
    idx = int(st.number_input("Applicant row", 0, len(X) - 1, 0, 1))
    row = X.iloc[[idx]]
    predicted = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    confidence = float(probabilities[list(model.classes_).index(predicted)])
    st.subheader(f"Model output: {labels[predicted]}")
    st.write(f"Selected-class probability: **{confidence:.1%}**. Treat this as model confidence, not the probability of repayment.")
    st.dataframe(
        local_contributions(model, row, predicted).head(8).style.format(
            {"value": "{:.3f}", "contribution": "{:+.3f}"}
        ),
        use_container_width=True,
    )
    st.info("Review prompt: corroborate material features, check data quality, and apply the lender's approved policy outside this prototype. This app does not make autonomous credit decisions.")
    with st.expander("Why there is no chat agent"):
        st.write("A language model can make an explanation sound plausible without adding evidence. This deployment keeps review deterministic: every displayed factor is traceable to a model coefficient and a transformed input.")


main()
