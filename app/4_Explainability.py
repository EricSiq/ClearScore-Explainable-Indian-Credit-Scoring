"""EBM-native, SHAP, and LIME explanations with explicit compute boundaries."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.components.model_loader import load_model
from app.components.utils import get_X_y, get_label_map
from app.components.xai import lime_explanation, shap_values


def _ebm_global(ebm):
    explanation = ebm.explain_global()
    data = explanation.data()
    names, scores = data.get("names", []), data.get("scores", [])
    order = np.argsort(scores)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(names)[order][::-1], np.array(scores)[order][::-1], color="#0f766e")
    ax.set_xlabel("EBM importance")
    ax.set_title("EBM global feature importance")
    fig.tight_layout()
    return fig


def main():
    st.title("Explainability workbench")
    st.caption("EBM shape-function evidence, SHAP attribution, and LIME local approximation.")
    X, _ = get_X_y()
    ebm = load_model("ebm_model.pkl", "ebm_model")
    if X is None or ebm is None:
        st.error("Prepare data and train the EBM before opening the explainability workbench.")
        return
    labels = get_label_map()
    st.subheader("1. Inherent EBM explanation")
    st.write("EBM is the primary glass-box model. Its learned additive shape functions are part of the model, not a post-hoc approximation.")
    fig = _ebm_global(ebm)
    st.pyplot(fig); plt.close(fig)

    row_index = int(st.number_input("Applicant row", 0, len(X) - 1, 0, 1))
    row = X.iloc[[row_index]]
    predicted = int(ebm.predict(row)[0])
    probabilities = ebm.predict_proba(row)[0]
    st.metric("EBM predicted tier", labels[predicted])
    st.bar_chart({labels[int(class_id)]: float(probabilities[position]) for position, class_id in enumerate(ebm.classes_)})

    st.subheader("2. SHAP attribution")
    st.caption("SHAP runs only when requested. Community Cloud uses 50 rows and 20 background rows, cached for the session; use local full validation for publication-grade analysis.")
    if st.button("Compute bounded SHAP sample"):
        try:
            sample, values = shap_values(ebm, X, "ebm")
            import shap
            shap.summary_plot(values, sample, max_display=15, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close("all")
        except Exception as error:
            st.error(f"SHAP computation failed: {error}")

    st.subheader("3. LIME local approximation")
    st.caption("LIME approximates the EBM decision around the selected applicant using a capped 1,000-row background sample.")
    if st.button("Compute LIME explanation"):
        try:
            explanation = lime_explanation(ebm, X, row_index, [labels[i] for i in sorted(labels)])
            class_position = list(ebm.classes_).index(predicted)
            pairs = explanation.as_list(label=class_position)
            st.dataframe({"local condition": [name for name, _ in pairs], "weight": [weight for _, weight in pairs]}, use_container_width=True)
        except Exception as error:
            st.error(f"LIME computation failed: {error}")
    st.warning("Use all three views together: EBM provides the model-native explanation; SHAP and LIME are complementary attribution/approximation methods. None of them is a customer-facing adverse-action notice without governed reason-code mapping.")


main()
