import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

def lime_local_explanation(model, X, y, row_index, num_features=10):
    """
    Render LIME local explanation for a given row index.
    Shows HTML explanation inside Streamlit.
    """

    try:
        X_np = X.values
        feature_names = X.columns.tolist()

        # Determine classification or regression mode
        if hasattr(model, "predict_proba"):
            mode = "classification"
            class_names = ["Reject", "Approve"]
        else:
            mode = "regression"
            class_names = None

        explainer = LimeTabularExplainer(
            X_np,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            discretize_continuous=True,
        )

        instance = X_np[int(row_index)]

        # Use predict_proba when available
        if mode == "classification":
            exp = explainer.explain_instance(instance, model.predict_proba, num_features=num_features)
        else:
            exp = explainer.explain_instance(instance, model.predict, num_features=num_features)

        html = exp.as_html()
        st.components.v1.html(html, height=420, scrolling=True)

    except Exception as e:
        st.error(f"❌ LIME explanation failed: {e}")
