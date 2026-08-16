"""
XClearScore — Single-Page Dashboard
A professional analyst workbench for credit-risk model review.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Configuration
TARGET_COL = "Approved_Flag"
TARGET_MAP = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
LABEL_MAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
DROP_COLS = ["PROSPECTID"]
CAT_COLS = ["last_prod_enq2", "first_prod_enq2"]
SENSITIVE_COLS = ["MARITALSTATUS", "EDUCATION", "GENDER"]
DATA_DIR = Path("Datasets")

# Model directory
_TMP_MODELS = os.path.join(tempfile.gettempdir(), "xclearscore_models")
MODEL_DIR = _TMP_MODELS if not os.access("app/models", os.W_OK) else "app/models"

# Tier colors (consistent across entire product)
TIER_COLORS = {
    "P1": "#0F766E",  # Teal
    "P2": "#2563EB",  # Blue
    "P3": "#B45309",  # Amber
    "P4": "#B91C1C",  # Red
}

# Design tokens
COLORS = {
    "ink": "#172033",
    "navy": "#0B1F3A",
    "teal": "#0F766E",
    "blue": "#2563EB",
    "amber": "#B45309",
    "red": "#B91C1C",
    "surface": "#FBFCFE",
    "panel": "#FFFFFF",
    "border": "#D8E0EA",
    "muted": "#64748B",
}


# ============================================================================
# CSS STYLING
# ============================================================================

def load_custom_css():
    """Load custom CSS for the dashboard design system."""
    st.markdown(
        """
        <style>
        /* Reset and base styles */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 1440px;
        }
        
        /* Header styling */
        .dashboard-header {
            background: linear-gradient(135deg, #0B1F3A 0%, #172033 100%);
            padding: 1.5rem 2rem;
            margin: -1rem -1rem 1.5rem -1rem;
            border-radius: 0 0 12px 12px;
        }
        
        .dashboard-header h1 {
            color: #FBFCFE;
            font-size: 28px;
            font-weight: 600;
            margin: 0;
            line-height: 34px;
        }
        
        .dashboard-header .subtitle {
            color: #D8E0EA;
            font-size: 14px;
            margin-top: 4px;
        }
        
        .dashboard-header .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 500;
            margin-left: 12px;
        }
        
        .status-ready {
            background: rgba(15, 118, 110, 0.2);
            color: #0F766E;
            border: 1px solid rgba(15, 118, 110, 0.3);
        }
        
        .status-training {
            background: rgba(37, 99, 235, 0.2);
            color: #2563EB;
            border: 1px solid rgba(37, 99, 235, 0.3);
        }
        
        .status-needs-data {
            background: rgba(180, 83, 9, 0.2);
            color: #B45309;
            border: 1px solid rgba(180, 83, 9, 0.3);
        }
        
        /* Card styling */
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #D8E0EA;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
        }
        
        .metric-card .label {
            font-size: 12px;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        
        .metric-card .value {
            font-size: 28px;
            font-weight: 600;
            color: #172033;
            font-variant-numeric: tabular-nums;
        }
        
        .metric-card .context {
            font-size: 12px;
            color: #64748B;
            margin-top: 4px;
        }
        
        /* Section headers */
        .section-header {
            font-size: 16px;
            font-weight: 600;
            color: #172033;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #D8E0EA;
        }
        
        /* Run control rail */
        .run-control-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            background: #FBFCFE;
            border: 1px solid #D8E0EA;
        }
        
        .run-control-item.active {
            background: #FFFFFF;
            border-color: #0F766E;
        }
        
        .run-control-item .step-number {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 600;
        }
        
        .step-complete {
            background: #0F766E;
            color: white;
        }
        
        .step-pending {
            background: #D8E0EA;
            color: #64748B;
        }
        
        /* Applicant panel */
        .applicant-header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 16px;
        }
        
        .tier-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 4px;
        }
        
        .status-low-gap {
            background: rgba(15, 118, 110, 0.1);
            color: #0F766E;
        }
        
        .status-review {
            background: rgba(180, 83, 9, 0.1);
            color: #B45309;
        }
        
        .status-investigate {
            background: rgba(185, 28, 28, 0.1);
            color: #B91C1C;
        }
        
        /* Governance warning */
        .governance-warning {
            background: rgba(180, 83, 9, 0.05);
            border-left: 3px solid #B45309;
            padding: 12px 16px;
            margin-top: 16px;
            border-radius: 0 8px 8px 0;
        }
        
        .governance-warning p {
            margin: 0;
            font-size: 13px;
            color: #172033;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Better dataframes */
        .stDataFrame {
            border: 1px solid #D8E0EA;
            border-radius: 8px;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div {
            background: #0F766E;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# DATA HELPERS
# ============================================================================

@st.cache_data(show_spinner=False)
def load_demo_data():
    """Load bundled demo datasets."""
    try:
        return (
            pd.read_excel(DATA_DIR / "Internal_Bank_Dataset.xlsx"),
            pd.read_excel(DATA_DIR / "External_Cibil_Dataset.xlsx"),
            pd.read_excel(DATA_DIR / "Unseen_Dataset.xlsx"),
        )
    except FileNotFoundError:
        return None, None, None


def merge_datasets(df_int: pd.DataFrame, df_ext: pd.DataFrame) -> pd.DataFrame:
    """Merge internal and external datasets on PROSPECTID."""
    key_a, key_b = "PROSPECTID", "PROSPECTID"
    if key_a in df_int.columns and key_b in df_ext.columns:
        df = pd.merge(df_int, df_ext, left_on=key_a, right_on=key_b, how="inner", suffixes=("_int", "_ext"))
        if key_b in df.columns and key_a != key_b:
            df.drop(columns=[key_b], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    # Fallback: concat by index
    if df_int.shape[0] == df_ext.shape[0]:
        df = pd.concat([df_int.reset_index(drop=True), df_ext.reset_index(drop=True)], axis=1)
        df.reset_index(drop=True, inplace=True)
        return df
    return None


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_data(df: pd.DataFrame):
    """Run preprocessing pipeline."""
    if TARGET_COL not in df.columns:
        return None, None, None, None
    
    cols_to_drop = [c for c in DROP_COLS + SENSITIVE_COLS if c in df.columns] + [TARGET_COL]
    X_raw = df.drop(columns=cols_to_drop)
    y_raw = df[TARGET_COL].copy()
    
    cat_cols_present = [c for c in CAT_COLS if c in X_raw.columns]
    num_cols_present = [c for c in X_raw.select_dtypes(include=["int64", "float64"]).columns
                        if c not in cat_cols_present]
    
    preprocessor = build_preprocessor(num_cols_present, cat_cols_present)
    X_transformed = preprocessor.fit_transform(X_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()
    X_processed = pd.DataFrame(X_transformed, columns=feature_names, dtype=np.float32)
    
    y = y_raw.map(TARGET_MAP).dropna().astype(int)
    X_processed = X_processed.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    return X_processed, y, feature_names, preprocessor


def load_model_from_disk(model_filename: str, session_key: str):
    """Load model from session state or disk."""
    if session_key in st.session_state:
        return st.session_state[session_key]
    
    candidates = [
        os.path.join(MODEL_DIR, model_filename),
        os.path.join(_TMP_MODELS, model_filename),
        os.path.join("app/models", model_filename),
    ]
    
    for path in candidates:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.session_state[session_key] = model
                return model
            except Exception:
                pass
    return None


def save_model(model, filename: str):
    """Save model to disk and session state."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_tier_distribution_chart(predictions: np.ndarray):
    """Create horizontal stacked bar for tier distribution."""
    labels = [LABEL_MAP[i] for i in predictions]
    counts = pd.Series(labels).value_counts().reindex(["P1", "P2", "P3", "P4"]).fillna(0)
    total = counts.sum()
    percentages = (counts / total * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(8, 2.5))
    
    # Horizontal stacked bar
    left = 0
    for tier in ["P1", "P2", "P3", "P4"]:
        width = percentages[tier]
        ax.barh(0, width, left=left, color=TIER_COLORS[tier], 
                label=f"{tier}: {int(counts[tier])} ({width}%)", height=0.6)
        # Add label inside bar if wide enough
        if width > 8:
            ax.text(left + width/2, 0, f"{tier}", ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=11)
        left += width
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Percentage of applicants")
    ax.set_yticks([])
    ax.set_title("Tier distribution", loc='left', fontsize=14, fontweight=600, color=COLORS["ink"])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    fig.tight_layout()
    return fig


def create_confidence_scatter(probabilities: np.ndarray, predictions: np.ndarray):
    """Create scatterplot of model confidence vs predicted tier."""
    # Get max probability for each prediction
    max_proba = probabilities.max(axis=1)
    
    # Map predictions to risk score (P4=high risk=4, P1=low risk=1)
    risk_scores = predictions + 1  # 0->1, 1->2, 2->3, 3->4
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create jittered scatter
    np.random.seed(42)
    jitter_y = risk_scores + np.random.uniform(-0.2, 0.2, size=len(risk_scores))
    
    scatter = ax.scatter(max_proba, jitter_y, alpha=0.3, 
                         c=[TIER_COLORS[LABEL_MAP[p]] for p in predictions], 
                         s=20, edgecolors='none')
    
    # Add review band (low confidence)
    ax.axvspan(0, 0.60, alpha=0.1, color=COLORS["amber"], label="Low-confidence review")
    ax.axvline(0.60, color=COLORS["amber"], linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel("Model confidence (max class probability)")
    ax.set_ylabel("Predicted tier")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["P1", "P2", "P3", "P4"])
    ax.set_title("Applicant confidence distribution", loc='left', fontsize=14, 
                fontweight=600, color=COLORS["ink"])
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add text annotation
    ax.text(0.30, 4.3, "Review band", fontsize=10, color=COLORS["amber"], 
            ha='center', style='italic')
    
    fig.tight_layout()
    return fig


def create_ebm_global_importance(ebm):
    """Create EBM global feature importance chart."""
    explanation = ebm.explain_global()
    data = explanation.data()
    names = data.get("names", [])
    scores = data.get("scores", [])
    
    if not names or not scores:
        return None
    
    # Sort and take top 10
    order = np.argsort(scores)[::-1][:10]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(names)[order][::-1], np.array(scores)[order][::-1], color=COLORS["teal"])
    ax.set_xlabel("EBM importance")
    ax.set_title("EBM global feature importance", loc='left', fontsize=14, 
                fontweight=600, color=COLORS["ink"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def create_shap_local_bars(shap_values: np.ndarray, feature_names: list, top_n: int = 5):
    """Create diverging horizontal bar chart for local SHAP values."""
    # Take absolute values and sort
    mean_abs = np.abs(shap_values).mean(axis=0) if shap_values.ndim > 1 else np.abs(shap_values)
    
    if len(mean_abs) > top_n:
        top_indices = np.argsort(mean_abs)[-top_n:]
    else:
        top_indices = np.arange(len(mean_abs))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    features = [feature_names[i] for i in top_indices]
    values = mean_abs[top_indices]
    
    ax.barh(features, values, color=COLORS["teal"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top SHAP contributors", loc='left', fontsize=14, 
                fontweight=600, color=COLORS["ink"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def create_fairness_bars(groups: list, rates: list, sample_sizes: list, attribute: str):
    """Create selection rate bars by group with sample sizes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    bars = ax.bar(range(len(groups)), rates, color=COLORS["teal"])
    
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_ylabel("Selection rate (%)")
    ax.set_title(f"Selection rate by {attribute}", loc='left', fontsize=14, 
                fontweight=600, color=COLORS["ink"])
    ax.set_ylim(0, 100)
    
    # Add sample size labels below bars
    for i, (bar, n) in enumerate(zip(bars, sample_sizes)):
        ax.text(bar.get_x() + bar.get_width()/2, -8, f"n={n:,}", 
               ha='center', va='top', fontsize=9, color=COLORS["muted"])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


# ============================================================================
# RUN CONTROL FUNCTIONS
# ============================================================================

def get_run_status():
    """Determine current run status."""
    if "ebm_model" in st.session_state:
        return {"label": "Ready", "css_class": "status-ready", "icon": "●"}
    elif "processed_df" in st.session_state:
        return {"label": "Needs training", "css_class": "status-needs-data", "icon": "○"}
    elif "internal_df" in st.session_state:
        return {"label": "Needs preprocessing", "css_class": "status-needs-data", "icon": "○"}
    else:
        return {"label": "Needs data", "css_class": "status-needs-data", "icon": "○"}


def load_and_preprocess_demo():
    """Load demo data and run preprocessing."""
    internal, external, unseen = load_demo_data()
    if internal is None:
        st.error("Demo data files not found. Please check the Datasets folder.")
        return
    
    st.session_state["internal_df"] = internal
    st.session_state["external_df"] = external
    st.session_state["unseen_df"] = unseen
    st.session_state["demo_mode"] = True
    
    # Auto-run preprocessing
    run_preprocessing()


def run_preprocessing():
    """Run preprocessing pipeline."""
    if "internal_df" not in st.session_state or "external_df" not in st.session_state:
        st.error("Load data first.")
        return
    
    df_merged = merge_datasets(st.session_state["internal_df"], st.session_state["external_df"])
    if df_merged is None:
        st.error("Failed to merge datasets.")
        return
    
    # Store merged df for fairness analysis (contains sensitive attrs)
    st.session_state["merged_df"] = df_merged
    
    X, y, feature_names, preprocessor = preprocess_data(df_merged)
    if X is None:
        st.error("Preprocessing failed. Check target column.")
        return
    
    # Store processed data
    processed_df = X.copy()
    processed_df[TARGET_COL] = y
    st.session_state["processed_df"] = processed_df
    st.session_state["feature_names"] = feature_names
    st.session_state["preprocessor"] = preprocessor
    st.session_state["label_map"] = LABEL_MAP


def train_model(cloud_mode: bool = True):
    """Train EBM model."""
    df = st.session_state.get("processed_df")
    if df is None:
        st.error("Run preprocessing first.")
        return
    
    features = st.session_state.get("feature_names", [c for c in df if c != TARGET_COL])
    X, y = df[features], df[TARGET_COL]
    
    # Cap for cloud mode
    cap = 10_000 if cloud_mode else len(df)
    
    if len(df) > cap:
        X_fit, _, y_fit, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=42)
    else:
        X_fit, y_fit = X, y
    
    X_train, X_test, y_train, y_test = train_test_split(X_fit, y_fit, test_size=0.2, 
                                                         stratify=y_fit, random_state=42)
    
    # Train LR baseline
    with st.spinner("Training logistic regression baseline..."):
        lr = LogisticRegression(max_iter=800, random_state=42, n_jobs=1).fit(X_train, y_train)
    
    # Train EBM
    with st.spinner("Training EBM..."):
        ebm = ExplainableBoostingClassifier(
            interactions=0, max_bins=64, outer_bags=4, random_state=42, n_jobs=1
        ).fit(X_train, y_train)
    
    # Evaluate
    def evaluate(model, X_test, y_test):
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)
        return {
            "f1_macro": f1_score(y_test, prediction, average="macro"),
            "auc_ovr": roc_auc_score(y_test, probability, multi_class="ovr", average="macro"),
            "y_pred": prediction,
            "y_proba": probability,
        }
    
    lr_metrics = evaluate(lr, X_test, y_test)
    ebm_metrics = evaluate(ebm, X_test, y_test)
    
    # Save to session state
    st.session_state["lr_model"] = lr
    st.session_state["ebm_model"] = ebm
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["model_metrics"] = {"lr": lr_metrics, "ebm": ebm_metrics}
    
    # Save models to disk
    save_model(lr, "lr_model.pkl")
    save_model(ebm, "ebm_model.pkl")


def reset_session():
    """Reset session state."""
    keys_to_remove = [k for k in st.session_state if k not in ["demo_mode"]]
    for key in keys_to_remove:
        del st.session_state[key]


# ============================================================================
# RENDER FUNCTIONS
# ============================================================================

def render_header():
    """Render dashboard header with status pill."""
    status = get_run_status()
    
    st.markdown(
        f"""
        <div class="dashboard-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>XClearScore</h1>
                    <div class="subtitle">Explainable AI Indian Credit Scoring</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span class="status-pill {status['css_class']}">{status['icon']} {status['label']}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_control_rail():
    """Render left rail with run controls."""
    st.markdown('<div class="section-header">Run control</div>', unsafe_allow_html=True)
    
    # Step 1: Data
    data_complete = "internal_df" in st.session_state and "external_df" in st.session_state
    row_count = st.session_state.get('internal_df', pd.DataFrame()).shape[0] if data_complete else 0
    step_class = "step-complete" if data_complete else "step-pending"
    
    st.markdown(
        f"""
        <div class="run-control-item {'active' if data_complete else ''}">
            <div class="step-number {step_class}">{'✓' if data_complete else '1'}</div>
            <div>
                <div style="font-weight: 600; font-size: 13px;">Data loaded</div>
                <div style="font-size: 11px; color: #64748B;">
                    {f"{row_count:,} rows" if data_complete else "Upload or load demo"}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Step 2: Preprocessing
    preproc_complete = "processed_df" in st.session_state
    feature_count = len(st.session_state.get('feature_names', [])) if preproc_complete else 0
    step_class = "step-complete" if preproc_complete else "step-pending"
    
    st.markdown(
        f"""
        <div class="run-control-item {'active' if preproc_complete else ''}">
            <div class="step-number {step_class}">{'✓' if preproc_complete else '2'}</div>
            <div>
                <div style="font-weight: 600; font-size: 13px;">Preprocessing complete</div>
                <div style="font-size: 11px; color: #64748B;">
                    {f"{feature_count} features" if preproc_complete else "Merge, impute, encode, scale"}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Step 3: Training
    train_complete = "ebm_model" in st.session_state
    auc = st.session_state.get('model_metrics', {}).get('ebm', {}).get('auc_ovr', 0) if train_complete else 0
    step_class = "step-complete" if train_complete else "step-pending"
    
    st.markdown(
        f"""
        <div class="run-control-item {'active' if train_complete else ''}">
            <div class="step-number {step_class}">{'✓' if train_complete else '3'}</div>
            <div>
                <div style="font-weight: 600; font-size: 13px;">EBM trained</div>
                <div style="font-size: 11px; color: #64748B;">
                    {f"AUC: {auc:.3f}" if train_complete else "Primary glass-box model"}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Step 4: Explainability
    explain_complete = train_complete
    step_class = "step-complete" if explain_complete else "step-pending"
    
    st.markdown(
        f"""
        <div class="run-control-item {'active' if explain_complete else ''}">
            <div class="step-number {step_class}">{'✓' if explain_complete else '4'}</div>
            <div>
                <div style="font-weight: 600; font-size: 13px;">Explainability ready</div>
                <div style="font-size: 11px; color: #64748B;">
                    {"EBM, SHAP, LIME available" if explain_complete else "Compute on demand"}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    # Action buttons
    if not data_complete:
        if st.button("Load demo data", type="primary", use_container_width=True):
            load_and_preprocess_demo()
            st.rerun()
    
    if data_complete and not preproc_complete:
        if st.button("Run preprocessing", type="primary", use_container_width=True):
            run_preprocessing()
            st.rerun()
    
    if preproc_complete and not train_complete:
        mode = st.radio("Profile", ["Cloud demo (10K)", "Local full"], horizontal=True, label_visibility="collapsed")
        if st.button("Train EBM", type="primary", use_container_width=True):
            train_model(cloud_mode=mode.startswith("Cloud"))
            st.rerun()
    
    if train_complete:
        if st.button("Reset session", use_container_width=True):
            reset_session()
            st.rerun()


def render_portfolio_health():
    """Render portfolio health metrics row."""
    st.markdown('<div class="section-header">Portfolio health</div>', unsafe_allow_html=True)
    
    metrics = st.session_state.get("model_metrics", {}).get("ebm", {})
    predictions = metrics.get("y_pred", np.array([]))
    probabilities = metrics.get("y_proba", np.array([]))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        auc = metrics.get("auc_ovr", 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Macro AUC</div>
                <div class="value">{auc:.3f}</div>
                <div class="context">Current holdout result</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        f1 = metrics.get("f1_macro", 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">F1 Macro</div>
                <div class="value">{f1:.1%}</div>
                <div class="context">Current holdout result</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        if len(predictions) > 0:
            # Count P3/P4 + low confidence cases
            labels = [LABEL_MAP[i] for i in predictions]
            p3_p4_count = sum(1 for l in labels if l in ["P3", "P4"])
            low_conf_count = sum(1 for p in probabilities.max(axis=1) if p < 0.60)
            review_count = max(p3_p4_count, low_conf_count)
        else:
            review_count = 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Review queue</div>
                <div class="value" style="color: #B45309;">{review_count:,}</div>
                <div class="context">P3/P4 + low confidence cases</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col4:
        # Fairness status placeholder
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Fairness monitor</div>
                <div class="value" style="font-size: 18px;">Check below</div>
                <div class="context">Scroll to governance section</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_portfolio_evidence():
    """Render tier distribution and confidence scatter."""
    st.markdown('<div class="section-header">Portfolio evidence</div>', unsafe_allow_html=True)
    
    metrics = st.session_state.get("model_metrics", {}).get("ebm", {})
    predictions = metrics.get("y_pred", np.array([]))
    probabilities = metrics.get("y_proba", np.array([]))
    
    if len(predictions) == 0:
        st.info("Train a model to see portfolio evidence.")
        return
    
    col1, col2 = st.columns([5, 7])
    
    with col1:
        fig = create_tier_distribution_chart(predictions)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        fig = create_confidence_scatter(probabilities, predictions)
        st.pyplot(fig)
        plt.close(fig)


def render_applicant_panel():
    """Render selected applicant evidence panel."""
    st.markdown('<div class="section-header">Selected applicant</div>', unsafe_allow_html=True)
    
    X_test = st.session_state.get("X_test")
    ebm = st.session_state.get("ebm_model")
    metrics = st.session_state.get("model_metrics", {}).get("ebm", {})
    predictions = metrics.get("y_pred", np.array([]))
    probabilities = metrics.get("y_proba", np.array([]))
    
    if X_test is None or ebm is None or len(predictions) == 0:
        st.info("Train a model to review individual applicants.")
        return
    
    # Row selector
    row_index = st.number_input(
        "Select applicant row", 
        min_value=0, 
        max_value=len(X_test) - 1, 
        value=0,
        step=1,
        help=f"Select from {len(X_test):,} applicants in holdout set"
    )
    
    predicted = predictions[row_index]
    predicted_label = LABEL_MAP[predicted]
    confidence = probabilities[row_index].max()
    
    # Applicant header
    tier_color = TIER_COLORS[predicted_label]
    review_status = "Needs analyst review" if predicted_label in ["P3", "P4"] or confidence < 0.70 else "Standard review"
    
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        st.markdown(
            f"""
            <div class="tier-badge" style="background: {tier_color}20; color: {tier_color}; border: 1px solid {tier_color};">
                {predicted_label}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Model confidence", f"{confidence:.1%}")
    with col3:
        st.markdown(f"**{review_status}**")
    
    st.caption("Probability is a classifier score, not a probability of repayment.")
    
    st.divider()
    
    # Three evidence cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**EBM model-native evidence**")
        st.caption("Global feature importance from the EBM model.")
        
        fig = create_ebm_global_importance(ebm)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
    
    with col2:
        st.markdown("**SHAP attribution**")
        st.caption("Compute on demand (bounded 50 rows).")
        
        if st.button("Compute SHAP", key="shap_btn"):
            with st.spinner("Computing bounded SHAP..."):
                try:
                    import shap
                    sample = X_test.sample(min(50, len(X_test)), random_state=42)
                    background = sample.sample(min(20, len(sample)), random_state=42)
                    explainer = shap.Explainer(ebm.predict_proba, background, algorithm="permutation")
                    shap_values = explainer(sample, max_evals=2 * X_test.shape[1] + 1)
                    
                    fig = create_shap_local_bars(shap_values.values, X_test.columns.tolist())
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"SHAP computation failed: {e}")
    
    with col3:
        st.markdown("**LIME local approximation**")
        st.caption("Local approximation for selected applicant.")
        
        if st.button("Compute LIME", key="lime_btn"):
            with st.spinner("Computing LIME explanation..."):
                try:
                    from lime.lime_tabular import LimeTabularExplainer
                    training = X_test.sample(min(1000, len(X_test)), random_state=42)
                    explainer = LimeTabularExplainer(
                        training.values,
                        feature_names=training.columns.tolist(),
                        class_names=["P1", "P2", "P3", "P4"],
                        mode="classification",
                        discretize_continuous=True,
                        random_state=42,
                    )
                    explanation = explainer.explain_instance(
                        X_test.iloc[row_index].values,
                        ebm.predict_proba,
                        num_features=5,
                        top_labels=1,
                    )
                    
                    class_position = list(ebm.classes_).index(predicted)
                    pairs = explanation.as_list(label=class_position)
                    
                    lime_df = pd.DataFrame({
                        "local condition": [name for name, _ in pairs],
                        "weight": [f"{weight:+.4f}" for _, weight in pairs]
                    })
                    st.dataframe(lime_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"LIME computation failed: {e}")


def render_governance():
    """Render fairness monitoring section."""
    st.markdown('<div class="section-header">Governance</div>', unsafe_allow_html=True)
    
    metrics = st.session_state.get("model_metrics", {}).get("ebm", {})
    predictions = metrics.get("y_pred", np.array([]))
    external_df = st.session_state.get("external_df")
    X_test = st.session_state.get("X_test")
    y_test = st.session_state.get("y_test")
    
    if len(predictions) == 0 or external_df is None or X_test is None or y_test is None:
        st.info("Train a model to see fairness diagnostics.")
        return
    
    # Attribute selector
    available_attrs = [col for col in SENSITIVE_COLS if col in external_df.columns]
    if not available_attrs:
        st.warning("No sensitive attributes available for monitoring.")
        return
    
    attribute = st.selectbox("Monitoring attribute", available_attrs)
    
    # Get groups from external_df
    try:
        groups = external_df.loc[X_test.index, attribute].reset_index(drop=True)
    except KeyError:
        groups = external_df[attribute].iloc[:len(X_test)].reset_index(drop=True)
    
    predicted_series = pd.Series(predictions)
    
    # Calculate selection rates (P1/P2 = approved)
    approved = predicted_series.isin([0, 1]).astype(int)
    summary = pd.DataFrame({"group": groups, "approved": approved})
    rates = summary.groupby("group")["approved"].agg(["mean", "size"]).rename(
        columns={"mean": "selection_rate", "size": "sample_size"}
    )
    
    # Calculate gaps
    dpd = float(rates["selection_rate"].max() - rates["selection_rate"].min())
    
    # Status indicator
    def get_gap_status(value):
        if value < 0.05:
            return "Low gap", "status-low-gap"
        elif value < 0.10:
            return "Review", "status-review"
        else:
            return "Investigate", "status-investigate"
    
    status_text, status_class = get_gap_status(dpd)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Selection-rate gap</div>
                <div class="value">{dpd:.3f}</div>
                <div class="status-indicator {status_class}">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.metric("Groups monitored", len(rates))
    
    # Selection rate chart
    groups_list = rates.index.tolist()
    rates_list = (rates["selection_rate"] * 100).tolist()
    sample_sizes = rates["sample_size"].astype(int).tolist()
    
    fig = create_fairness_bars(groups_list, rates_list, sample_sizes, attribute)
    st.pyplot(fig)
    plt.close(fig)
    
    # Rates table
    display_rates = rates.copy()
    display_rates["selection_rate"] = (display_rates["selection_rate"] * 100).round(1).astype(str) + "%"
    display_rates = display_rates.rename(columns={
        "selection_rate": "Selection rate",
        "sample_size": "Sample size"
    })
    st.dataframe(display_rates, use_container_width=True)
    
    # Governance warning
    st.markdown(
        """
        <div class="governance-warning">
            <p><strong>What to check next:</strong> Verify data quality, group sample sizes, 
            proxy features, calibration, and policy impacts. These diagnostics do not prove 
            a model is fair or make it suitable for automated decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Main dashboard entry point."""
    # Load custom CSS
    load_custom_css()
    
    # Render header
    render_header()
    
    # Main layout: left rail + main content
    col_left, col_main = st.columns([1, 4])
    
    with col_left:
        render_run_control_rail()
    
    with col_main:
        # Portfolio health
        render_portfolio_health()
        
        st.divider()
        
        # Portfolio evidence
        render_portfolio_evidence()
        
        st.divider()
        
        # Selected applicant
        render_applicant_panel()
        
        st.divider()
        
        # Governance
        render_governance()
        
        # Footer context
        st.caption("""
        **Session context:** This hosted demo uses bounded training and lazy explanations. 
        Metrics shown are from the current session, not the full published benchmark. 
        The 95.6% / AUC 0.982 benchmark comes from the full 51,336-row run with a 10,268-applicant holdout.
        """)


if __name__ == "__main__":
    main()
