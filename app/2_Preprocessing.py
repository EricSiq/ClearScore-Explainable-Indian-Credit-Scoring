import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ── Schema constants ────────────────────────────────────────────────────────
TARGET_COL  = "Approved_Flag"
TARGET_MAP  = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
DROP_COLS   = ["PROSPECTID"]

# Categorical columns present after the internal + external merge.
# These are the only string-typed columns in the merged dataset.
CAT_COLS = [
    "MARITALSTATUS",
    "EDUCATION",
    "GENDER",
    "last_prod_enq2",
    "first_prod_enq2",
]


# ── Merge helpers ────────────────────────────────────────────────────────────

def find_common_key(a: pd.DataFrame, b: pd.DataFrame):
    """
    Detect a shared ID column between two DataFrames.
    First tries a case-insensitive exact match across all columns,
    then falls back to a list of common ID column name candidates.
    Returns (col_in_a, col_in_b) or (None, None).
    """
    candidates = [
        "id", "customer_id", "cust_id", "client_id",
        "account_id", "clientid", "prospectid",
    ]

    a_lower = {col.lower(): col for col in a.columns}
    b_lower = {col.lower(): col for col in b.columns}

    # Exact case-insensitive match
    for lc, orig in a_lower.items():
        if lc in b_lower:
            return orig, b_lower[lc]

    # Candidate list fallback
    for cand in candidates:
        if cand in a_lower and cand in b_lower:
            return a_lower[cand], b_lower[cand]

    return None, None


def merge_datasets(df_int: pd.DataFrame, df_ext: pd.DataFrame) -> pd.DataFrame:
    """
    Merge internal and external datasets.
    Prefers key-based inner join; falls back to index concat when row counts match.
    """
    key_a, key_b = find_common_key(df_int, df_ext)

    if key_a and key_b:
        st.success(f"Merge key detected — Internal: `{key_a}` ↔ External: `{key_b}`")
        df = pd.merge(
            df_int, df_ext,
            left_on=key_a, right_on=key_b,
            how="inner",
            suffixes=("_int", "_ext"),
        )
        # Drop the duplicate key column from the right side if names differ
        if key_a != key_b and key_b in df.columns:
            df.drop(columns=[key_b], inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.info(f"Merged shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    if df_int.shape[0] == df_ext.shape[0]:
        st.warning("No common ID column found — merging by index (row counts match).")
        df = pd.concat(
            [df_int.reset_index(drop=True), df_ext.reset_index(drop=True)], axis=1
        )
        df.reset_index(drop=True, inplace=True)
        return df

    st.error(
        "No common ID column and row counts differ. "
        "Cannot safely merge — please check your datasets."
    )
    st.stop()


# ── Preprocessing pipeline ───────────────────────────────────────────────────

def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Numeric cols: median imputation → standard scaling
      - Categorical cols: most-frequent imputation → one-hot encoding
    verbose_feature_names_out=False gives clean names (e.g. 'GENDER_M')
    instead of prefixed names (e.g. 'cat__GENDER_M').
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",                  # drops PROSPECTID and any other leftovers
        verbose_feature_names_out=False,   # clean column names after get_feature_names_out()
    )
    return preprocessor


def run_preprocessing(df: pd.DataFrame):
    """
    Full preprocessing flow on the merged DataFrame:
      1. Separate target and drop identifier columns
      2. Detect numeric / categorical feature columns
      3. Fit ColumnTransformer and reconstruct a named DataFrame
      4. Label-encode the target (P1→0, P2→1, P3→2, P4→3)
      5. Return (X_processed, y, feature_names, preprocessor)
    """
    # ── 1. Validate target ───────────────────────────────────────────────────
    if TARGET_COL not in df.columns:
        st.error(
            f"Target column `{TARGET_COL}` not found in the merged dataset. "
            "Ensure the External CIBIL dataset contains this column."
        )
        st.stop()

    # ── 2. Separate features and target ─────────────────────────────────────
    cols_to_drop = [c for c in DROP_COLS if c in df.columns] + [TARGET_COL]
    X_raw = df.drop(columns=cols_to_drop)
    y_raw = df[TARGET_COL].copy()

    # ── 3. Detect column types ───────────────────────────────────────────────
    # Only keep CAT_COLS that actually exist in the merged frame
    cat_cols_present = [c for c in CAT_COLS if c in X_raw.columns]

    # Everything else that is numeric (int64 / float64)
    num_cols_present = (
        X_raw
        .select_dtypes(include=["int64", "float64"])
        .columns
        .tolist()
    )
    # Remove any numeric col that is also in cat_cols (shouldn't happen, but safe)
    num_cols_present = [c for c in num_cols_present if c not in cat_cols_present]

    st.write(f"**Numeric features**: {len(num_cols_present)}")
    st.write(f"**Categorical features**: {len(cat_cols_present)} — {cat_cols_present}")

    # ── 4. Fit ColumnTransformer ─────────────────────────────────────────────
    preprocessor = build_preprocessor(num_cols_present, cat_cols_present)
    X_transformed = preprocessor.fit_transform(X_raw)

    # Reconstruct a named DataFrame so EBM / SHAP / LIME all see real column names
    feature_names = preprocessor.get_feature_names_out().tolist()
    X_processed = pd.DataFrame(X_transformed, columns=feature_names, dtype=np.float32)

    # ── 5. Label-encode target ───────────────────────────────────────────────
    unknown_labels = set(y_raw.unique()) - set(TARGET_MAP.keys())
    if unknown_labels:
        st.warning(
            f"Unexpected target values found: {unknown_labels}. "
            "They will be mapped to NaN and dropped."
        )
    y = y_raw.map(TARGET_MAP).dropna().astype(int)

    # Align X to the valid y indices (in case any rows were dropped)
    X_processed = X_processed.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X_processed, y, feature_names, preprocessor


# ── Streamlit page ───────────────────────────────────────────────────────────

def main():
    st.title("🔧 Preprocessing & Feature Engineering")

    if "internal_df" not in st.session_state or "external_df" not in st.session_state:
        st.error("⚠ Upload both datasets first on the **Upload Data** page.")
        return

    df_int = st.session_state["internal_df"]
    df_ext = st.session_state["external_df"]

    # ── Step 1: Merge ────────────────────────────────────────────────────────
    st.subheader("Step 1 — Merge Datasets")
    with st.spinner("Merging internal and external datasets..."):
        df_merged = merge_datasets(df_int, df_ext)

    with st.expander("Preview merged data (first 5 rows)"):
        st.dataframe(df_merged.head())

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    st.subheader("Step 2 — Encode, Impute & Scale")

    if st.button("▶ Run Preprocessing"):
        with st.spinner("Running preprocessing pipeline..."):
            try:
                X_processed, y, feature_names, preprocessor = run_preprocessing(df_merged)
            except SystemExit:
                # st.stop() raises SystemExit — let Streamlit handle it cleanly
                return
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                return

        # ── Store everything downstream pages need ───────────────────────────
        # Reconstruct a single processed_df with target appended for Page 3
        processed_df = X_processed.copy()
        processed_df[TARGET_COL] = y.values

        st.session_state["processed_df"]  = processed_df
        st.session_state["feature_names"] = feature_names
        st.session_state["preprocessor"]  = preprocessor
        st.session_state["label_map"]     = {v: k for k, v in TARGET_MAP.items()}

        # ── Summary ──────────────────────────────────────────────────────────
        st.success("✅ Preprocessing complete!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total features", len(feature_names))
        col2.metric("Training samples", len(processed_df))
        col3.metric("Target classes", y.nunique())

        # Class distribution
        st.subheader("Target Distribution")
        label_map = st.session_state["label_map"]
        dist = (
            y.map(label_map)
             .value_counts()
             .reindex(["P1", "P2", "P3", "P4"])
             .rename("Count")
             .to_frame()
        )
        dist["Pct"] = (dist["Count"] / dist["Count"].sum() * 100).round(1).astype(str) + "%"
        dist.index.name = "Class"
        st.table(dist)

        # Feature name preview
        with st.expander(f"Feature names after encoding ({len(feature_names)} total)"):
            # Show in 3 columns for readability
            cols = st.columns(3)
            chunk = len(feature_names) // 3 + 1
            for i, col in enumerate(cols):
                col.write("\n".join(feature_names[i * chunk : (i + 1) * chunk]))

        # Processed data preview
        with st.expander("Preview processed data (first 5 rows)"):
            st.dataframe(processed_df.head())

        st.info("➡ Proceed to **Train Models** to fit Logistic Regression and EBM.")


if __name__ == "__main__":
    main()
