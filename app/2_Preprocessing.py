import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

TARGET_COL = "Approved_Flag"
TARGET_MAP = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
DROP_COLS  = ["PROSPECTID"]
CAT_COLS   = ["MARITALSTATUS", "EDUCATION", "GENDER", "last_prod_enq2", "first_prod_enq2"]


def find_common_key(a: pd.DataFrame, b: pd.DataFrame):
    candidates = ["id", "customer_id", "cust_id", "client_id", "account_id", "clientid", "prospectid"]
    a_lower = {col.lower(): col for col in a.columns}
    b_lower = {col.lower(): col for col in b.columns}
    for lc, orig in a_lower.items():
        if lc in b_lower:
            return orig, b_lower[lc]
    for cand in candidates:
        if cand in a_lower and cand in b_lower:
            return a_lower[cand], b_lower[cand]
    return None, None


def merge_datasets(df_int: pd.DataFrame, df_ext: pd.DataFrame) -> pd.DataFrame:
    key_a, key_b = find_common_key(df_int, df_ext)
    if key_a and key_b:
        st.success(f"Merge key detected — Internal: `{key_a}` / External: `{key_b}`")
        df = pd.merge(df_int, df_ext, left_on=key_a, right_on=key_b, how="inner", suffixes=("_int", "_ext"))
        if key_a != key_b and key_b in df.columns:
            df.drop(columns=[key_b], inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.info(f"Merged shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df
    if df_int.shape[0] == df_ext.shape[0]:
        st.warning("No common ID column found — merging by index (row counts match).")
        df = pd.concat([df_int.reset_index(drop=True), df_ext.reset_index(drop=True)], axis=1)
        df.reset_index(drop=True, inplace=True)
        return df
    st.error("No common ID column and row counts differ. Cannot safely merge.")
    st.stop()


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
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


def run_preprocessing(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        st.error(f"Target column `{TARGET_COL}` not found. Ensure the External CIBIL dataset contains this column.")
        st.stop()

    cols_to_drop = [c for c in DROP_COLS if c in df.columns] + [TARGET_COL]
    X_raw = df.drop(columns=cols_to_drop)
    y_raw = df[TARGET_COL].copy()

    cat_cols_present = [c for c in CAT_COLS if c in X_raw.columns]
    num_cols_present = [c for c in X_raw.select_dtypes(include=["int64", "float64"]).columns
                        if c not in cat_cols_present]

    st.write(f"**Numeric features**: {len(num_cols_present)}")
    st.write(f"**Categorical features**: {len(cat_cols_present)} — {cat_cols_present}")

    preprocessor = build_preprocessor(num_cols_present, cat_cols_present)
    X_transformed = preprocessor.fit_transform(X_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()
    X_processed = pd.DataFrame(X_transformed, columns=feature_names, dtype=np.float32)

    unknown_labels = set(y_raw.unique()) - set(TARGET_MAP.keys())
    if unknown_labels:
        st.warning(f"Unexpected target values found: {unknown_labels}. They will be dropped.")
    y = y_raw.map(TARGET_MAP).dropna().astype(int)

    X_processed = X_processed.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X_processed, y, feature_names, preprocessor


def main():
    st.title("Preprocessing & Feature Engineering")

    if "internal_df" not in st.session_state or "external_df" not in st.session_state:
        st.error("Upload both datasets first on the Upload Data page.")
        return

    df_int = st.session_state["internal_df"]
    df_ext = st.session_state["external_df"]

    st.subheader("Step 1 — Merge Datasets")
    with st.spinner("Merging datasets..."):
        df_merged = merge_datasets(df_int, df_ext)

    with st.expander("Preview merged data (first 5 rows)"):
        st.dataframe(df_merged.head())

    st.subheader("Step 2 — Encode, Impute & Scale")

    if st.button("Run Preprocessing"):
        with st.spinner("Running preprocessing pipeline..."):
            try:
                X_processed, y, feature_names, preprocessor = run_preprocessing(df_merged)
            except SystemExit:
                return
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                return

        processed_df = X_processed.copy()
        processed_df[TARGET_COL] = y.values

        st.session_state["processed_df"]  = processed_df
        st.session_state["feature_names"] = feature_names
        st.session_state["preprocessor"]  = preprocessor
        st.session_state["label_map"]     = {v: k for k, v in TARGET_MAP.items()}

        st.success("Preprocessing complete.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total features", len(feature_names))
        col2.metric("Training samples", len(processed_df))
        col3.metric("Target classes", y.nunique())

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

        with st.expander(f"Feature names after encoding ({len(feature_names)} total)"):
            cols = st.columns(3)
            chunk = len(feature_names) // 3 + 1
            for i, col in enumerate(cols):
                col.write("\n".join(feature_names[i * chunk: (i + 1) * chunk]))

        with st.expander("Preview processed data (first 5 rows)"):
            st.dataframe(processed_df.head())

        st.info("Proceed to Train Models.")


main()
