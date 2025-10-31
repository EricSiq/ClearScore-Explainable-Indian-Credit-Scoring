import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def find_common_key(a: pd.DataFrame, b: pd.DataFrame):
    candidates = ['ID','Id','id','customer_id','cust_id','Client_ID',
                  'CLIENT_ID','client_id','Account_ID','account_id','clientid']

    a_cols_lower = {col.lower(): col for col in a.columns}
    b_cols_lower = {col.lower(): col for col in b.columns}

    # Direct exact match (case-insensitive)
    for lower_col, orig_col in a_cols_lower.items():
        if lower_col in b_cols_lower:
            return orig_col, b_cols_lower[lower_col]

    # Try candidate names
    for cand in candidates:
        if cand.lower() in a_cols_lower and cand.lower() in b_cols_lower:
            return a_cols_lower[cand.lower()], b_cols_lower[cand.lower()]

    return None, None


def main():
    st.title("Preprocessing & Feature Engineering")

    if "internal_df" not in st.session_state or "external_df" not in st.session_state:
        st.error("⚠ Upload datasets first from the 'Upload Data' page.")
        return

    df_int = st.session_state["internal_df"]
    df_ext = st.session_state["external_df"]

    st.write("🔍 Detecting best merge key...")

    try:
        key_a, key_b = find_common_key(df_int, df_ext)

        if key_a and key_b:
            st.success(f"Merging on detected key: Internal '{key_a}' <--> External '{key_b}'")

            df = pd.merge(
                df_int, df_ext,
                left_on=key_a,
                right_on=key_b,
                how='inner',
                suffixes=('_int', '_ext')
            )

            # Drop duplicate right key if names differ
            if key_a != key_b and key_b in df.columns:
                df.drop(columns=[key_b], inplace=True)

            df.reset_index(drop=True, inplace=True)

        else:
            # Fallback case - if same number of rows
            if df_int.shape[0] == df_ext.shape[0]:
                st.warning("⚠ No common ID column found. Merging by index (row counts match).")
                df = pd.concat([df_int.reset_index(drop=True),
                                df_ext.reset_index(drop=True)], axis=1)
            else:
                st.error("❌ No merge key and row sizes differ. Doing outer concat - check alignment.")
                df = pd.concat([df_int, df_ext], axis=1)
                df.reset_index(drop=True, inplace=True)

        st.write("Merge complete")
        st.dataframe(df.head())

        # Now store processed dataframe
        st.session_state["processed_df"] = df
        st.success("Data Prprocessed Successfully!")

    except Exception as e:
        st.error(f"Error during preprocessing: {e}")


if __name__ == "__main__":
    main()
