import streamlit as st
import pandas as pd


def load_excel_or_csv(upload):
    try:
        if upload.name.endswith(".xlsx"):
            return pd.read_excel(upload)
        return pd.read_csv(upload)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def main():
    st.title("Upload Datasets")
    st.write("Upload the internal bank dataset and external CIBIL dataset to begin.")

    internal_file = st.file_uploader(
        "Internal Bank Dataset (.xlsx or .csv)", type=["xlsx", "csv"]
    )
    external_file = st.file_uploader(
        "External CIBIL Dataset (.xlsx or .csv)", type=["xlsx", "csv"]
    )
    unseen_file = st.file_uploader(
        "Unseen Dataset for scoring (.xlsx or .csv) — optional",
        type=["xlsx", "csv"],
    )

    if st.button("Load Data"):
        if not internal_file or not external_file:
            st.error("Upload both the Internal and External datasets before continuing.")
            return

        internal_df = load_excel_or_csv(internal_file)
        external_df = load_excel_or_csv(external_file)

        if internal_df is not None and external_df is not None:
            st.success("Datasets loaded.")

            st.subheader("Internal Dataset Preview")
            st.dataframe(internal_df.head())

            st.subheader("External Dataset Preview")
            st.dataframe(external_df.head())

            st.session_state["internal_df"] = internal_df
            st.session_state["external_df"] = external_df

        if unseen_file:
            unseen_df = load_excel_or_csv(unseen_file)
            if unseen_df is not None:
                st.session_state["unseen_df"] = unseen_df
                st.success("Unseen dataset loaded.")

        st.info("Proceed to Preprocessing.")


main()
