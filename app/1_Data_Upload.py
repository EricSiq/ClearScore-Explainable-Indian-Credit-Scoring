import streamlit as st
import pandas as pd

def load_csv(upload):
    try:
        return pd.read_csv(upload)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def main():
    st.title("Upload Datasets")

    st.write("Upload the required datasets:")

    internal_file = st.file_uploader("Upload Internal Bank Dataset (.csv)", type="csv")
    external_file = st.file_uploader("Upload External CIBIL Dataset (.csv)", type="csv")
    unseen_file = st.file_uploader("Upload Unseen Dataset (.csv)", type="csv")

    if st.button("Load Data"):

        if not internal_file or not external_file:
            st.error("Please upload both Internal and External datasets.")
            return

        internal_df = load_csv(internal_file)
        external_df = load_csv(external_file)

        if internal_df is not None and external_df is not None:
            st.success("Files loaded successfully!")

            st.subheader("Internal Dataset Preview")
            st.dataframe(internal_df.head())

            st.subheader("External Dataset Preview")
            st.dataframe(external_df.head())

            # optional: save to session_state for later pages
            st.session_state["internal_df"] = internal_df
            st.session_state["external_df"] = external_df

        if unseen_file:
            unseen_df = load_csv(unseen_file)
            if unseen_df is not None:
                st.session_state["unseen_df"] = unseen_df
                st.success("Unseen dataset loaded")

if __name__ == "__main__":
    main()
