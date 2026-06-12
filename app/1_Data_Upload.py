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
    st.caption("Upload the internal bank trade-line data and external CIBIL bureau data.")

    # Schema reference
    with st.expander("Expected dataset schemas"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Internal Bank Dataset** (26 columns)")
            st.markdown(
                "Key columns: `PROSPECTID`, `Total_TL`, `Tot_Missed_Pmnt`, "
                "`CC_TL`, `Home_TL`, `PL_TL`, `Age_Oldest_TL`, `Age_Newest_TL`, "
                "and percentage-based trade-line metrics.\n\n"
                "Merge key: **PROSPECTID**"
            )
        with col2:
            st.markdown("**External CIBIL Dataset** (62 columns)")
            st.markdown(
                "Key columns: `PROSPECTID`, `num_times_delinquent`, `CC_utilization`, "
                "`GENDER`, `EDUCATION`, `MARITALSTATUS`, `AGE`, `NETMONTHLYINCOME`, "
                "`Credit_Score`, **`Approved_Flag`** (target: P1/P2/P3/P4).\n\n"
                "Merge key: **PROSPECTID**"
            )
        st.caption(
            "Using the bundled sample datasets? Return to Home and click Launch Demo "
            "to skip this page entirely."
        )

    internal_file = st.file_uploader(
        "Internal Bank Dataset (.xlsx or .csv)", type=["xlsx", "csv"]
    )
    external_file = st.file_uploader(
        "External CIBIL Dataset (.xlsx or .csv)", type=["xlsx", "csv"]
    )
    unseen_file = st.file_uploader(
        "Unseen Dataset for scoring (.xlsx or .csv) (optional)",
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
