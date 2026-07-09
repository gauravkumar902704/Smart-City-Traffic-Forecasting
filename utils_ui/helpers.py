"""
helpers.py
==========

Utility helper functions for the Streamlit Dashboard.

Author:
    Gaurav Kumar
"""

from pathlib import Path

import pandas as pd
import streamlit as st


# ==========================================================
# Load CSV
# ==========================================================
@st.cache_data

def load_uploaded_csv(uploaded_file):
    """
    Load uploaded CSV file.
    """

    if uploaded_file is None:
        return None

    try:

        df = pd.read_csv(uploaded_file)

        return df

    except Exception as e:

        st.error(f"Unable to read CSV file.\n{e}")

        return None


# ==========================================================
# Dataset Summary
# ==========================================================

def dataset_summary(df):

    if df is None:

        return {
            "Rows": 0,
            "Columns": 0,
            "Missing Values": 0,
            "Duplicates": 0,
            "Memory": "0 MB"
        }

    memory = df.memory_usage(deep=True).sum() / 1024**2

    return {

        "Rows": len(df),

        "Columns": len(df.columns),

        "Missing Values": int(df.isna().sum().sum()),

        "Duplicates": int(df.duplicated().sum()),

        "Memory": f"{memory:.2f} MB"

    }


# ==========================================================
# Display Dataset
# ==========================================================

def preview_dataset(df):

    if df is None:

        return

    st.subheader("📊 Dataset Preview")

    st.dataframe(

        df,

        use_container_width="stretch",

        height=400

    )


# ==========================================================
# Dataset Statistics
# ==========================================================

def show_dataset_statistics(df):

    summary = dataset_summary(df)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(

            "Rows",

            summary["Rows"]

        )

        st.metric(

            "Missing",

            summary["Missing Values"]

        )

    with c2:

        st.metric(

            "Columns",

            summary["Columns"]

        )

        st.metric(

            "Duplicates",

            summary["Duplicates"]

        )

    with c3:

        st.metric(

            "Memory",

            summary["Memory"]

        )


# ==========================================================
# Data Types
# ==========================================================

def show_dtypes(df):

    if df is None:

        return

    st.subheader("📋 Data Types")

    datatype = pd.DataFrame({

        "Column": df.columns,

        "Data Type": df.dtypes.astype(str)

    })

    st.dataframe(

        datatype,

        use_container_width="stretch",

    )


# ==========================================================
# Download CSV
# ==========================================================

def download_dataframe(df):

    if df is None:

        return

    csv = df.to_csv(index=False)

    st.download_button(

        label="📥 Download CSV",

        data=csv,

        file_name="prediction.csv",

        mime="text/csv"

    )


# ==========================================================
# Model Status
# ==========================================================

def model_exists():

    from src.config import MODEL_FILE

    return MODEL_FILE.exists()


# ==========================================================
# Output Files
# ==========================================================

def output_files():

    outputs = Path("outputs")

    if not outputs.exists():

        return []

    return list(outputs.glob("*"))


# ==========================================================
# Success Message
# ==========================================================

def success(msg):

    st.success(msg)


# ==========================================================
# Error Message
# ==========================================================

def error(msg):

    st.error(msg)


# ==========================================================
# Info Message
# ==========================================================

def info(msg):

    st.info(msg)


# ==========================================================
# Warning Message
# ==========================================================

def warning(msg):

    st.warning(msg)