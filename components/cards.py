"""
cards.py
========

Professional Metric Cards
for Smart City Traffic Forecasting Dashboard.

Author:
    Gaurav Kumar
"""

from pathlib import Path
import pandas as pd
import streamlit as st


# ==========================================================
# Read Metrics
# ==========================================================

def load_metrics():

    from src.config import OUTPUT_DIR

    metrics_file = OUTPUT_DIR / "metrics.csv"

    if not metrics_file.exists():

        return {
            "MAE": "--",
            "RMSE": "--",
            "R2 Score": "--"
        }

    df = pd.read_csv(metrics_file)

    metrics = {}

    for _, row in df.iterrows():

        metrics[row["Metric"]] = round(row["Value"], 4)

    return metrics


# ==========================================================
# Model Status
# ==========================================================

def model_status():

    from src.config import MODEL_FILE

    model = MODEL_FILE

    if model.exists():

        return "🟢 Loaded"

    return "🔴 Not Found"


# ==========================================================
# Dataset Information
# ==========================================================

def dataset_information(df=None):

    if df is None:

        return "--", "--"

    return len(df), len(df.columns)


# ==========================================================
# Card Component
# ==========================================================

def create_card(title, value):

    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


# ==========================================================
# Dashboard Cards
# ==========================================================

def render_cards(df=None):

    metrics = load_metrics()

    rows, columns = dataset_information(df)

    c1, c2, c3 = st.columns(3)

    with c1:

        create_card(
            "📈 MAE",
            metrics.get("MAE", "--")
        )

    with c2:

        create_card(
            "📉 RMSE",
            metrics.get("RMSE", "--")
        )

    with c3:

        create_card(
            "🎯 R² Score",
            metrics.get("R2 Score", "--")
        )

    st.write("")

    c4, c5, c6 = st.columns(3)

    with c4:

        create_card(
            "🤖 Model",
            model_status()
        )

    with c5:

        create_card(
            "📄 Rows",
            rows
        )

    with c6:

        create_card(
            "📊 Columns",
            columns
        )