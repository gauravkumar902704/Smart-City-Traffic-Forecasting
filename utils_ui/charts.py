"""
charts.py
=========

Interactive Plotly charts for the Smart City Traffic Forecasting Dashboard.

Author:
    Gaurav Kumar
"""

from pathlib import Path
from src.config import OUTPUT_DIR

import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================================
# Feature Importance
# ==========================================================

def feature_importance_chart():

    file = OUTPUT_DIR / "feature_importance.csv"

    if not file.exists():

        st.warning("Feature importance file not found.")

        return

    df = pd.read_csv(file)

    fig = px.bar(

        df,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        title="Feature Importance",

        template="plotly_dark"

    )

    fig.update_layout(

        height=550,

        yaxis=dict(categoryorder="total ascending")

    )

    st.plotly_chart(

        fig,

        use_container_width="stretch"

    )


# ==========================================================
# Prediction Distribution
# ==========================================================

def prediction_distribution(df):

    if df is None:

        return

    if "Vehicles" not in df.columns:

        return

    fig = px.histogram(

        df,

        x="Vehicles",

        nbins=40,

        title="Prediction Distribution",

        template="plotly_dark"

    )

    st.plotly_chart(

        fig,

        use_container_width="stretch"

    )


# ==========================================================
# Correlation Heatmap
# ==========================================================

def correlation_heatmap(df):

    if df is None:

        return

    numeric = df.select_dtypes(include="number")

    if numeric.empty:

        return

    corr = numeric.corr()

    fig = px.imshow(

        corr,

        text_auto=True,

        aspect="auto",

        title="Correlation Heatmap",

        color_continuous_scale="Viridis"

    )

    st.plotly_chart(

        fig,

        use_container_width="stretch"

    )


# ==========================================================
# Traffic Trend
# ==========================================================

def traffic_trend(df):

    if df is None:

        return

    if "DateTime" not in df.columns:

        return

    if "Vehicles" not in df.columns:

        return

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    fig = px.line(

        df,

        x="DateTime",

        y="Vehicles",

        title="Traffic Trend",

        template="plotly_dark"

    )

    st.plotly_chart(

        fig,

        use_container_width="stretch"

    )


# ==========================================================
# Junction Distribution
# ==========================================================

def junction_distribution(df):

    if df is None:

        return

    if "Junction" not in df.columns:

        return

    fig = px.pie(

        df,

        names="Junction",

        title="Traffic by Junction",

        template="plotly_dark"

    )

    st.plotly_chart(

        fig,

        use_container_width="stretch"

    )


# ==========================================================
# Box Plot
# ==========================================================

def box_plot(df):

    if df is None:

        return

    if "Vehicles" not in df.columns:

        return

    fig = px.box(

        df,

        y="Vehicles",

        title="Vehicle Distribution",

        template="plotly_dark"

    )

    st.plotly_chart(

        fig,

        use_container_width="stretch"

    )


# ==========================================================
# Missing Values
# ==========================================================

def missing_values_chart(df):

    if df is None:

        return

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        st.success("No Missing Values")

        return

    fig = px.bar(

        x=missing.index,

        y=missing.values,

        labels={

            "x":"Column",

            "y":"Missing Values"

        },

        title="Missing Values",

        template="plotly_dark"

    )

    st.plotly_chart(

        fig,

        use_container_width="stretch"

    )


# ==========================================================
# Dataset Overview
# ==========================================================

def dataset_overview(df):

    if df is None:

        return

    c1, c2 = st.columns(2)

    with c1:

        prediction_distribution(df)

        box_plot(df)

    with c2:

        correlation_heatmap(df)

        junction_distribution(df)