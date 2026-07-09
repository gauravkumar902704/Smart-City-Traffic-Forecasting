"""
==============================================================
Smart City Traffic Forecasting Dashboard
==============================================================

Author  : Gaurav Kumar
Version : 1.0.0

Run:

streamlit run app.py
"""

# ==========================================================
# Imports
# ==========================================================

from pathlib import Path

import pandas as pd
import streamlit as st

# ==========================================================
# Components
# ==========================================================

from components.cards import render_cards
from components.footer import render_footer
from components.header import render_header
from components.sidebar import render_sidebar

# ==========================================================
# UI Helpers
# ==========================================================

from utils_ui.helpers import (
    load_uploaded_csv,
    preview_dataset,
    show_dataset_statistics,
    show_dtypes,
    model_exists,
)

from utils_ui.charts import (
    dataset_overview,
    feature_importance_chart,
    prediction_distribution,
    correlation_heatmap,
)

# ==========================================================
# Backend
# ==========================================================

from src.config import (
    APP_NAME,
    VERSION,
    OUTPUT_DIR,
    FEATURE_IMPORTANCE_PLOT,
    SUBMISSION_FILE,
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# Load CSS
# ==========================================================

css_file = Path("assets/styles.css")

if css_file.exists():

    st.markdown(
        f"<style>{css_file.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# ==========================================================
# Session State
# ==========================================================

DEFAULT_SESSION = {
    "dataset": None,
    "prediction": None,
    "model_loaded": model_exists(),
}

for key, value in DEFAULT_SESSION.items():

    if key not in st.session_state:

        st.session_state[key] = value

# ==========================================================
# Sidebar Navigation
# ==========================================================

selected_page = render_sidebar()

# ==========================================================
# Common Helper Functions
# ==========================================================

def page_title(title: str):

    render_header()

    st.title(title)

    st.write("")


def show_model_status():

    if model_exists():

        st.success("🟢 Random Forest Model Loaded")

    else:

        st.error("🔴 Model Not Found")


def upload_dataset_widget():

    uploaded = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
    )

    if uploaded is not None:

        df = load_uploaded_csv(uploaded)

        if df is not None:

            st.session_state.dataset = df

            st.success("✅ Dataset Uploaded Successfully")

    return st.session_state.dataset

# ==========================================================
# Dashboard
# ==========================================================

if selected_page == "Dashboard":

    page_title("🚦 Smart City Traffic Forecasting")

    render_cards()

    st.success("✅ Dashboard Loaded Successfully")

    st.divider()

    st.header("📊 Dashboard")

    col1, col2 = st.columns([2, 1])

    with col1:

        dataset = upload_dataset_widget()

        if dataset is not None:

            st.subheader("📊 Dataset Preview")

            preview_dataset(dataset)

            st.write("")

            show_dataset_statistics(dataset)

            st.write("")

            show_dtypes(dataset)

        else:

            st.info("📂 Upload a dataset to view its details.")

    with col2:

        st.subheader("🎯 Model")

        show_model_status()

        st.write("")

        if st.session_state.dataset is not None:

            st.subheader("📋 Dataset Information")

            rows, cols = st.session_state.dataset.shape

            c1, c2 = st.columns(2)

            c1.metric(
                "Rows",
                rows,
            )

            c2.metric(
                "Columns",
                cols,
            )

            st.metric(
                "Memory",
                f"{st.session_state.dataset.memory_usage(deep=True).sum()/1024**2:.2f} MB",
            )

    st.divider()

    if st.session_state.dataset is not None:

        st.header("📈 Dataset Overview")

        dataset_overview(
            st.session_state.dataset,
        )

    else:

        st.info(
            "Upload a dataset to view charts."
        )

# ==========================================================
# Upload Dataset
# ==========================================================

elif selected_page == "Upload Dataset":

    page_title("📂 Upload Dataset")

    left, right = st.columns([2, 1])

    with left:

        dataset = upload_dataset_widget()

    with right:

        st.info(
            """
### 📋 Upload Guidelines

✅ Supported Format

- CSV

✅ Maximum Size

- 200 MB

✅ Required Columns

Training Dataset

- DateTime
- Junction
- Vehicles

Prediction Dataset

- DateTime
- Junction
- ID
"""
        )

    st.divider()

    if dataset is not None:

        st.success("✅ Dataset Loaded Successfully")

        preview_dataset(dataset)

        st.divider()

        st.subheader("📊 Dataset Statistics")

        show_dataset_statistics(dataset)

        st.divider()

        st.subheader("📋 Column Data Types")

        show_dtypes(dataset)

        st.divider()

        st.subheader("📈 Dataset Visualizations")

        dataset_overview(dataset)

    else:

        st.warning(
            "📂 Please upload a CSV dataset to continue."
        )

# ==========================================================
# Train Model
# ==========================================================

elif selected_page == "Train Model":

    page_title("🤖 Train Machine Learning Model")

    left, right = st.columns([2, 1])

    # ======================================================
    # Training Section
    # ======================================================

    with left:

        st.write(
            """
Train the **Random Forest Regressor** using the available
training dataset.
"""
        )

        if st.button(
            "🚀 Start Model Training",
            use_container_width=True,
        ):

            from src.train import train_model

            try:

                with st.spinner("Training model... Please wait..."):

                    train_model()

                st.session_state.model_loaded = True

                st.success(
                    "✅ Model trained successfully."
                )

            except Exception as e:

                st.error(e)

    # ======================================================
    # Status Section
    # ======================================================

    with right:

        st.subheader("📌 Model Status")

        show_model_status()

        st.write("")

        metrics_file = OUTPUT_DIR / "metrics.csv"

        if metrics_file.exists():

            st.subheader("📊 Evaluation Metrics")

            metrics = pd.read_csv(metrics_file)

            st.dataframe(
                metrics,
                width="stretch",
            )

        else:

            st.info(
                "Train the model to generate metrics."
            )

    st.divider()

    # ======================================================
    # Feature Importance
    # ======================================================

    st.subheader("📈 Feature Importance")

    if FEATURE_IMPORTANCE_PLOT.exists():

        st.image(
            str(FEATURE_IMPORTANCE_PLOT),
            width="stretch",
        )

    else:

        st.info(
            "Train the model to generate the feature importance plot."
        )

# ==========================================================
# Prediction
# ==========================================================

elif selected_page == "Prediction":

    page_title("🚦 Traffic Prediction")

    prediction_mode = st.radio(
        "Prediction Mode",
        [
            "Single Prediction",
            "Bulk Prediction",
        ],
        horizontal=True,
    )

    st.divider()

    # ======================================================
    # Single Prediction
    # ======================================================

    if prediction_mode == "Single Prediction":

        c1, c2 = st.columns(2)

        with c1:

            junction = st.selectbox(
                "🚥 Junction",
                [1, 2, 3, 4],
            )

            date = st.date_input(
                "📅 Date",
            )

        with c2:

            hour = st.slider(
                "🕒 Hour",
                0,
                23,
                12,
            )

        st.write("")

        if st.button(
            "🚀 Predict Traffic",
            use_container_width=True,
        ):

            # TODO:
            # Replace this placeholder with the actual
            # single prediction pipeline.

            st.info(
                "Single Prediction module is under development."
            )

    # ======================================================
    # Bulk Prediction
    # ======================================================

    else:

        uploaded = st.file_uploader(
            "Upload Test CSV",
            type=["csv"],
            key="prediction_upload",
        )

        if uploaded is not None:

            df = load_uploaded_csv(uploaded)

            if df is not None:

                preview_dataset(df)

                st.divider()

                if st.button(
                    "🚀 Generate Predictions",
                    use_container_width=True,
                ):

                    try:

                        from src.predict import predict

                        with st.spinner(
                            "Generating predictions..."
                        ):

                            pred_df = predict(df)

                        if pred_df is not None:

                            st.session_state.prediction = pred_df

                            st.success(
                                "✅ Predictions generated successfully."
                            )

                            st.subheader(
                                "📊 Prediction Preview"
                            )

                            preview_dataset(pred_df)

                            st.divider()

                            show_dataset_statistics(
                                pred_df
                            )

                            st.divider()

                            st.download_button(
                                label="📥 Download Prediction CSV",
                                data=pred_df.to_csv(index=False),
                                file_name="submission.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                    except Exception as e:

                        st.exception(e)

        else:

            st.info(
                "Upload a test dataset to generate predictions."
            )

# ==========================================================
# Model Analysis
# ==========================================================

elif selected_page == "Model Analysis":

    page_title("📈 Model Analysis")

    tabs = st.tabs(
        [
            "📊 Feature Importance",
            "📈 Prediction Distribution",
            "🔥 Correlation Heatmap",
            "📉 Residual Plot",
            "📋 Model Metrics",
        ]
    )

    # ======================================================
    # Feature Importance
    # ======================================================

    with tabs[0]:

        st.subheader("📊 Feature Importance")

        feature_importance_chart()

    # ======================================================
    # Prediction Distribution
    # ======================================================

    with tabs[1]:

        st.subheader("📈 Prediction Distribution")

        if st.session_state.prediction is not None:

            prediction_distribution(
                st.session_state.prediction
            )

        elif st.session_state.dataset is not None:

            if "Vehicles" in st.session_state.dataset.columns:

                prediction_distribution(
                    st.session_state.dataset
                )

            else:

                st.info(
                    "Prediction data not available.\n"
                    "Generate predictions first."
                )

        else:

            st.info(
                "No dataset available."
            )

    # ======================================================
    # Correlation Heatmap
    # ======================================================

    with tabs[2]:

        st.subheader("🔥 Correlation Heatmap")

        if st.session_state.dataset is not None:

            correlation_heatmap(
                st.session_state.dataset
            )

        else:

            st.info(
                "Upload a dataset first."
            )

    # ======================================================
    # Residual Plot
    # ======================================================

    with tabs[3]:

        st.subheader("📉 Residual Plot")

        residual_plot = OUTPUT_DIR / "residual_plot.png"

        if residual_plot.exists():

            st.image(
                str(residual_plot),
                width="stretch",
            )

        else:

            st.info(
                "Residual plot not found.\n"
                "Train the model first."
            )

    # ======================================================
    # Metrics
    # ======================================================

    with tabs[4]:

        st.subheader("📋 Model Performance")

        metrics_file = OUTPUT_DIR / "metrics.csv"

        if metrics_file.exists():

            metrics = pd.read_csv(metrics_file)

            st.dataframe(
                metrics,
                width="stretch",
            )

        else:

            st.info(
                "Metrics not available.\n"
                "Train the model first."
            )


# ==========================================================
# About Project
# ==========================================================

elif selected_page == "About Project":

    page_title("📖 About Smart City Traffic Forecasting")

    st.markdown(
        """
## 🚦 Smart City Traffic Forecasting

An AI-powered Machine Learning application for predicting
traffic volume across multiple smart city junctions.

---

### 🎯 Project Objectives

- Predict future traffic volume
- Analyze traffic patterns
- Assist Smart City traffic management
- Visualize traffic insights
- Build an end-to-end ML Pipeline

---

### ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
- Matplotlib
- Joblib

---

### 🤖 Machine Learning Pipeline

1. Data Loading
2. Data Cleaning
3. DateTime Processing
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Traffic Prediction
8. Visualization

---

### 📂 Project Structure

""")