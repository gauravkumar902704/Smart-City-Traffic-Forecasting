"""
sidebar.py
==========

Professional Sidebar Component
for Smart City Traffic Forecasting Dashboard.

Author:
    Gaurav Kumar
"""

from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu
from src.config import VERSION, MODEL_NAME


# ==========================================================
# Sidebar
# ==========================================================

def render_sidebar():

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-title">
                🚦 Smart City
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        selected = option_menu(

            menu_title="Navigation",

            options=[
                "Dashboard",
                "Upload Dataset",
                "Train Model",
                "Prediction",
                "Model Analysis",
                "About Project",
            ],

            icons=[
                "house-fill",
                "cloud-upload-fill",
                "cpu-fill",
                "activity",
                "bar-chart-fill",
                "info-circle-fill",
            ],

            menu_icon="list",

            default_index=0,

            styles={
                "container": {
                    "padding": "5px",
                    "background-color": "#111827",
                },

                "icon": {
                    "color": "#4F8BF9",
                    "font-size": "18px",
                },

                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px",
                    "--hover-color": "#1e293b",
                },

                "nav-link-selected": {
                    "background-color": "#2563EB",
                },
            },
        )

        st.markdown("---")

        st.subheader("🤖 Model Status")

        from src.config import MODEL_FILE

        if MODEL_FILE.exists():

            st.success("Model Loaded")

        else:

            st.error("Model Not Found")

        st.markdown("---")

        st.subheader("📁 Project")

        st.write(f"Version : **{VERSION}**")

        st.write(f"Model : **{MODEL_NAME}**")

        st.write("Framework : **Scikit-Learn**")

        st.markdown("---")



    return selected