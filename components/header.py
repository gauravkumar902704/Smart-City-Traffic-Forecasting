"""
header.py
=========

Professional Header Component
for Smart City Traffic Forecasting Dashboard.

Author:
    Gaurav Kumar
"""

from datetime import datetime
from pathlib import Path
from src.config import MODEL_NAME

import streamlit as st


# ==========================================================
# Logo
# ==========================================================

from PIL import UnidentifiedImageError

def show_logo():

    logo = Path("assets/logo.png")

    if not logo.exists():
        return

    try:

        st.image(
            str(logo),
            width=110
        )

    except (UnidentifiedImageError, OSError):

        st.warning("Logo image is invalid. Please replace assets/logo.png")

# ==========================================================
# Main Header
# ==========================================================

def show_header():

    st.markdown(
        """
        <div class="main-title">
            🚦 Smart City Traffic Forecasting
        </div>

        <div class="subtitle">
            Smart City Traffic Volume Forecasting using Machine Learning
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# Project Information
# ==========================================================

def show_project_information():

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            📅 **Date**

            {}
            """.format(
                datetime.now().strftime("%d %B %Y")
            )
        )

    with col2:

        st.info(
            """
            ⏰ **Time**

            {}
            """.format(
                datetime.now().strftime("%I:%M:%S %p")
            )
        )
    with col3:
        st.success(
        f"""
        🤖 **Model**

        {MODEL_NAME}
        """
    )


# ==========================================================
# Status Banner
# ==========================================================

def show_status_banner():

    st.markdown("")

    st.success(
        "✅ Dashboard Loaded Successfully"
    )


# ==========================================================
# Dashboard Header
# ==========================================================

def render_header():

    show_logo()

    show_header()

    show_project_information()

    show_status_banner()

    st.divider()