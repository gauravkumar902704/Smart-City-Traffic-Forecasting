"""
feature_engineering.py
======================

Feature engineering module for Smart City Traffic Forecasting.

This module creates calendar-based and cyclic time features
from the DateTime column.

Author:
    Gaurav Kumar
"""

import logging
import numpy as np
import pandas as pd

from src.config import DATETIME_COLUMN

logger = logging.getLogger(__name__)


# ==============================================================================
# Helper Function
# ==============================================================================

def get_season(month: int) -> str:
    """
    Map month to season.
    """

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Summer"

    if month in [6, 7, 8, 9]:
        return "Monsoon"

    return "Autumn"


# ==============================================================================
# Feature Engineering
# ==============================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from DateTime.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()

    dt = df[DATETIME_COLUMN]

    # --------------------------------------------------
    # Basic Date Features
    # --------------------------------------------------

    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Hour"] = dt.dt.hour
    df["Minute"] = dt.dt.minute

    # --------------------------------------------------
    # Week Features
    # --------------------------------------------------

    df["DayOfWeek"] = dt.dt.dayofweek
    df["DayName"] = dt.dt.day_name()

    df["WeekOfYear"] = (
        dt.dt.isocalendar().week.astype(int)
    )

    df["DayOfYear"] = dt.dt.dayofyear

    df["Quarter"] = dt.dt.quarter

    # --------------------------------------------------
    # Weekend
    # --------------------------------------------------

    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # Month Position
    # --------------------------------------------------

    df["IsMonthStart"] = (
        dt.dt.is_month_start
    ).astype(int)

    df["IsMonthEnd"] = (
        dt.dt.is_month_end
    ).astype(int)

    df["IsQuarterStart"] = (
        dt.dt.is_quarter_start
    ).astype(int)

    df["IsQuarterEnd"] = (
        dt.dt.is_quarter_end
    ).astype(int)

    # --------------------------------------------------
    # Peak Hour Features
    # --------------------------------------------------

    df["MorningRush"] = (
        df["Hour"].between(7, 10)
    ).astype(int)

    df["EveningRush"] = (
        df["Hour"].between(16, 20)
    ).astype(int)

    df["BusinessHours"] = (
        df["Hour"].between(9, 18)
    ).astype(int)

    df["NightHours"] = (
        (
            df["Hour"] <= 5
        ) |
        (
            df["Hour"] >= 22
        )
    ).astype(int)

    # --------------------------------------------------
    # Cyclic Encoding
    # --------------------------------------------------

    df["Hour_sin"] = np.sin(
        2 * np.pi * df["Hour"] / 24
    )

    df["Hour_cos"] = np.cos(
        2 * np.pi * df["Hour"] / 24
    )

    df["Month_sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    df["DayOfWeek_sin"] = np.sin(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    df["DayOfWeek_cos"] = np.cos(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    # --------------------------------------------------
    # Season
    # --------------------------------------------------

    df["Season"] = df["Month"].apply(get_season)

    season_map = {
        "Winter": 0,
        "Summer": 1,
        "Monsoon": 2,
        "Autumn": 3,
    }

    df["Season"] = (
        df["Season"]
        .map(season_map)
        .astype(int)
    )

    logger.info("Time features created successfully.")

    return df