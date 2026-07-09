"""
preprocessing.py
================

Data preprocessing module for Smart City Traffic Forecasting.

This module provides utilities for:

1. Loading datasets
2. Validating datasets
3. Cleaning data
4. Datetime conversion
5. Preparing model-ready features

Author:
    Gaurav Kumar
"""

from pathlib import Path
import logging
from typing import Tuple

import pandas as pd

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    TARGET_COLUMN,
    DATETIME_COLUMN,
    ID_COLUMN,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Load Dataset
# ==============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test datasets.

    Returns
    -------
    tuple
        (train_dataframe, test_dataframe)

    Raises
    ------
    FileNotFoundError
        If dataset files are missing.
    """

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    logger.info("Datasets loaded successfully.")

    return train_df, test_df


# ==============================================================================
# Validate Dataset
# ==============================================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate required columns.
    """

    required_columns = [DATETIME_COLUMN]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# ==============================================================================
# Clean Dataset
# ==============================================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe.

    Operations
    ----------
    - Remove duplicate rows
    - Drop rows with missing DateTime
    """

    df = df.copy()

    df.drop_duplicates(inplace=True)

    df.dropna(
        subset=[DATETIME_COLUMN],
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    logger.info("Dataset cleaned successfully.")

    return df


# ==============================================================================
# Convert DateTime
# ==============================================================================

def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DateTime column into pandas datetime format.
    """

    df = df.copy()

    df[DATETIME_COLUMN] = pd.to_datetime(
        df[DATETIME_COLUMN],
        errors="coerce",
    )

    df.dropna(
        subset=[DATETIME_COLUMN],
        inplace=True,
    )

    logger.info("Datetime conversion completed.")

    return df


# ==============================================================================
# Prepare Model Data
# ==============================================================================

def prepare_model_data(
    df: pd.DataFrame,
    is_train: bool = True,
):
    """
    Prepare model-ready dataset.

    Parameters
    ----------
    df : DataFrame

    is_train : bool

    Returns
    -------
    Training:
        X, y

    Prediction:
        X
    """

    df = df.copy()

    columns_to_drop = [
        DATETIME_COLUMN,
        "DayName",
        ID_COLUMN,
    ]

    df.drop(
        columns=columns_to_drop,
        inplace=True,
        errors="ignore",
    )

    if is_train:

        X = df.drop(columns=[TARGET_COLUMN])

        y = df[TARGET_COLUMN]

        logger.info("Training data prepared.")

        return X, y

    logger.info("Prediction data prepared.")

    return df