"""
utils.py
========

Utility functions for Smart City Traffic Forecasting.

This module provides:

• Logging configuration
• Model evaluation
• Model save/load
• Feature importance visualization
• Evaluation report generation

Author:
    Gaurav Kumar
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import (
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    FEATURE_IMPORTANCE_PLOT,
)

# ==============================================================================
# Logger
# ==============================================================================

logger = logging.getLogger(__name__)


def setup_logging():
    """
    Configure logging for the project.
    """

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )


# ==============================================================================
# Evaluate Model
# ==============================================================================

def evaluate_model(
    y_true,
    y_pred,
):
    """
    Evaluate regression model.

    Returns
    -------
    dict
        Regression metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
    }

    logger.info("=" * 60)

    logger.info("Model Evaluation")

    logger.info("=" * 60)

    for key, value in metrics.items():
        logger.info(f"{key:<10}: {value:.4f}")

    return metrics


# ==============================================================================
# Save Evaluation Report
# ==============================================================================

def save_metrics(
    metrics: dict,
    filepath: Path,
):
    """
    Save evaluation metrics to CSV.
    """

    df = pd.DataFrame(
        metrics.items(),
        columns=["Metric", "Value"],
    )

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        filepath,
        index=False,
    )

    logger.info(f"Metrics saved to {filepath}")


# ==============================================================================
# Save Model
# ==============================================================================

def save_model(
    model,
    filepath: Path,
):
    """
    Save trained model.
    """

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        filepath,
    )

    size = filepath.stat().st_size / (1024 * 1024)

    logger.info(f"Model saved successfully.")

    logger.info(f"Location : {filepath}")

    logger.info(f"Size     : {size:.2f} MB")


# ==============================================================================
# Load Model
# ==============================================================================

def load_model(filepath: Path):
    """
    Load trained model.
    """

    if not filepath.exists():
        raise FileNotFoundError(
            f"Model not found : {filepath}"
        )

    model = joblib.load(filepath)

    logger.info("Model loaded successfully.")

    return model


# ==============================================================================
# Feature Importance
# ==============================================================================

def plot_feature_importance(
    model,
    feature_names,
    top_n=20,
):
    """
    Plot feature importance.
    """

    if not hasattr(model, "feature_importances_"):
        logger.warning(
            "Model does not support feature importance."
        )
        return

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    )

    csv_path = FEATURE_IMPORTANCE_PLOT.parent / "feature_importance.csv"

    importance.to_csv(
        csv_path,
        index=False,
    )

    logger.info(f"Feature importance CSV saved at: {csv_path}")

    


    plt.figure(figsize=(10, 6))

    plt.barh(
        importance["Feature"],
        importance["Importance"],
    )

    plt.gca().invert_yaxis()

    plt.title("Top Feature Importance")

    plt.tight_layout()

    FEATURE_IMPORTANCE_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        FEATURE_IMPORTANCE_PLOT,
        dpi=300,
    )

    plt.close()

    logger.info(
        f"Feature importance plot saved at {FEATURE_IMPORTANCE_PLOT}"
    )


# ==============================================================================
# Residual Plot
# ==============================================================================

def plot_residuals(
    y_true,
    y_pred,
    output_path: Path,
):
    """
    Save residual plot.
    """

    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))

    plt.scatter(
        y_pred,
        residuals,
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.xlabel("Predicted")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    logger.info(
        f"Residual plot saved at {output_path}"
    )



    