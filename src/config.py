"""
config.py
=========

Central configuration file for the Smart City Traffic Forecasting project.

This module contains:
    • Project directory paths
    • Dataset paths
    • Model paths
    • Output paths
    • Training configuration
    • Random seed
    • Logging configuration

Author:
    Gaurav Kumar
"""

from pathlib import Path

# ==============================================================================
# Project Directories
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# Create required directories automatically
for directory in (
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    LOG_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Dataset Files
# ==============================================================================

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

# ==============================================================================
# Model Files
# ==============================================================================

MODEL_NAME = "RandomForest"

MODEL_FILE = MODEL_DIR / "random_forest.joblib"

FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "feature_importance.png"

# ==============================================================================
# Prediction Output
# ==============================================================================

SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

# ==============================================================================
# Training Configuration
# ==============================================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

# ==============================================================================
# Random Forest Hyperparameters
# ==============================================================================

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ==============================================================================
# Cross Validation
# ==============================================================================

N_SPLITS = 5

# ==============================================================================
# Logging Configuration
# ==============================================================================

LOG_FILE = LOG_DIR / "training.log"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

LOG_LEVEL = "INFO"

# ==============================================================================
# Feature Columns
# ==============================================================================

TARGET_COLUMN = "Vehicles"

ID_COLUMN = "ID"

DATETIME_COLUMN = "DateTime"

# ==============================================================================
# Supported Models
# ==============================================================================

SUPPORTED_MODELS = [
    "LinearRegression",
    "DecisionTree",
    "RandomForest",
    "XGBoost",
]

# ==============================================================================
# Application Information
# ==============================================================================

APP_NAME = "Smart City Traffic Forecasting"

VERSION = "1.0.0"