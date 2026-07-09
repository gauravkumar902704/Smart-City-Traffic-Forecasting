"""
train.py
========

Training pipeline for Smart City Traffic Forecasting.

Workflow
--------
Load Data
    ↓
Validate Data
    ↓
Clean Data
    ↓
Datetime Conversion
    ↓
Feature Engineering
    ↓
Prepare Features
    ↓
Train / Validation Split
    ↓
Train Model
    ↓
Evaluate
    ↓
Save Metrics
    ↓
Feature Importance
    ↓
Residual Plot
    ↓
Save Model

Author:
    Gaurav Kumar
"""

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.config import (
    RF_PARAMS,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_FILE,
    OUTPUT_DIR,
)

from src.preprocessing import (
    load_data,
    validate_dataframe,
    clean_dataframe,
    convert_datetime,
    prepare_model_data,
)

from src.feature_engineering import (
    create_time_features,
)

from src.utils import (
    setup_logging,
    evaluate_model,
    save_metrics,
    save_model,
    plot_feature_importance,
    plot_residuals,
)

import logging

logger = logging.getLogger(__name__)


def train_model():

    setup_logging()

    logger.info("=" * 70)
    logger.info("SMART CITY TRAFFIC FORECASTING")
    logger.info("=" * 70)

    try:

        # =====================================================
        # Load Dataset
        # =====================================================

        logger.info("Loading Dataset...")

        train_df, _ = load_data()

        # =====================================================
        # Validation
        # =====================================================

        validate_dataframe(train_df)

        train_df = clean_dataframe(train_df)

        # =====================================================
        # Datetime Conversion
        # =====================================================

        train_df = convert_datetime(train_df)

        # Sort for Time Series

        train_df = train_df.sort_values(
            "DateTime"
        ).reset_index(drop=True)


        # =====================================================
        # Feature Engineering
        # =====================================================

        train_df = create_time_features(train_df)

        # =====================================================
        # Prepare Data
        # =====================================================

        X, y = prepare_model_data(
            train_df,
            is_train=True,
        )

        # =====================================================
        # Train Validation Split
        # =====================================================

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        # =====================================================
        # Model
        # =====================================================

        logger.info("Training Random Forest...")

        model = RandomForestRegressor(
            **RF_PARAMS
        )

        model.fit(
            X_train,
            y_train,
        )

        logger.info("Training Completed.")

        # =====================================================
        # Prediction
        # =====================================================

        predictions = model.predict(
            X_valid
        )

        # =====================================================
        # Evaluation
        # =====================================================

        metrics = evaluate_model(
            y_valid,
            predictions,
        )

        save_metrics(
            metrics,
            OUTPUT_DIR / "metrics.csv",
        )

        # =====================================================
        # Feature Importance
        # =====================================================

        plot_feature_importance(
            model,
            X.columns,
        )

        # =====================================================
        # Residual Plot
        # =====================================================

        plot_residuals(
            y_valid,
            predictions,
            OUTPUT_DIR / "residual_plot.png",
        )

        # =====================================================
        # Final Training
        # =====================================================

        logger.info("Training Final Model...")

        final_model = RandomForestRegressor(
            **RF_PARAMS
        )

        final_model.fit(
            X,
            y,
        )

        # =====================================================
        # Save Model
        # =====================================================

        save_model(
            final_model,
            MODEL_FILE,
        )

        logger.info("=" * 70)
        logger.info("Training Completed Successfully.")
        logger.info("=" * 70)

    except Exception as e:

        logger.exception(
            f"Training Failed : {e}"
        )

        raise


if __name__ == "__main__":
    train_model()