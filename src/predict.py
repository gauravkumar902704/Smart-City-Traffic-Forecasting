"""
predict.py
==========

Prediction pipeline for Smart City Traffic Forecasting.

Workflow
--------
Load Model
    ↓
Load Test Dataset
    ↓
Validate Dataset
    ↓
Clean Dataset
    ↓
Datetime Conversion
    ↓
Feature Engineering
    ↓
Prepare Features
    ↓
Generate Predictions
    ↓
Save Submission

Author:
    Gaurav Kumar
"""

import logging
import pandas as pd

from src.config import (
    MODEL_FILE,
    SUBMISSION_FILE,
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
    load_model,
)

logger = logging.getLogger(__name__)


def predict(test_df=None):

    setup_logging()

    logger.info("=" * 70)
    logger.info("SMART CITY TRAFFIC FORECASTING")
    logger.info("Prediction Pipeline Started")
    logger.info("=" * 70)

    try:

        # =====================================================
        # Load Model
        # =====================================================

        logger.info("Loading trained model...")

        model = load_model(MODEL_FILE)

        # =====================================================
        # Load Dataset
        # =====================================================

        if test_df is None:

            logger.info("Loading default test dataset...")

            _, test_df = load_data()

        else:

            logger.info("Using uploaded dataset...")

        # =====================================================
        # Validate Dataset
        # =====================================================

        validate_dataframe(test_df)

        # =====================================================
        # Clean Dataset
        # =====================================================

        test_df = clean_dataframe(test_df)

        # =====================================================
        # Datetime Conversion
        # =====================================================

        test_df = convert_datetime(test_df)

        # Keep chronological order

        test_df = test_df.sort_values(
            "DateTime"
        ).reset_index(drop=True)

        # =====================================================
        # Feature Engineering
        # =====================================================

        logger.info("Creating features...")

        test_df = create_time_features(test_df)

        # =====================================================
        # Save IDs
        # =====================================================

        submission = pd.DataFrame()

        submission["ID"] = test_df["ID"]

        # =====================================================
        # Prepare Features
        # =====================================================

        X_test = prepare_model_data(
            test_df,
            is_train=False,
        )

        logger.info(f"Prediction samples : {len(X_test)}")

        # =====================================================
        # Predict
        # =====================================================

        predictions = model.predict(X_test)

        submission["Vehicles"] = predictions

        # =====================================================
        # Save CSV
        # =====================================================

        SUBMISSION_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        submission.to_csv(
            SUBMISSION_FILE,
            index=False,
        )


        # =====================================================
        # Prediction Summary
        # =====================================================

        logger.info("=" * 70)
        logger.info("Prediction Summary")

        logger.info(f"Minimum : {predictions.min():.2f}")

        logger.info(f"Maximum : {predictions.max():.2f}")

        logger.info(f"Average : {predictions.mean():.2f}")

        logger.info(f"Std Dev : {predictions.std():.2f}")

        logger.info("=" * 70)

        logger.info(f"Submission saved : {SUBMISSION_FILE}")

        logger.info("Prediction Pipeline Completed Successfully.")

        return submission

    except Exception as e:

        logger.exception(
            f"Prediction Failed : {e}"
        )

        raise


if __name__ == "__main__":
    predict()