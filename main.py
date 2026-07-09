"""
main.py
========

Smart City Traffic Forecasting

Main entry point of the application.

Run:

    python main.py

Author:
    Gaurav Kumar
"""

import logging
import sys

from src.config import (
    APP_NAME,
    VERSION,
)

from src.train import train_model
from src.predict import predict
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def show_header():

    print("\n" + "=" * 70)
    print(APP_NAME)
    print(f"Version : {VERSION}")
    print("=" * 70)


def show_menu():

    print("\nChoose an option:\n")

    print("1. Train Model")

    print("2. Generate Predictions")

    print("3. Train + Predict")

    print("4. Project Information")

    print("5. Exit")


def project_info():

    print("\nProject Information")
    print("-" * 40)

    print("Project : Smart City Traffic Forecasting")

    print("Model   : Random Forest Regressor")

    print("Author  : Gaurav Kumar")

    print("Version :", VERSION)

    print("-" * 40)


def main():

    setup_logging()

    show_header()

    while True:

        show_menu()

        choice = input("\nEnter your choice : ").strip()

        try:

            if choice == "1":

                logger.info("Training Started...")

                train_model()

                print("\nTraining Completed Successfully.\n")

            elif choice == "2":

                logger.info("Prediction Started...")

                predict()

                print("\nPrediction Completed Successfully.\n")

            elif choice == "3":

                logger.info("Training Started...")

                train_model()

                logger.info("Prediction Started...")

                predict()

                print("\nTraining and Prediction Completed Successfully.\n")

            elif choice == "4":

                project_info()

            elif choice == "5":

                print("\nThank you for using the project.")

                sys.exit()

            else:

                print("\nInvalid choice.\n")

        except Exception as e:

            logger.exception(e)

            print("\nError:", e)


if __name__ == "__main__":

    main()