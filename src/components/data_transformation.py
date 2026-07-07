import os
import joblib
import pandas as pd
import numpy as np

from src.utils import read_yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataTransformation:
    """
    Handles feature engineering and preprocessing.
    """

    def __init__(self):
       config = read_yaml("config/config.yaml")

       self.train_path = config["data_ingestion"]["train_data_path"]
       self.test_path = config["data_ingestion"]["test_data_path"]

       self.target_column = config["features"]["target_column"]

       self.numerical_cols = config["features"]["numerical_features"]
       self.categorical_cols = config["features"]["categorical_features"]

       self.preprocessor_path = config["data_transformation"]["preprocessor_path"]
       
       self.x_train_path = config["data_transformation"]["x_train_path"]
       self.x_test_path = config["data_transformation"]["x_test_path"]
       self.y_train_path = config["data_transformation"]["y_train_path"]
       self.y_test_path = config["data_transformation"]["y_test_path"]

    def initiate_data_transformation(self):

        print("Starting Data Transformation...")

        # Load train and test datasets
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        print("Train Shape:", train_df.shape)
        print("Test Shape :", test_df.shape)

        # ==========================
        # Feature Engineering
        # ==========================

        for df in [train_df, test_df]:

            df["hour"] = pd.to_datetime(
                df["time_ist"],
                format="%H:%M"
            ).dt.hour

            df["date_ist"] = pd.to_datetime(
                df["date_ist"],
                format="%d/%m/%Y"
            )

            df["month"] = df["date_ist"].dt.month

            df.drop(
                columns=[
                    "time_ist",
                    "date_ist",
                    "description",
                    "lat",
                    "lon"
                ],
                inplace=True
            )

        # ==========================
        # Split Features & Target
        # ==========================

        X_train = train_df.drop(self.target_column, axis=1)
        y_train = train_df[self.target_column]

        X_test = test_df.drop(self.target_column, axis=1)
        y_test = test_df[self.target_column]


        # ==========================
        # Preprocessor
        # ==========================

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    self.numerical_cols
                ),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore"
                    ),
                    self.categorical_cols
                )
            ]
        )

        print("Fitting preprocessor...")

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # ==========================
        # Save Transformed Data
        # ==========================

        np.save(
            self.x_train_path,
            X_train_processed
        )

        np.save(
            self.x_test_path,
            X_test_processed
        )

        np.save(
            self.y_train_path,
            y_train.to_numpy()
        )

        np.save(
            self.y_test_path,
            y_test.to_numpy()
        )

        # ==========================
        # Save Preprocessor
        # ==========================

        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(
            preprocessor,
            self.preprocessor_path
        )

        print("Preprocessor saved successfully.")

        print("Transformation Completed.")

        return (
            self.x_train_path,
            self.x_test_path,
            self.y_train_path,
            self.y_test_path,
            self.preprocessor_path
        )


if __name__ == "__main__":

    transformer = DataTransformation()

    x_train_path, x_test_path, y_train_path, y_test_path, preprocessor = (
        transformer.initiate_data_transformation()
    )

    print("\nData Transformation Completed Successfully!")

    print("X Train :", x_train_path)
    print("X Test  :", x_test_path)
    print("Y Train :", y_train_path)
    print("Y Test  :", y_test_path)