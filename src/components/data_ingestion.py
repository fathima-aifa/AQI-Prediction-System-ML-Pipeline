import os
import pandas as pd
from src.utils import read_yaml
from sklearn.model_selection import train_test_split


class DataIngestion:
    """
    Handles data loading and train-test splitting.
    """

    def __init__(self):
        config = read_yaml("config/config.yaml")

        self.raw_data_path = config["data"]["raw_data_path"]
        self.train_data_path = config["data"]["train_data_path"]
        self.test_data_path = config["data"]["test_data_path"]

        self.test_size = config["model"]["test_size"]
        self.random_state = config["model"]["random_state"]

    def initiate_data_ingestion(self):
        print("Starting Data Ingestion...")

        # Load dataset
        df = pd.read_csv(self.raw_data_path)
        print(f"Dataset loaded successfully.")
        print(f"Shape: {df.shape}")

        # Split dataset
        train_df, test_df = train_test_split(
            df,
            test_size=0.20,
            random_state=self.random_state
        )

        # Create processed folder if it doesn't exist
        os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

        # Save train and test data
        train_df.to_csv(self.train_data_path, index=False)
        test_df.to_csv(self.test_data_path, index=False)

        print("Train and Test datasets saved successfully.")
        print(f"Train Shape: {train_df.shape}")
        print(f"Test Shape: {test_df.shape}")

        return (
            self.train_data_path,
            self.test_data_path
        )


if __name__ == "__main__":
    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    print("\nData Ingestion Completed Successfully!")
    print("Train File:", train_path)
    print("Test File :", test_path)