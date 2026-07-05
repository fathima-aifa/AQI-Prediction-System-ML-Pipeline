import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    """
    Handles data loading and train-test splitting.
    """

    def __init__(self):
        self.raw_data_path = "data/raw/data.csv"
        self.train_data_path = "data/processed/train.csv"
        self.test_data_path = "data/processed/test.csv"

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
            random_state=42
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