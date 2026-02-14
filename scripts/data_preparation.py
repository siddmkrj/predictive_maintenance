import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_REPO = "mukherjee78/predictive-maintenance-engine-data"
RAW_DATA_FILE = "raw_data.csv"
PROCESSED_DIR = "data/processed"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


def standardize_column_names(df: pd.DataFrame) -> pd.Index:
    """Standardize column names: strip, lower, replace spaces with underscores."""
    return df.columns.str.strip().str.lower().str.replace(" ", "_")


def main():
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset(DATASET_REPO, data_files=RAW_DATA_FILE)
    df = dataset["train"].to_pandas()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    df.columns = standardize_column_names(df)

    selected_columns = [
        "engine_rpm",
        "lub_oil_pressure",
        "fuel_pressure",
        "coolant_pressure",
        "lub_oil_temp",
        "coolant_temp",
        "engine_condition",
    ]
    df_clean = df[selected_columns].copy()

    missing = df_clean.isnull().sum().sum()
    print(f"Missing values: {missing}")

    X = df_clean.drop(columns=["engine_condition"])
    y = df_clean["engine_condition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print(f"Train set: {train_df.shape[0]} rows")
    print(f"Test set:  {test_df.shape[0]} rows")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Saved train/test to {TRAIN_PATH}, {TEST_PATH}")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=TRAIN_PATH,
        path_in_repo="train.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=TEST_PATH,
        path_in_repo="test.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print("Train/test datasets uploaded to Hugging Face.")


if __name__ == "__main__":
    main()
