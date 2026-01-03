import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


RAW_DATA_PATH = "data/raw/diabetes.csv"
PROCESSED_DATA_DIR = "data/processed"
SCALER_PATH = "models/scaler.pkl"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cols_with_zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].replace(0, np.nan)

    # Fill missing values with median (robust for healthcare data)
    for col in cols_with_zero_as_missing:
        df[col].fillna(df[col].median(), inplace=True)

    return df


def split_and_scale(df: pd.DataFrame):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_outputs(X_train, X_test, y_train, y_test, scaler):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    pd.DataFrame(X_train).to_csv(f"{PROCESSED_DATA_DIR}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{PROCESSED_DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DATA_DIR}/y_test.csv", index=False)

    joblib.dump(scaler, SCALER_PATH)


def preprocess_pipeline():
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    save_outputs(X_train, X_test, y_train, y_test, scaler)


if __name__ == "__main__":
    preprocess_pipeline()

