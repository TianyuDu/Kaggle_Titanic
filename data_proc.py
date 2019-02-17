"""
Feb. 15 2019
Data Preprocessing Script
"""
import os
import numpy as np
import pandas as pd


def load_data(file_dir: str) -> pd.DataFrame:
    raw = pd.read_csv(file_dir)
    df = raw.drop(columns=["PassengerId", "Name",
                           "Ticket", "Cabin", "Embarked"])
    df["Cabin"] = pd.notna(raw["Cabin"]).astype(np.float32)
    df.dropna(inplace=True)
    return df


def parse_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    df["Sex"] = (df["Sex"] == "female")  # 0=male; 1=Female
    X = df.drop(columns=["Survived"]).values.astype(np.float64)
    y = df["Survived"].values
    y = y.reshape(-1, 1)
    return X, y


if __name__ == "__main__":
    TRAIN_DATA = "./data/train.csv"
    TEST_DATA = "./data/test.csv"
    df = load_data(TRAIN_DATA)
    X, y = parse_data(df)
    print(f"X shape={X.shape}\ny shape={y.shape}")
