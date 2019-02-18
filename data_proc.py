"""
Feb. 15 2019
Data Preprocessing Script
"""
import os
import numpy as np
import pandas as pd


def load_data(file_dir: str) -> pd.DataFrame:
    raw = pd.read_csv(file_dir)
    df = raw.drop(columns=[
        "PassengerId", "Ticket"])
    return df


def create_features(raw: pd.DataFrame):
    df = raw.copy()
    df["Cabin"] = pd.notna(df["Cabin"]).astype(np.float32)
    # True=have cabin
    df["Sex"] = (df["Sex"] == "female").astype(np.float32)
    # False=male; True=Female
    df['Age'] = df['Age'].fillna(df['Age'].median())

    df = df.fillna({"Embarked": "S"})
    embarked_mapping = {'S': 1, 'C': 2, 'Q': 3}
    df['Embarked'] = df['Embarked'].map(embarked_mapping)

    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    # title_mapping = {'Master': 1, 'Ms': 2, 'Mlle': 2, 'the Countess': 3, 'Capt': 4, 'Rev': 4, 'Mrs': 3, 'Col': 4, 
    # 'Miss': 2, 'Lady': 3, 'Mr': 4, 'Sir': 4, 'Mme': 3, 'Jonkheer': 4, 'Major': 4, 'Don': 4, 'Dr': 5}
    df = change_title(df)
    title_mapping = {"Rare": 1, "Miss": 2, "Mrs": 3, "Mr": 4}
    df['Title'] = df['Title'].map(title_mapping)
    # df['Family_Size'] = df['SibSp'] + df['Parch']
    df = discrete_age(df)
    df = df.dropna()
    return df

def discrete_age(dataset: pd.DataFrame):
    dataset.loc[dataset["Age"] <= 9, "Age"] = 0
    dataset.loc[(dataset["Age"] > 9) & (dataset["Age"] <= 19), "Age"] = 1
    dataset.loc[(dataset["Age"] > 19) & (dataset["Age"] <= 29), "Age"] = 2
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[dataset["Age"] > 39, "Age"] = 4
    return dataset

def change_title(dataset: pd.DataFrame):
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    return dataset
    

def parse_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Extract data into feature array and target array.
    """
    df = create_features(df)
    X = df.drop(columns=["Survived", "Name"]).values.astype(np.float32)
    y = df["Survived"].values
    y = y.reshape(-1, 1)
    return X, y


if __name__ == "__main__":
    TRAIN_DATA = "./data/train.csv"
    TEST_DATA = "./data/test.csv"
    df = load_data(TRAIN_DATA)
    X, y = parse_data(df)
    print(f"X shape={X.shape}\ny shape={y.shape}")
