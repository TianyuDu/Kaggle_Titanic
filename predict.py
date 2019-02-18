"""
Feb 17 2019
Make Prediction
"""
import numpy as np
import pandas as pd
import data_proc
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
import model_eval
import model_selection
from typing import List
from datetime import datetime
import random
import getpass
import model_builder


PARAM = {
    'epochs': 500,
    'num_features': 7,
    'num_classes': 2,
    'neurons': (512, 512, 512),
    'drop_out': 0.1, 'lr': 0.01
}


if __name__ == "__main__":
    X, y = data_proc.parse_data(
        data_proc.load_data("./data/train.csv"))
    (X_train, X_val, y_train, y_val) = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    c = model_builder.Classifier(PARAM, verbose=True)
    c.fit(X_train, y_train)
    auc = c.evaluate_auc(X_val, y_val)
    print(f"Validation set AUC={auc}")
    c.model.evaluate(X_val, y_val)
    