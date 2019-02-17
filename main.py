"""
Feb 15 2019
Main File
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


PARAM_SRC_TEST = {
    "epochs": [50 * x for x in range(1, 2)],
    "num_features": 7,
    "num_classes": 2,  # number of label classes
    "neurons": [
        (64, 128),
        (64, 128, 256)
    ],
    "drop_out": [0.1 * x for x in range(1, 2)],
    "lr": [0.01, 0.03]
}

PARAM_SRC = {
    "epochs": [50 * x for x in range(1, 5)],
    "num_features": 7,
    "num_classes": 2,  # number of label classes
    "neurons": [
        (64, 128),
        (128, 256),
        (256, 512),
        (64, 128, 256),
        (128, 256, 512)
    ],
    "drop_out": [0.1 * x for x in range(1, 4)],
    "lr": [0.01, 0.03, 0.1, 0.3]
}


def save_result(candidates: List[dict]) -> None:
    now = str(datetime.now())
    with open(f"./hps/{now}.txt", "w") as f:
        f.writelines(
            [str(r) + "\n" for r in candidates]
        )
    print(f"Reseult written to ./hps/{now}.txt")


if __name__ == "__main__":
    start = datetime.now()
    X, y = data_proc.parse_data(
        data_proc.load_data("./data/train.csv"))

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.2, shuffle=True)

    PARAM_SET = model_selection.gen_hparam_set(PARAM_SRC_TEST)
    random.shuffle(PARAM_SET)
    candidates = model_selection.grid_search(
        PARAM_SET,
        X_train, X_test,
        y_train, y_test
    )
    save_result(candidates)
    end = datetime.now()
    print(f"Time taken for [{len(PARAM_SET)}] HP: {end - start}")

    # # Prediction with Keras based model
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(128, activation="relu"))
    # model.add(tf.keras.layers.Dense(256, activation="relu"))
    # # model.add(tf.keras.layers.Dense(256, activation="relu"))
    # model.add(tf.keras.layers.Dense(2, activation="softmax"))
    # model.compile(optimizer="adam",
    #               loss="sparse_categorical_crossentropy",
    #               metrics=["accuracy"])
    # print("Fitting...")
    # model.fit(
    #     X_train, y_train,
    #     validation_split=0.1,
    #     verbose=0,
    #     epochs=150)
    # print("Test set evaluation")
    # print(model.evaluate(X_test, y_test))
    # pred_prob = model.predict_proba(X_test)
    # pred_class = model.predict_classes(X_test)
    # roc_auc = model_eval.auc(y_test, pred_prob[:, 0], 0)
    # print(f"AUC = {roc_auc}")
    # print(f"Train set density: {np.mean(y_test)}")
    # print(f"Test set density: {np.mean(y_test)}")
