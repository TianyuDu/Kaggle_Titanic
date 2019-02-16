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


if __name__ == "__main__":
    X, y = data_proc.parse_data(
        data_proc.load_data("./data/train.csv"))

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.2, shuffle=True)

    # Prediction with Keras based model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    # model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print("Fitting...")
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        verbose=0,
        epochs=150)
    print("Test set evaluation")
    model.evaluate(X_test, y_test)
    pred_prob = model.predict_proba(X_test)
    pred_class = model.predict_classes(X_test)
    roc_auc = model_eval.auc(y_test, pred_prob[:, 0], 0)
    print(f"AUC = {roc_auc}")
    print(f"Train set density: {np.mean(y_test)}")
    print(f"Test set density: {np.mean(y_test)}")
