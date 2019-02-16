"""
Feb 16 2019
Model builders with tensorflow
Model builders are wrapped into class and takes parameter dictionary as input.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import model_eval


SAMPLE_PARAM = {
    "epochs": 100,
    "num_features": 6,
    "num_classes": 2,  # number of label classes
    "neurons": (64, 128, 256),
    "drop_out": 0.2,
    "lr": 0.01
}


class Classifier():
    def __init__(self, param: dict = SAMPLE_PARAM) -> None:
        print("Loading Parameters...")
        self.__dict__.update(param)

    def build_model(self):
        print("Building Model Layers...")
        layers = [
            tf.keras.layers.Dense(x, activation="relu")
            for x in self.neurons
        ]
        self.model = tf.keras.Sequential(layers)
        self.model.add(
            tf.keras.layers.Dropout(self.drop_out)
        )

        self.model.add(
            tf.keras.layers.Dense(self.num_classes, activation="softmax")
        )

    def compile_model(self):
        print("Compiling Model...")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.Accuracy]
        )

    def fit(self, X_train, y_train, validation_split=0.1):
        self.X_train, self.y_train = X_train, y_train
        self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            verbose=0,
            epochs=self.epochs
        )

    def evaluate_auc(self, X_test, y_test):
        pred_prob = self.model.predict_proba(X_test)
        pred_class = self.model.predict_classes(X_test)
        roc_auc = self.model_eval.auc(y_test, pred_prob[:, 0], 0)
        return roc_auc


if __name__ == "__main__":
    c = Classifier()
    c.build_model()
    c.compile_model()
