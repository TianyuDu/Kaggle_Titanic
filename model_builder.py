"""
Feb 16 2019
Model builders with tensorflow
Model builders are wrapped into class and takes parameter dictionary as input.
"""
import numpy as np
import pandas as pd
import tensorflow as tf


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

    def fit(self, X_train, y_train):
        self.model.fit(
            X_train, y_train,
            validation_split=0.1,
            verbose=0,
            epochs=self.epochs
        )


if __name__ == "__main__":
    c = Classifier()
    c.build_model()
    c.compile_model()
