import base64
import fnmatch
import os
from io import BytesIO
from typing import Any, List

import numpy as np
import tensorflow as tf  # Use TensorFlow as the base library for Keras
from tensorflow.keras import layers  # Import layers from tf.keras, not standalone keras

DATA_DIR = 'str'
__all__ = ['MNISTClassifier']


class MNISTClassifier:
    def __init__(self, name: str, data_source: str):
        self.name = name
        self.data_source = data_source
        self.model = self.create_model(784)

    @staticmethod
    def create_model(input_dimension: int) -> tf.keras.Model:
        model = tf.keras.Sequential([  # Ensure using tf.keras.Sequential
            layers.Dense(units=128, activation='relu', input_shape=(input_dimension,)),
            layers.Dense(units=64, activation='relu'),
            layers.Dense(units=10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def load_data(data_dir: str) -> List[Any]:
        data_files = []
        for file in os.listdir(data_dir):
            if fnmatch.fnmatch(file, '*.npz'):
                data = np.load(os.path.join(data_dir, file))
                data_files.append(data)
        return data_files

    def train_model(self, data_files: List[dict], epochs: int = 10, batch_size: int = 32):
        if not data_files:
            raise ValueError("No data files provided for training.")

        for data in data_files:
            X = data['X']
            y = data['y']
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def save_model(self):
        self.model.save('model.h5')

    def load_model(self):
        if not os.path.exists('model.h5'):
            raise FileNotFoundError("Model file not found.")
        self.model = tf.keras.models.load_model('model.h5')  # Corrected tf.keras.models

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[Any, Any]:
        if self.model is None:
            raise RuntimeError("Model must be loaded or trained before evaluation.")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def serialize_model(self) -> str:
        with BytesIO() as model_io:
            tf.keras.models.save_model(self.model, model_io)  # Corrected tf.keras.models.save_model
            model_io.seek(0)
            return base64.b64encode(model_io.read()).decode()
