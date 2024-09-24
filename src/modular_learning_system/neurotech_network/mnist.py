import base64
import fnmatch
import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

__all__ = ['MNISTClassifier']

# Define constants for data and model paths


DATAPATH = '/Users/derrickbass/Public/adaptai/datasets/'
MODELPATH = '/Users/derrickbass/Public/adaptai/neurotech_model.h5'


class MNISTClassifier:
    def __init__(self, name: str, data_source: str):
        """
        Initialize the MNISTClassifier instance.

        :param name: Name of the model
        :param data_source: Source directory for data
        """
        self.name = name
        self.data_source = data_source
        self.model = self.create_model(784)  # Input dimension for MNIST is 784 (28x28)

    @staticmethod
    def create_model(input_dimension: int) -> tf.keras.Model:
        """
        Create a TensorFlow model for classification.

        :param input_dimension: Number of input features
        :return: Compiled TensorFlow model
        """
        model = tf.keras.Sequential([
            layers.Dense(units=128, activation='relu', input_shape=(input_dimension,)),
            layers.Dense(units=64, activation='relu'),
            layers.Dense(units=10, activation='softmax')  # Output for 10 classes (digits 0-9)
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def load_data(data_dir: str) -> list:
        """
        Load data files from a directory.

        :param data_dir: Directory containing data files
        :return: List of data arrays
        """
        data_files = []
        for file in os.listdir(data_dir):
            if fnmatch.fnmatch(file, '*.npz'):
                data = np.load(os.path.join(data_dir, file))
                data_files.append(data)
        return data_files

    def train_model(self, data_files: list):
        """
        Train the model using data files.

        :param data_files: List of data arrays
        """
        if not data_files:
            raise ValueError("No data files provided for training.")

        for data in data_files:
            X = data['X']
            y = data['y']
            self.model.fit(X, y, epochs=10, batch_size=32)  # Added batch_size for training stability

    def save_model(self):
        """
        Save the model to a file.
        """
        self.model.save(MODELPATH)

    def load_model(self):
        """
        Load a model from a file.
        """
        self.model = tf.keras.models.load_model(MODELPATH)

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the model on test data.

        :param X_test: Test features
        :param y_test: True labels
        :return: Loss and accuracy values
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before evaluation.")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def serialize_model(self) -> str:
        """
        Serialize the model to a base64 string.
        """
        with BytesIO() as model_io:
            tf.keras.models.save_model(self.model, model_io)
            model_io.seek(0)
            return base64.b64encode(model_io.read()).decode('utf-8')
