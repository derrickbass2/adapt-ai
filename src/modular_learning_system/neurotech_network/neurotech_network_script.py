import fnmatch
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

DATAPATH = '/Users/derrickbass/Public/adaptai/datasets/'
MODELPATH = '/Users/derrickbass/Public/adaptai/neurotech_model.h5'


class NeurotechNetwork:
    def __init__(self, name: str, data_source: str):
        """
        Initialize the NeurotechNetwork instance.

        :param name: Name of the model
        :param data_source: Source directory for data
        """
        self.name = name
        self.data_source = data_source
        self.model = self.create_model(10)  # Assuming X_dimension is 10

    @staticmethod
    def create_model(X_dimension: int) -> tf.keras.Model:
        """
        Create a TensorFlow model for regression.

        :param X_dimension: Number of input features
        :return: Compiled TensorFlow model
        """
        model = tf.keras.Sequential([
            layers.Dense(units=64, input_shape=(X_dimension,)),
            layers.LeakyReLU(alpha=0.1),
            layers.Dense(units=32),
            layers.LeakyReLU(alpha=0.1),
            layers.Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
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
            self.model.fit(X, y, epochs=10)

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

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate the model on test data.

        :param X_test: Test features
        :param y_test: True labels
        :return: Loss value
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before evaluation.")
        return self.model.evaluate(X_test, y_test)

    def serialize_model(self) -> str:
        """
        Serialize the model to a base64 string.
        """
        import base64
        from io import BytesIO

        with BytesIO() as model_io:
            tf.keras.models.save_model(self.model, model_io)
            model_io.seek(0)
            return base64.b64encode(model_io.read()).decode('utf-8')

    def predict(self, data_files):
        pass
