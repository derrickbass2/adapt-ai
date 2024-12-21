import base64
import fnmatch
import logging
import os
from io import BytesIO
from typing import Any, List

import numpy as np
import tensorflow as tf  # Use TensorFlow as the base library for Keras

DATA_DIR = 'data'  # Default directory for data
__all__ = ['MNISTClassifier']

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MNISTClassifier:
    def __init__(self, name: str, data_source: str):
        self.name = name
        self.data_source = data_source
        self.model = self.create_model(784)
        logger.info(f"Initialized MNISTClassifier with name: {self.name} and data_source: {self.data_source}")

    @staticmethod
    def create_model(input_dimension: int) -> tf.keras.Model:
        """
        Creates a simple feed-forward neural network for MNIST classification.
        Args:
            input_dimension (int): Input dimension of the model (e.g., flattened MNIST images).

        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        logger.info("Creating model...")
        model = tf.keras.Sequential([
            layers.Dense(units=128, activation='relu', input_shape=(input_dimension,)),
            layers.Dense(units=64, activation='relu'),
            layers.Dense(units=10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logger.info("Model created successfully.")
        return model

    @staticmethod
    def load_data(data_dir: str) -> List[dict]:
        """
        Loads MNIST-like data files from the specified directory.
        Args:
            data_dir (str): Path to the directory containing .npz files.

        Returns:
            List[dict]: List of loaded data dictionaries containing features (X) and labels (y).
        """
        logger.info(f"Loading data from directory: {data_dir}")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} does not exist.")

        data_files = []
        for file in os.listdir(data_dir):
            if fnmatch.fnmatch(file, '*.npz'):
                logger.info(f"Loading file: {file}")
                data = np.load(os.path.join(data_dir, file))
                if 'X' in data and 'y' in data:
                    data_files.append({'X': data['X'], 'y': data['y']})
                else:
                    logger.warning(f"File {file} does not contain 'X' and 'y' keys.")
        if not data_files:
            logger.warning("No valid data files were found in the directory.")
        return data_files

    def train_model(self, data_files: List[dict], epochs: int = 10, batch_size: int = 32):
        """
        Trains the model using the provided data files.
        Args:
            data_files (List[dict]): List of training data files with 'X' and 'y'.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Training batch size. Defaults to 32.
        """
        if not data_files:
            raise ValueError("No data files provided for training.")
        logger.info(f"Starting training for {len(data_files)} data files...")

        for idx, data in enumerate(data_files):
            X = data.get('X')
            y = data.get('y')
            if X is None or y is None:
                logger.error(f"Data file at index {idx} is missing 'X' or 'y'. Skipping...")
                continue

            self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
            logger.info(f"Finished training on data file {idx + 1} of {len(data_files)}.")

    def save_model(self, file_path: str = 'model.h5'):
        """
        Saves the trained model to a specified file path.
        Args:
            file_path (str, optional): Path to save the model. Defaults to 'model.h5'.
        """
        logger.info(f"Saving model to {file_path}...")
        self.model.save(file_path)
        logger.info("Model saved successfully.")

    def load_model(self, file_path: str = 'model.h5'):
        """
        Loads a pre-trained model from a specified file path.
        Args:
            file_path (str, optional): Path to load the model from. Defaults to 'model.h5'.
        """
        logger.info(f"Loading model from {file_path}...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} does not exist.")
        self.model = tf.keras.models.load_model(file_path)
        logger.info("Model loaded successfully.")

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[Any, Any]:
        """
        Evaluates the model on a test dataset.
        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            tuple: Loss and accuracy on the test data.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded or trained before evaluation.")
        logger.info("Evaluating model on test data...")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        logger.info(f"Evaluation completed. Loss: {loss}, Accuracy: {accuracy}")
        return loss, accuracy

    def serialize_model(self) -> str:
        """
        Serializes the model to a base64-encoded string.
        Returns:
            str: Base64 representation of the model.
        """
        logger.info("Serializing model to base64 string...")
        with BytesIO() as model_io:
            tf.keras.models.save_model(self.model, model_io, save_format='h5')
            model_io.seek(0)
            serialized_model = base64.b64encode(model_io.read()).decode()
        logger.info("Model serialized successfully.")
        return serialized_model

    def build_and_train(self, train_data: tuple, val_data: tuple, epochs: int = 10, batch_size: int = 32):
        """
        A shortcut function to build, train, and validate the model.
        Args:
            train_data (tuple): Tuple of (X_train, y_train).
            val_data (tuple): Tuple of (X_val, y_val).
            epochs (int, optional): Number of epochs. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        logger.info("Starting build and train process...")
        if self.model is None:
            self.model = self.create_model(input_dimension=784)

        X_train, y_train = train_data
        X_val, y_val = val_data

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
        logger.info("Model training completed.")

    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input images.
        Args:
            images (np.ndarray): Array of images to predict.

        Returns:
            np.ndarray: Array of predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded or trained before making predictions.")
        logger.info("Predicting on input images...")
        predictions = self.model.predict(images)
        logger.info("Prediction completed.")
        return predictions
