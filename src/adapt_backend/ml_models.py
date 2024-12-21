from typing import List, Optional, Tuple

import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def create_neurotech_model(input_shape: Tuple[int], is_classification: bool, num_classes: int = 1) -> Sequential:
    """
    Create and compile a neural network model for regression or classification.

    Args:
        input_shape (Tuple[int]): The shape of the input data (e.g., `(num_features,)`).
        is_classification (bool): Whether the model is for classification.
        num_classes (int): Number of output classes (only applicable for classification).

    Returns:
        Sequential: A compiled Keras model ready for training.
    Raises:
        ValueError: If `num_classes` is invalid for the chosen mode (classification).
    """
    # Validate `num_classes` for classification
    if is_classification and num_classes < 2:
        raise ValueError("For classification tasks, `num_classes` must be 2 or greater.")

    # Initialize the Sequential model
    model = Sequential()

    # Input layer + first hidden layer
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.2))  # Add dropout to prevent overfitting

    # Second hidden layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))  # Add dropout for regularization

    # Output layer
    if is_classification:
        if num_classes == 2:
            # Binary classification
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            # Multi-class classification
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
    else:
        # Regression
        model.add(Dense(1, activation='linear'))
        loss = 'mean_squared_error'
        metrics = ['mae']

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Adam optimizer with default parameters
        loss=loss,
        metrics=metrics
    )

    return model


class RandomForestModel(RandomForestClassifier):
    def __init__(self, n_estimators: int = 100, random_state: Optional[int] = 42):
        """
        Initialize a Random Forest Model with specified hyperparameters.

        Args:
            n_estimators (int): Number of trees in the forest. Default is 100.
            random_state (Optional[int]): Random seed for reproducibility.
        """
        super().__init__(n_estimators=n_estimators, random_state=random_state)

    def fit(self, *args, **kwargs):
        """
        Fit method with dataset size validation.
        Raises:
            ValueError: If the dataset is too small for Random Forest.
        """
        X, y = args
        if len(X) < 10:
            raise ValueError("RandomForestClassifier requires at least 10 samples to generate results effectively.")
        super().fit(*args, **kwargs)


class SVMModel(SVC):
    def __init__(self, kernel: str = 'rbf', random_state: Optional[int] = None, **kwargs):
        """
        Initialize an SVM model with specified kernel and random state.

        Args:
            kernel (str): Type of kernel to use in the SVM (e.g., 'linear', 'rbf'). Default is 'rbf'.
            random_state (Optional[int]): Random seed for reproducibility. Not supported for all kernels.
        """
        super().__init__(kernel=kernel, **kwargs)
        self.random_state = random_state

    def fit(self, *args, **kwargs):
        """
        Fit method with scalability warning.
        Raises:
            ValueError: If the dataset size is too large for SVM.
        """
        X, _ = args
        if len(X) > 10000:
            raise ValueError(
                "SVM does not scale well for datasets larger than 10,000 samples. Consider using LinearSVC.")
        super().fit(*args, **kwargs)


class AA_Genome:
    def __init__(self):
        """
        Genetic Algorithm/Model Handling Class.

        Attributes:
            best_solution (Optional[np.ndarray]): Placeholder for the best solution of the genome model.
        """
        self.best_solution: Optional[np.ndarray] = None

    @staticmethod
    def train_AA_genome_model(data: np.ndarray, labels: np.ndarray):
        """
        Placeholder for training the AA Genome model. Replace this logic with actual implementation.

        Args:
            data (np.ndarray): Input feature data for training.
            labels (np.ndarray): Corresponding labels for training.
        """
        # Validate input data
        if not isinstance(data, np.ndarray) or not isinstance(labels, np.ndarray):
            raise ValueError("Both data and labels must be NumPy arrays.")

        # Add your logic for training the genome model here
        raise NotImplementedError("train_AA_genome_model logic is not yet implemented.")

    def get_best_solution(self) -> np.ndarray:
        """
        Return the best solution found by the AA Genome model.

        Returns:
            np.ndarray: Best solution.
        Raises:
            ValueError: If no solution has been generated yet.
        """
        if self.best_solution is None:
            raise ValueError("No solution has been generated yet.")
        return self.best_solution

    @staticmethod
    def _convert_to_numpy(cleaned_data: List[List[float]], feature_cols: List[int]) -> np.ndarray:
        """
        Helper function to convert cleaned input data into a NumPy array.

        Args:
            cleaned_data (List[List[float]]): List of cleaned data points.
            feature_cols (List[int]): List of feature column indices to include.

        Returns:
            np.ndarray: Converted NumPy array of cleaned data.
        Raises:
            ValueError: If there are issues during data conversion.
        """
        # Validate types and structure
        if not all(isinstance(row, list) for row in cleaned_data):
            raise ValueError("`cleaned_data` must be a list of lists.")
        if not all(isinstance(col, int) for col in feature_cols):
            raise ValueError("`feature_cols` must be a list of integers.")

        # Perform conversion
        try:
            return np.array([
                [row[col] for col in feature_cols]
                for row in cleaned_data
            ])
        except Exception as e:
            raise ValueError(f"Data conversion failed: {e}")
