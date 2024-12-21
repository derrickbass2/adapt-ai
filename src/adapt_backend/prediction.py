from typing import List, Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


def make_predictions(model: BaseEstimator, trained_model: Any, vectorizer: Any, new_data: List[str]) -> np.ndarray:
    """
    Make predictions on new data using a trained model and a vectorizer.

    Args:
        model (BaseEstimator): The model class (used for validation purposes).
        trained_model (Any): The trained model instance.
        vectorizer (Any): The vectorizer instance (e.g., TF-IDF or other feature preprocessors).
        new_data (List[str]): A list of new data points (strings) for prediction.

    Returns:
        np.ndarray: The predictions made by the trained model on the transformed data.

    Raises:
        ValueError: If new_data is empty or not a list.
        NotFittedError: If either the vectorizer or the trained model is not fitted.
    """
    # Validate input data
    if not isinstance(new_data, list) or len(new_data) == 0:
        raise ValueError("The 'new_data' parameter should be a non-empty list of strings.")

    # Ensure the vectorizer is fitted
    try:
        transformed_data = vectorizer.transform(new_data)
    except NotFittedError as e:
        raise NotFittedError("The vectorizer is not fitted. Ensure it is trained before passing it here.") from e

    # Ensure the trained model is fitted
    if not hasattr(trained_model, "predict"):
        raise ValueError("The provided trained model does not have a 'predict' method.")

    try:
        predictions = trained_model.predict(transformed_data)
    except NotFittedError as e:
        raise NotFittedError("The trained model is not fitted. Fit the model before using it for predictions.") from e

    # Validate the predictions output
    if not isinstance(predictions, np.ndarray):
        raise TypeError("The trained model should return predictions as a NumPy array.")

    return predictions
