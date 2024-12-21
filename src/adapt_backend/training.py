import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report


def train_and_evaluate(model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray, problem_type: str = "classification") -> float:
    """
    Train and evaluate a model on given training and testing datasets.

    Args:
        model (BaseEstimator): The machine learning model (conforms to scikit-learn API).
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training target labels.
        X_test (np.ndarray): Testing feature set.
        y_test (np.ndarray): Testing target labels.
        problem_type (str): Define the type of problem: "classification" or "regression".

    Returns:
        float: Evaluation result (accuracy for classification, RMSE for regression).
    """
    # Step 1: Model Training
    print("Training the model...")
    model.fit(X_train, y_train)

    # Step 2: Predictions
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)

    # Step 3: Evaluation
    if problem_type == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print(f"Model Accuracy: {accuracy:.4f}")
        return accuracy

    elif problem_type == "regression":
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        return rmse

    else:
        raise ValueError("Invalid problem_type. Please specify 'classification' or 'regression'.")
