import numpy as np
from flask import Blueprint, request, jsonify

from adapt_backend.ml_models import create_neurotech_model

# Initialize the Blueprint
neurotech_network_bp = Blueprint("neurotech_network", __name__)

# In-memory model storage for simplicity (not ideal for production systems)
neural_network_model = None


@neurotech_network_bp.route("/train", methods=["POST"])
def train_neural_network():
    """
    Endpoint to train a neural network model using the Neurotech model.
    Expects JSON input with `features`, `labels`, `is_classification`, and `num_classes`.
    """
    global neural_network_model
    try:
        # Parse input JSON data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract input data and model parameters
        features = np.array(data.get("features"))
        labels = np.array(data.get("labels"))
        is_classification = data.get("is_classification", True)
        num_classes = data.get("num_classes", 2)
        epochs = data.get("epochs", 10)
        batch_size = data.get("batch_size", 32)

        # Validate features and labels
        if features.size == 0 or labels.size == 0:
            return jsonify({"error": "Features and labels cannot be empty."}), 400
        if len(features) != len(labels):
            return jsonify({"error": "Number of features and labels must match."}), 400

        # Create a neural network model
        input_shape = (features.shape[1],)  # Assuming features are 2D (samples, features)
        neural_network_model = create_neurotech_model(input_shape, is_classification, num_classes)

        # Train the model
        neural_network_model.fit(features, labels, epochs=epochs, batch_size=batch_size, verbose=1)

        return jsonify({"message": "Model trained successfully"}), 200

    except ValueError as e:
        # Handle value-specific issues (e.g., invalid parameters)
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        # Handle any other errors
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@neurotech_network_bp.route("/predict", methods=["POST"])
def predict_with_neural_network():
    """
    Endpoint to make predictions using the trained neural network model.
    Expects JSON input with `features`.
    """
    global neural_network_model
    try:
        if neural_network_model is None:
            return jsonify({"error": "Model has not been trained yet."}), 400

        # Parse features from the request
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        features = np.array(data.get("features"))

        # Validate features
        if features.ndim != 2:
            return jsonify({"error": "Features must be a 2-dimensional array."}), 400

        # Make predictions
        predictions = neural_network_model.predict(features)

        # Convert predictions to list for JSON serialization
        return jsonify({"predictions": predictions.tolist()}), 200

    except ValueError as e:
        # Handle value-specific issues
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@neurotech_network_bp.route("/")
def home():
    """
    A simple home route for the Neurotech API.
    """
    return jsonify({"message": "Welcome to the Neurotech Network API!"}), 200
