import logging

import numpy as np
from flask import Flask, request, jsonify, abort

from modular_learning_system.neurotech_network.mnist import MNISTClassifier
from modular_learning_system.neurotech_network.neurotech_network_script import NeurotechNetwork

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model instances
mnist_classifier = MNISTClassifier()
neurotech_network = NeurotechNetwork(name="Neurotech Model", data_source="data_source")


def validate_json(required_keys, data):
    """
    Utility function to validate JSON input keys.
    """
    if not data or not isinstance(data, dict):
        abort(400, {"error": "Invalid input or missing JSON payload."})

    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        abort(400, {"error": f"Missing required fields: {', '.join(missing_keys)}"})


@app.route('/train_mnist', methods=['POST'])
def train_mnist():
    """
    Endpoint to train the MNIST classifier.
    """
    try:
        data = request.json
        validate_json(['train_images', 'train_labels', 'val_images', 'val_labels'], data)

        train_images = np.array(data['train_images'])
        train_labels = np.array(data['train_labels'])
        val_images = np.array(data['val_images'])
        val_labels = np.array(data['val_labels'])

        mnist_classifier.build_and_train((train_images, train_labels), (val_images, val_labels))
        logger.info("MNIST model trained successfully.")
        return jsonify({"message": "MNIST model trained successfully."}), 200
    except Exception as e:
        logger.error(f"Error training MNIST model: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route('/predict_mnist', methods=['POST'])
def predict_mnist():
    """
    Endpoint to make predictions with the MNIST classifier.
    """
    try:
        data = request.json
        validate_json(['images'], data)

        images = np.array(data['images'])
        predictions = mnist_classifier.predict(images)
        logger.info("MNIST prediction completed successfully.")
        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        logger.error(f"Error during MNIST prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route('/train_neurotech', methods=['POST'])
def train_neurotech():
    """
    Endpoint to train the Neurotech Network.
    """
    try:
        data = request.json
        validate_json(['data_source'], data)

        data_source = data['data_source']
        data_files = neurotech_network.load_data(data_source)
        neurotech_network.train_model(data_files)
        neurotech_network.save_model()

        logger.info("Neurotech model trained successfully.")
        return jsonify({"message": "Neurotech model trained successfully."}), 200
    except Exception as e:
        logger.error(f"Error training Neurotech model: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route('/predict_neurotech', methods=['POST'])
def predict_neurotech():
    """
    Endpoint to make predictions using the Neurotech Network.
    """
    try:
        data = request.json
        validate_json(['data_source'], data)

        data_source = data['data_source']
        data_files = neurotech_network.load_data(data_source)
        predictions = neurotech_network.predict(data_files)
        logger.info("Neurotech prediction completed successfully.")
        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        logger.error(f"Error during Neurotech prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route('/')
def home():
    """
    A simple home route to test the API.
    """
    logger.info("Home endpoint accessed.")
    return jsonify({"message": "Welcome to the Neurotech Network API"}), 200


if __name__ == '__main__':
    app.run(debug=True)
