import numpy as np
from flask import Blueprint, request, jsonify

from modular_learning_system.neurotech_network.neurotech_network_script import NeurotechNetwork

# Initialize the Blueprint
neurotech_network_bp = Blueprint('neurotech_network', __name__)

# Initialize a global instance or handle instantiation properly
engine = NeurotechNetwork("test-adaptai", None)


@neurotech_network_bp.route('/api/neurotech-network/predict', methods=['POST'])
def predict_neurotech_network():
    """
    Endpoint to make predictions using the NeurotechNetwork model.
    Expects JSON input with input_data for prediction.
    """
    try:
        data = request.json['input_data']
        input_data = np.array(data)
        result = engine.predict(input_data)  # Use the instance method
        return jsonify({'prediction': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Additional routes for Neurotech Network can be added here

@neurotech_network_bp.route('/')
def home():
    return jsonify({"message": "Welcome to the Neurotech Network API"})
