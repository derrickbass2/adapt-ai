from flask import Blueprint, request, jsonify
from modular_learning_system.neurotech_network import neurotech_network_script
import numpy as np

neurotech_network_bp = Blueprint('neurotech_network', __name__)


@neurotech_network_bp.route('/api/neurotech-network/predict', methods=['POST'])
def predict_neurotech_network():
    try:
        data = request.json['input_data']  # Assumes input data is in JSON format
        input_data = np.array(data)
        result = neurotech_network_script.predict(input_data)  # Call function from neurotech_network_script.py
        return jsonify({'prediction': result.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
