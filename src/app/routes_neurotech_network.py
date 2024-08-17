from flask import Blueprint, request, jsonify
from modular_learning_system.neurotech_network import neurotech_network_script

neurotech_network_bp = Blueprint('neurotech_network', __name__)

@neurotech_network_bp.route('/api/neurotech-network/predict', methods=['POST'])
def predict_neurotech_network():
    data = request.json
    result = neurotech_network_script.predict(data)  # Call function from neurotech_network_script.py
    return jsonify(result)

