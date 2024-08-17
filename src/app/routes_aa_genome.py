from flask import Blueprint, request, jsonify
from modular_learning_system.aa_genome import aa_genome_script

aa_genome_bp = Blueprint('aa_genome', __name__)

@aa_genome_bp.route('/api/aa-genome/run', methods=['POST'])
def run_aa_genome():
    data = request.json
    result = aa_genome_script.run_algorithm(data)  # Call function from aa_genome_script.py
    return jsonify(result)

