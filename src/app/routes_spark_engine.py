from flask import Blueprint, request, jsonify
from modular_learning_system.spark_engine import spark_engine_script

spark_engine_bp = Blueprint('spark_engine', __name__)

@spark_engine_bp.route('/api/spark-engine/run', methods=['POST'])
def run_spark_engine():
    data = request.json
    result = spark_engine_script.run_algorithm(data)  # Call function from spark_engine_script.py
    return jsonify(result)

