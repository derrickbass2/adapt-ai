import os
from uuid import uuid4

import pandas as pd
from flask import Blueprint, request, jsonify

from modular_learning_system.spark_engine.spark_engine import SparkEngine

spark_engine_bp = Blueprint("spark_engine", __name__)

# Initialize the SparkEngine instance
spark_engine = SparkEngine()


def validate_file_path(file_path):
    """Validate if a file exists and is accessible."""
    if not file_path or not isinstance(file_path, str):
        return jsonify({"error": "'file_path' must be a non-empty string"}), 400
    if not os.path.exists(file_path):
        return jsonify({"error": f"The file path '{file_path}' does not exist"}), 400
    return None  # No error


def validate_list_param(param, param_name):
    """Validate if a parameter is a list."""
    if not isinstance(param, list):
        return jsonify({"error": f"'{param_name}' must be a list"}), 400
    return None  # No error


def generate_unique_output_path(default_path):
    """Generate a unique output path by appending a UUID."""
    base, ext = os.path.splitext(default_path)
    return f"{base}-{uuid4().hex}{ext}"


@spark_engine_bp.route("/preprocess", methods=["POST"])
def preprocess_data():
    """
    Endpoint to preprocess data using SparkEngine.
    Expects JSON input with file_path, feature_cols, and label_col.
    """
    try:
        # Parse input data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract and validate parameters
        file_path = data.get("file_path")
        feature_cols = data.get("feature_cols", [])
        label_col = data.get("label_col", None)
        output_path = data.get("output_path", "/tmp/processed_data.parquet")

        # Validate input
        error = validate_file_path(file_path)
        if error:
            return error
        error = validate_list_param(feature_cols, "feature_cols")
        if error:
            return error

        # Generate unique output path
        output_path = generate_unique_output_path(output_path)

        # Perform preprocessing
        df = spark_engine.read_csv(file_path)
        processed_df = spark_engine.preprocess_data(df, feature_cols, label_col)
        spark_engine.write_parquet(processed_df, output_path)

        return jsonify({"message": "Data processed successfully.", "output_path": output_path}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@spark_engine_bp.route("/cluster", methods=["POST"])
def cluster_data():
    """
    Endpoint to cluster data using SparkEngine.
    Expects JSON input with file_path and num_clusters.
    """
    try:
        # Parse input data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract and validate parameters
        file_path = data.get("file_path")
        num_clusters = data.get("num_clusters", 3)
        output_path = data.get("output_path", "/tmp/clustered_data.parquet")

        if not isinstance(num_clusters, int) or num_clusters <= 0:
            return jsonify({"error": "'num_clusters' must be a positive integer"}), 400

        # Validate input
        error = validate_file_path(file_path)
        if error:
            return error

        # Generate unique output path
        output_path = generate_unique_output_path(output_path)

        # Perform clustering
        df = spark_engine.read_csv(file_path)
        df_preprocessed = spark_engine.preprocess_data(df, [], None)
        clustered_df = spark_engine.cluster_data(df_preprocessed, num_clusters)
        spark_engine.write_parquet(clustered_df, output_path)

        return jsonify({"message": "Data clustered successfully.", "output_path": output_path}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@spark_engine_bp.route("/train", methods=["POST"])
def train_model():
    """
    Endpoint to train a model using SparkEngine.
    Expects JSON input with file_path and label_col.
    """
    try:
        # Parse input data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract and validate parameters
        file_path = data.get("file_path")
        label_col = data.get("label_col")
        model_path = data.get("model_path", "/tmp/model")

        if not label_col or not isinstance(label_col, str):
            return jsonify({"error": "'label_col' must be a non-empty string"}), 400

        # Validate input
        error = validate_file_path(file_path)
        if error:
            return error

        # Generate unique model path
        model_path = generate_unique_output_path(model_path)

        # Perform model training
        df = spark_engine.read_csv(file_path)
        df_preprocessed = spark_engine.preprocess_data(df, [], label_col)
        model = spark_engine.train_model(df_preprocessed, label_col)
        model.save(model_path)

        return jsonify({"message": "Model trained successfully.", "model_path": model_path}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@spark_engine_bp.route("/predict", methods=["POST"])
def predict_data():
    """
    Endpoint to make predictions using a pre-trained model.
    Expects JSON input with model_path and file_path.
    """
    try:
        # Parse input data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract and validate parameters
        model_path = data.get("model_path")
        file_path = data.get("file_path")

        error = validate_file_path(file_path)
        if error:
            return error
        if not model_path or not os.path.exists(model_path):
            return jsonify({"error": f"The model path '{model_path}' does not exist"}), 400

        # Perform predictions
        df = spark_engine.read_csv(file_path)
        predictions = spark_engine.predict(df, model_path)

        # Serialize predictions to JSON-compatible format
        if isinstance(predictions, pd.DataFrame):  # Example for pandas DataFrame output
            predictions = predictions.to_dict(orient="records")

        return jsonify({"predictions": predictions}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
