from flask import Blueprint, request, jsonify

from modular_learning_system.spark_engine.spark_engine import SparkEngine

# Initialize the Blueprint
spark_engine_bp = Blueprint("spark_engine", __name__)

# Initialize the SparkEngine instance
spark_engine = SparkEngine()


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

        if not file_path:
            return jsonify({"error": "Missing required parameter 'file_path'"}), 400

        # Perform preprocessing
        df = spark_engine.read_csv(file_path)
        processed_df = spark_engine.preprocess_data(df, feature_cols, label_col)
        spark_engine.write_parquet(processed_df, output_path)

        return jsonify({"message": "Data processed successfully.", "output_path": output_path}), 200
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

        if not file_path:
            return jsonify({"error": "Missing required parameter 'file_path'"}), 400

        # Perform clustering
        df = spark_engine.read_csv(file_path)
        df_preprocessed = spark_engine.preprocess_data(df, [], "")
        clustered_df = spark_engine.cluster_data(df_preprocessed, num_clusters)
        spark_engine.write_parquet(clustered_df, output_path)

        return jsonify({"message": "Data clustered successfully.", "output_path": output_path}), 200
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

        if not file_path or not label_col:
            return jsonify({"error": "Missing required parameters 'file_path' and/or 'label_col'"}), 400

        # Perform model training
        df = spark_engine.read_csv(file_path)
        df_preprocessed = spark_engine.preprocess_data(df, [], label_col)
        model = spark_engine.train_model(df_preprocessed, label_col)
        model.save(model_path)

        return jsonify({"message": "Model trained successfully.", "model_path": model_path}), 200
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

        if not model_path:
            return jsonify({"error": "Missing required parameter 'model_path'"}), 400
        if not file_path:
            return jsonify({"error": "Missing required parameter 'file_path'"}), 400

        # Perform predictions
        df = spark_engine.read_csv(file_path)
        predictions = spark_engine.predict(df, model_path)

        return jsonify({"predictions": predictions}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
