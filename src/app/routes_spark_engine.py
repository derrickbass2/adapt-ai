from flask import Blueprint, request, jsonify
from pyspark.ml.classification import RandomForestClassifier

from modular_learning_system import SparkEngine

# Initialize the Blueprint
spark_engine_bp = Blueprint('spark_engine', __name__)

# Initialize the SparkEngine instance
spark_engine = SparkEngine()


@spark_engine_bp.route('/api/spark-engine/preprocess', methods=['POST'])
def preprocess_data():
    """
    Endpoint to preprocess data using SparkEngine.
    Expects JSON input with file_path, feature_cols, and label_col.
    """
    data = request.json
    file_path = data.get('file_path')
    feature_cols = data.get('feature_cols', [])
    label_col = data.get('label_col', None)

    df = spark_engine.read_csv(file_path)
    processed_df = spark_engine.preprocess_data(df, feature_cols, label_col)

    # Save or return the processed DataFrame as needed
    output_path = data.get('output_path', '/tmp/processed_data.parquet')
    spark_engine.write_parquet(processed_df, output_path)

    return jsonify({"message": "Data processed and saved.", "output_path": output_path}), 200


@spark_engine_bp.route('/api/spark-engine/cluster', methods=['POST'])
def cluster_data():
    """
    Endpoint to cluster data using SparkEngine.
    Expects JSON input with file_path and num_clusters.
    """
    data = request.json
    file_path = data.get('file_path')
    num_clusters = data.get('num_clusters', 3)

    df = spark_engine.read_csv(file_path)
    df_preprocessed = spark_engine.preprocess_data(df, [], '')  # Provide appropriate feature columns
    clustered_df = spark_engine.cluster_data(df_preprocessed, num_clusters)

    # Save clustered data to the specified output path
    output_path = data.get('output_path', '/tmp/clustered_data.parquet')
    spark_engine.write_parquet(clustered_df, output_path)

    return jsonify({"message": "Data clustered and saved.", "output_path": output_path}), 200


@spark_engine_bp.route('/api/spark-engine/train', methods=['POST'])
def train_model():
    """
    Endpoint to train a model using SparkEngine.
    Expects JSON input with file_path and label_col.
    """
    data = request.json
    file_path = data.get('file_path')
    label_col = data.get('label_col', '')

    df = spark_engine.read_csv(file_path)
    df_preprocessed = spark_engine.preprocess_data(df, [], label_col)
    model = spark_engine.train_model(df_preprocessed, label_col)

    # Save the trained model to the specified path
    model_path = data.get('model_path', '/tmp/model')
    model.save(model_path)

    return jsonify({"message": "Model trained and saved.", "model_path": model_path}), 200


@spark_engine_bp.route('/api/spark-engine/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions using a trained model in SparkEngine.
    Expects JSON input with file_path and model_path.
    """
    data = request.json
    file_path = data.get('file_path')
    model_path = data.get('model_path')

    df = spark_engine.read_csv(file_path)
    model = RandomForestClassifier.load(model_path)
    predictions_df = spark_engine.predict(model, df)

    # Save the predictions to the specified output path
    output_path = data.get('output_path', '/tmp/predictions.parquet')
    spark_engine.write_parquet(predictions_df, output_path)

    return jsonify({"message": "Predictions made and saved.", "output_path": output_path}), 200


# Additional routes for Spark Engine can be added here


@spark_engine_bp.route('/')
def home():
    return jsonify({"message": "Welcome to the Spark Engine API"})
