from flask import Blueprint, request, jsonify
from modular_learning_system.spark_engine.setup import SparkEngine

spark_engine_bp = Blueprint('spark_engine', __name__)
spark_engine = SparkEngine()


@spark_engine_bp.route('/api/spark-engine/preprocess', methods=['POST'])
def preprocess_data():
    data = request.json
    file_path = data.get('file_path')
    feature_cols = data.get('feature_cols', [])
    label_col = data.get('label_col', None)

    df = spark_engine.read_csv(file_path)
    processed_df = spark_engine.preprocess_data(df, feature_cols, label_col)
    # Save or return the processed DataFrame as needed
    output_path = data.get('output_path', '/tmp/processed_data.parquet')
    spark_engine.write_parquet(processed_df, output_path)

    return jsonify({"message": "Data processed and saved.", "output_path": output_path})


@spark_engine_bp.route('/api/spark-engine/cluster', methods=['POST'])
def cluster_data():
    data = request.json
    file_path = data.get('file_path')
    num_clusters = data.get('num_clusters', 3)

    df = spark_engine.read_csv(file_path)
    df_preprocessed = spark_engine.preprocess_data(df, [], '')  # Provide appropriate feature columns
    clustered_df = spark_engine.cluster_data(df_preprocessed, num_clusters)
    output_path = data.get('output_path', '/tmp/clustered_data.parquet')
    spark_engine.write_parquet(clustered_df, output_path)

    return jsonify({"message": "Data clustered and saved.", "output_path": output_path})


@spark_engine_bp.route('/api/spark-engine/train', methods=['POST'])
def train_model():
    data = request.json
    file_path = data.get('file_path')
    label_col = data.get('label_col', '')

    df = spark_engine.read_csv(file_path)
    df_preprocessed = spark_engine.preprocess_data(df, [], label_col)
    model = spark_engine.train_model(df_preprocessed, label_col)
    # Save or return the model as needed
    model_path = data.get('model_path', '/tmp/model')
    model.save(model_path)

    return jsonify({"message": "Model trained and saved.", "model_path": model_path})


@spark_engine_bp.route('/api/spark-engine/predict', methods=['POST'])
def predict():
    data = request.json
    file_path = data.get('file_path')
    model_path = data.get('model_path')

    df = spark_engine.read_csv(file_path)
    model = RandomForestClassifier.load(model_path)
    predictions_df = spark_engine.predict(model, df)
    output_path = data.get('output_path', '/tmp/predictions.parquet')
    spark_engine.write_parquet(predictions_df, output_path)

    return jsonify({"message": "Predictions made and saved.", "output_path": output_path})