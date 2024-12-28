import os
from uuid import uuid4

from flask import Blueprint, request, jsonify
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression as SparkLinearRegression
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder.appName("SparkEngine").getOrCreate()

# Flask Blueprint for Spark operations
spark_engine_bp = Blueprint("spark_engine", __name__)


class SparkEngine:
    @staticmethod
    def read_csv(file_path):
        """
        Reads a CSV file into a Spark DataFrame.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return spark.read.csv(file_path, header=True, inferSchema=True)

    @staticmethod
    def preprocess_data(df: DataFrame, feature_cols, label_col=None):
        """
        Preprocesses the input Spark DataFrame for clustering or training.
        """
        if not all(col in df.columns for col in feature_cols):
            raise ValueError("Some feature columns are missing in the DataFrame.")
        if label_col and label_col not in df.columns:
            raise ValueError("The label column is missing in the DataFrame.")

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        if label_col:
            return assembler.transform(df).select("features", label_col)
        return assembler.transform(df).select("features")

    @staticmethod
    def cluster_data(df: DataFrame, num_clusters: int):
        """
        Perform clustering using Spark MLlib's KMeans.
        """
        kmeans = SparkKMeans(k=num_clusters, seed=42, featuresCol="features")
        model = kmeans.fit(df)
        return model.transform(df)

    @staticmethod
    def train_model(df: DataFrame, label_col: str, is_classification=True):
        """
        Train a Spark-based Machine Learning model (Logistic Regression or Linear Regression).
        """
        if is_classification:
            model = SparkLogisticRegression(featuresCol="features", labelCol=label_col, maxIter=10)
        else:
            model = SparkLinearRegression(featuresCol="features", labelCol=label_col)

        pipeline = Pipeline(stages=[model])
        trained_model = pipeline.fit(df)
        return trained_model

    @staticmethod
    def save_model(model, model_path):
        """
        Save a given Spark MLlib model.
        """
        model.write().overwrite().save(model_path)


@spark_engine_bp.route("/preprocess", methods=["POST"])
def preprocess_data():
    """
    Flask endpoint for preprocessing data.
    """
    data = request.json
    try:
        file_path = data["file_path"]
        feature_cols = data["feature_cols"]
        label_col = data.get("label_col", None)

        df = SparkEngine.read_csv(file_path)
        processed_df = SparkEngine.preprocess_data(df, feature_cols, label_col)
        output_path = f"/tmp/processed_data-{uuid4().hex}.parquet"
        processed_df.write.mode("overwrite").parquet(output_path)

        return jsonify({"message": "Data preprocessed successfully", "output_path": output_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@spark_engine_bp.route("/cluster", methods=["POST"])
def cluster_data():
    """
    Flask endpoint for clustering.
    """
    data = request.json
    try:
        file_path = data["file_path"]
        num_clusters = data["num_clusters"]

        df = SparkEngine.read_csv(file_path)
        processed_df = SparkEngine.preprocess_data(df, feature_cols=df.columns)
        clustered_df = SparkEngine.cluster_data(processed_df, num_clusters)
        output_path = f"/tmp/clustered_data-{uuid4().hex}.parquet"
        clustered_df.write.mode("overwrite").parquet(output_path)

        return jsonify({"message": "Data clustered successfully", "output_path": output_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@spark_engine_bp.route("/train", methods=["POST"])
def train_model():
    """
    Flask endpoint for training a machine learning model.
    """
    data = request.json
    try:
        file_path = data["file_path"]
        label_col = data["label_col"]
        is_classification = data.get("is_classification", True)

        df = SparkEngine.read_csv(file_path)
        processed_df = SparkEngine.preprocess_data(df, feature_cols=df.columns, label_col=label_col)
        model = SparkEngine.train_model(processed_df, label_col, is_classification)

        model_path = f"/tmp/model-{uuid4().hex}"
        SparkEngine.save_model(model, model_path)

        return jsonify({"message": "Model trained successfully", "model_path": model_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
