from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType


class SparkEngine:
    def __init__(self):
        """Initialize SparkSession."""
        self.spark = SparkSession.builder \
            .appName("adaptai") \
            .config("spark.sql.shuffle.partitions", "8") \
            .getOrCreate()

    def read_csv(self, path: str) -> DataFrame:
        """
        Read a CSV file into a Spark DataFrame.

        Args:
            path (str): The file path to the CSV file.

        Returns:
            DataFrame: The loaded Spark DataFrame.
        """
        return self.spark.read.option("header", "true").option("inferSchema", "true").csv(path)

    def write_parquet(self, df: DataFrame, path: str) -> None:
        """
        Write a Spark DataFrame to a Parquet file.

        Args:
            df (DataFrame): The Spark DataFrame to write.
            path (str): The file path to write the Parquet file.
        """
        df.write.mode("overwrite").parquet(path)

    def preprocess_data(self, df: DataFrame, feature_cols: list, label_col: str) -> DataFrame:
        """
        Preprocess the DataFrame by assembling features and scaling.

        Args:
            df (DataFrame): The DataFrame to preprocess.
            feature_cols (list): List of feature column names.
            label_col (str): The name of the label column.

        Returns:
            DataFrame: The preprocessed DataFrame.
        """
        vec_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        pipeline = Pipeline(stages=[vec_assembler, scaler])
        model = pipeline.fit(df)
        df_preprocessed = model.transform(df)
        return df_preprocessed

    def cluster_data(self, df: DataFrame, num_clusters: int) -> DataFrame:
        """
        Perform k-means clustering on the DataFrame.

        Args:
            df (DataFrame): The DataFrame to cluster.
            num_clusters (int): Number of clusters for k-means.

        Returns:
            DataFrame: DataFrame with cluster assignments.
        """
        kmeans = KMeans(k=num_clusters, seed=1, featuresCol="scaled_features", predictionCol="cluster")
        model = kmeans.fit(df)
        df_clustered = model.transform(df)
        return df_clustered

    def train_model(self, df: DataFrame, label_col: str) -> RandomForestClassifier:
        """
        Train a random forest classifier.

        Args:
            df (DataFrame): The DataFrame with features and labels.
            label_col (str): The name of the label column.

        Returns:
            RandomForestClassifier: Trained RandomForestClassifier model.
        """
        rf = RandomForestClassifier(featuresCol="scaled_features", labelCol=label_col)
        model = rf.fit(df)
        return model

    def predict(self, model: RandomForestClassifier, df: DataFrame) -> DataFrame:
        """
        Predict labels using the trained model.

        Args:
            model (RandomForestClassifier): The trained model.
            df (DataFrame): The DataFrame with features.

        Returns:
            DataFrame: DataFrame with predictions.
        """
        return model.transform(df)

    def __del__(self):
        """Stop the SparkSession."""
        if self.spark:
            self.spark.stop()