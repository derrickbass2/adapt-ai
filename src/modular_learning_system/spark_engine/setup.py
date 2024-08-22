import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml import Pipeline
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkEngine:
    def __init__(self, app_name: str = "adaptai", shuffle_partitions: int = 8):
        """
        Initialize SparkSession with customizable configurations.

        Args:
            app_name (str): Name of the Spark application.
            shuffle_partitions (int): Number of shuffle partitions for Spark.
        """
        try:
            self.spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
                .getOrCreate()
            logger.info(
                f"SparkSession initialized with app name '{app_name}' and shuffle partitions '{shuffle_partitions}'.")
        except Exception as e:
            logger.error(f"Error initializing SparkSession: {e}")
            raise

    def read_csv(self, path: str) -> Optional[DataFrame]:
        """
        Read a CSV file into a Spark DataFrame with error handling.

        Args:
            path (str): The file path to the CSV file.

        Returns:
            Optional[DataFrame]: The loaded Spark DataFrame, or None if an error occurred.
        """
        try:
            df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(path)
            logger.info(f"CSV file read successfully from path: {path}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file at {path}: {e}")
            return None

    def write_parquet(self, df: DataFrame, path: str) -> None:
        """
        Write a Spark DataFrame to a Parquet file.

        Args:
            df (DataFrame): The Spark DataFrame to write.
            path (str): The file path to write the Parquet file.
        """
        try:
            df.write.mode("overwrite").parquet(path)
            logger.info(f"DataFrame written to Parquet at path: {path}")
        except Exception as e:
            logger.error(f"Error writing DataFrame to Parquet at {path}: {e}")
            raise

    def preprocess_data(self, df: DataFrame, feature_cols: List[str], label_col: str) -> Optional[DataFrame]:
        """
        Preprocess the DataFrame by assembling features and scaling.

        Args:
            df (DataFrame): The DataFrame to preprocess.
            feature_cols (List[str]): List of feature column names.
            label_col (str): The name of the label column.

        Returns:
            Optional[DataFrame]: The preprocessed DataFrame, or None if an error occurred.
        """
        try:
            vec_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
            pipeline = Pipeline(stages=[vec_assembler, scaler])
            model = pipeline.fit(df)
            df_preprocessed = model.transform(df)
            logger.info("Data preprocessing completed successfully.")
            return df_preprocessed
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            return None

    def cluster_data(self, df: DataFrame, num_clusters: int) -> Optional[DataFrame]:
        """
        Perform k-means clustering on the DataFrame.

        Args:
            df (DataFrame): The DataFrame to cluster.
            num_clusters (int): Number of clusters for k-means.

        Returns:
            Optional[DataFrame]: DataFrame with cluster assignments, or None if an error occurred.
        """
        try:
            kmeans = KMeans(k=num_clusters, seed=1, featuresCol="scaled_features", predictionCol="cluster")
            model = kmeans.fit(df)
            df_clustered = model.transform(df)
            logger.info(f"Clustering completed with {num_clusters} clusters.")
            return df_clustered
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            return None

    def train_model(self, df: DataFrame, label_col: str) -> Optional[RandomForestClassificationModel]:
        """
        Train a random forest classifier.

        Args:
            df (DataFrame): The DataFrame with features and labels.
            label_col (str): The name of the label column.

        Returns:
            Optional[RandomForestClassificationModel]: Trained RandomForestClassificationModel model, or None if an error occurred.
        """
        try:
            rf = RandomForestClassifier(featuresCol="scaled_features", labelCol=label_col)
            model = rf.fit(df)
            logger.info("Random forest model training completed.")
            return model
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return None

    def predict(self, model: RandomForestClassificationModel, df: DataFrame) -> Optional[DataFrame]:
        """
        Predict labels using the trained model.

        Args:
            model (RandomForestClassificationModel): The trained model.
            df (DataFrame): The DataFrame with features.

        Returns:
            Optional[DataFrame]: DataFrame with predictions, or None if an error occurred.
        """
        try:
            predictions = model.transform(df)
            logger.info("Prediction completed successfully.")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None

    def stop(self) -> None:
        """Stop the SparkSession."""
        try:
            if self.spark:
                self.spark.stop()
                logger.info("SparkSession stopped.")
        except Exception as e:
            logger.error(f"Error stopping SparkSession: {e}")