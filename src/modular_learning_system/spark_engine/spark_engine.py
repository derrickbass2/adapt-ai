import logging
import os

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkEngine:
    def __init__(self, spark: SparkSession = None):
        """
        Initialize SparkEngine with a Spark session.
        If no Spark session is provided, a new one is created.
        """
        self.spark = spark or SparkSession.builder.appName('ADAPTAI').getOrCreate()

    def get_spark_session(self) -> SparkSession:
        """
        Returns the Spark session.
        """
        if not self.spark:
            raise Exception("SparkSession has been closed")
        return self.spark

    def close_spark_session(self):
        """
        Closes the Spark session.
        """
        if self.spark:
            self.spark.stop()
            self.spark = None
        else:
            logger.warning("SparkSession is already closed")

    def read_csv(self, path: str) -> DataFrame:
        """
        Read a CSV file into a DataFrame.

        :param path: Path to the CSV file
        :return: DataFrame
        """
        return self.spark.read.option('header', 'true').csv(path)

    def write_parquet(self, df: DataFrame, path: str):
        """
        Write DataFrame to Parquet format.

        :param df: DataFrame to be written
        :param path: Path where the Parquet file will be saved
        """
        df.write.parquet(path)

    def preprocess_data(self, df, feature_cols):
        """
        Placeholder for data preprocessing method.
        """
        pass

    def __del__(self):
        """
        Destructor to ensure Spark session is stopped if it is still running.
        """
        try:
            self.close_spark_session()
        except Exception as e:
            logger.error(f"Error during SparkEngine cleanup: {e}")


class SparkEngineUtils:
    def __init__(self, spark: SparkSession):
        """
        Initialize SparkEngineUtils with a Spark session.

        :param spark: SparkSession object
        """
        self.spark = spark

    def advanced_cleaning(self, df: DataFrame) -> DataFrame:
        """
        Perform advanced cleaning and transformations, such as missing value handling,
        outlier removal, or complex filtering.

        :param df: DataFrame to be cleaned
        :return: Cleaned DataFrame
        """
        logger.info("Performing advanced cleaning...")
        try:
            cleaned_df = df.dropna()  # Example of dropping rows with missing values
            # Add more transformations as needed
            return cleaned_df
        except Exception as e:
            logger.error(f"Error during advanced cleaning: {e}")
            raise

    def transform_data(self, df: DataFrame, feature_cols: list, label_col: str = None) -> DataFrame:
        """
        Assemble feature vectors for modeling or clustering.

        :param df: DataFrame to be transformed
        :param feature_cols: List of feature column names
        :param label_col: Optional label column name
        :return: Transformed DataFrame with features column
        """
        logger.info("Transforming data...")
        try:
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            transformed_df = assembler.transform(df)

            if label_col:
                transformed_df = transformed_df.select("features", label_col)
            else:
                transformed_df = transformed_df.select("features")

            return transformed_df
        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise

    def run_kmeans(self, df: DataFrame, feature_cols: list, k: int = 3) -> DataFrame:
        """
        Perform KMeans clustering on the given DataFrame.

        :param df: DataFrame to be clustered
        :param feature_cols: List of feature column names
        :param k: Number of clusters
        :return: DataFrame with clustering results
        """
        logger.info("Running KMeans clustering...")
        try:
            transformed_df = self.transform_data(df, feature_cols)
            kmeans = KMeans(k=k)
            model = kmeans.fit(transformed_df)
            predictions = model.transform(transformed_df)
            return predictions.select('features', 'prediction')
        except Exception as e:
            logger.error(f"Error during KMeans clustering: {e}")
            raise

    def run_linear_regression(self, df: DataFrame, feature_cols: list, label_col: str) -> DataFrame:
        """
        Perform Linear Regression on the given DataFrame.

        :param df: DataFrame to be used for linear regression
        :param feature_cols: List of feature column names
        :param label_col: Name of the label column
        :return: DataFrame with regression predictions
        """
        logger.info("Running Linear Regression...")
        try:
            transformed_df = self.transform_data(df, feature_cols, label_col)
            lr = LinearRegression(featuresCol='features', labelCol=label_col)
            lr_model = lr.fit(transformed_df)
            predictions = lr_model.transform(transformed_df)
            return predictions.select("features", "prediction", label_col)
        except Exception as e:
            logger.error(f"Error during Linear Regression: {e}")
            raise

    def write_output(self, df: DataFrame, output_path: str):
        """
        Write DataFrame to the specified output path in Parquet format.

        :param df: DataFrame to be written
        :param output_path: Directory path where the output will be saved
        """
        logger.info(f"Saving data to {output_path}...")
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df.write.parquet(output_path)
            logger.info(f"Data saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise

    def read_csv(self, file_path):
        """
        Placeholder for CSV reading logic.
        """
        pass
