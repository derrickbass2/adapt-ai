import logging
import os
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

# Configure logging centrally
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SparkEngine:
    def __init__(self, spark_session: SparkSession = None, app_name: str = 'ADAPTAI'):
        """
        Initialize SparkEngine with a Spark session.
        If no Spark session is provided, a new one is created.

        :param spark_session: Optional SparkSession object. If None, a new session will be created.
        :param app_name: Application name for the Spark session.
        """
        self.spark_session = spark_session or SparkSession.builder.appName(app_name).getOrCreate()
        logger.info(f"Spark session started with app name '{app_name}'.")

    def get_spark_session(self) -> SparkSession:
        """
        Returns the Spark session.
        """
        if not self.spark_session:
            raise Exception("SparkSession has been closed.")
        return self.spark_session

    def close_spark_session(self):
        """
        Closes the Spark session and cleans up resources.
        """
        if self.spark_session:
            self.spark_session.stop()
            self.spark_session = None
            logger.info("SparkSession closed successfully.")
        else:
            logger.warning("SparkSession is already closed.")

    def read_csv_file(self, file_path: str) -> DataFrame:
        """
        Read a CSV file into a DataFrame.

        :param file_path: Path to the CSV file.
        :return: DataFrame containing the CSV data.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        logger.info(f"Reading CSV file from {file_path}...")
        return self.spark_session.read.option('header', 'true').csv(file_path)

    @staticmethod
    def write_parquet(df: DataFrame, output_path: str):
        """
        Write DataFrame to Parquet format.

        :param df: DataFrame to be written.
        :param output_path: Path where the Parquet file will be saved.
        """
        output_dir = Path(output_path).parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True)
            except OSError as e:
                logger.error(f"Error creating output directory {output_dir}: {e}")
                raise
        logger.info(f"Writing DataFrame to Parquet at {output_path}...")
        df.write.parquet(output_path)
        logger.info(f"Data successfully written to {output_path}.")

    @staticmethod
    def preprocess_data(input_data: DataFrame, feature_columns: list, label_column: str = None) -> DataFrame:
        """
        Perform basic data preprocessing like feature selection and handling missing values.

        :param input_data: Input DataFrame to preprocess.
        :param feature_columns: List of feature column names.
        :param label_column: Optional label column name.
        :return: Preprocessed DataFrame.
        """
        logger.info("Preprocessing data...")

        # Validate input types
        if not isinstance(input_data, DataFrame):
            raise TypeError("input_data must be a PySpark DataFrame.")
        if not isinstance(feature_columns, list) or not feature_columns:
            raise ValueError("feature_columns must be a non-empty list of column names.")

        try:
            # Ensure all feature columns exist in the DataFrame
            missing_columns = [col for col in feature_columns if col not in input_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")

            # Check if label column is present, if provided
            if label_column and label_column not in input_data.columns:
                raise ValueError(f"Label column '{label_column}' does not exist in the DataFrame.")

            # Count rows before preprocessing (optimization: retrieve once)
            original_count = input_data.count()

            # Drop rows with missing values in feature and label columns
            columns_to_check = feature_columns + ([label_column] if label_column else [])
            preprocessed_data = input_data.dropna(subset=columns_to_check)

            # Count rows after preprocessing
            reduced_count = preprocessed_data.count()
            logger.info(f"Data reduced from {original_count} to {reduced_count} rows after preprocessing.")

            return preprocessed_data

        except ValueError as ve:
            logger.error(f"ValueError in preprocessing: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data preprocessing: {e}")
            raise

    def __del__(self):
        """
        Clean up Spark session on object destruction
        """
        try:
            self.close_spark_session()
        except Exception as e:
            logger.error(f"Error during SparkEngine cleanup: {e}")
