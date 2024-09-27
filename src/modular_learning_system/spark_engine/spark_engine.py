import logging
import os

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType

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

    def get_spark_session(self) -> SparkSession:
        """
        Returns the Spark session.
        """
        if not self.spark_session:
            raise Exception("SparkSession has been closed")
        return self.spark_session

    def close_spark_session(self):
        """
        Closes the Spark session.
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
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
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
        try:
            # Drop rows with missing values for the features or label
            preprocessed_data = input_data.dropna(subset=feature_columns + ([label_column] if label_column else []))
            logger.info(f"Dataframe reduced to {preprocessed_data.count()} rows after preprocessing.")
            return preprocessed_data
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise

    def __del__(self):
        """
        Destructor to ensure Spark session is stopped if it is still running.
        """
        try:
            self.close_spark_session()
        except Exception as e:
            logger.error(f"Error during SparkEngine cleanup: {e}")

    def read_csv(self, file_path):
        pass

    def cluster_data(self, df_preprocessed, num_clusters):
        pass

    def train_model(self, df_preprocessed, label_col):
        pass

    def predict(self, df, model_path):
        pass


class SparkEngineUtils:
    def __init__(self, spark_session_obj: SparkSession):
        """
        Initialize SparkEngineUtils with a Spark session.

        :param spark_session_obj: SparkSession object.
        """
        self.spark_session_obj = spark_session_obj

    @staticmethod
    def advanced_cleaning(df_to_clean: DataFrame) -> DataFrame:
        """
        Perform advanced cleaning and transformations like handling missing values,
        outlier removal, or complex filtering.

        :param df_to_clean: DataFrame to be cleaned.
        :return: Cleaned DataFrame.
        """
        logger.info("Performing advanced cleaning...")
        try:
            # Example of filtering out rows with negative values in all numeric columns
            numeric_cols = [f.name for f in df_to_clean.schema.fields if
                            isinstance(f.dataType, (IntegerType, DoubleType))]
            cleaned_df = df_to_clean.filter(col(numeric_cols[0]) >= 0)  # Replace with actual condition
            return cleaned_df
        except Exception as e:
            logger.error(f"Error during advanced cleaning: {e}")
            raise

    @staticmethod
    def transform_data(df_to_transform: DataFrame, feature_columns_list: list,
                       label_column_name: str = None) -> DataFrame:
        """
        Assemble feature vectors for modeling or clustering.

        :param df_to_transform: DataFrame to be transformed.
        :param feature_columns_list: List of feature column names.
        :param label_column_name: Optional label column name.
        :return: Transformed DataFrame with features column.
        """
        logger.info("Transforming data...")
        try:
            assembler = VectorAssembler(inputCols=feature_columns_list, outputCol="features")
            transformed_df = assembler.transform(df_to_transform)

            if label_column_name:
                transformed_df = transformed_df.select("features", label_column_name)
            else:
                transformed_df = transformed_df.select("features")

            return transformed_df
        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise

    def run_kmeans(self, input_df: DataFrame, feature_columns_list: list, k: int = 3) -> DataFrame:
        """
        Perform KMeans clustering on the given DataFrame.

        :param input_df: DataFrame to be clustered.
        :param feature_columns_list: List of feature column names.
        :param k: Number of clusters.
        :return: DataFrame with clustering results.
        """
        logger.info("Running KMeans clustering...")
        try:
            transformed_df = self.transform_data(input_df, feature_columns_list)
            kmeans = KMeans(k=k)
            model = kmeans.fit(transformed_df)
            predictions = model.transform(transformed_df)
            return predictions.select('features', 'prediction')
        except Exception as e:
            logger.error(f"Error during KMeans clustering: {e}")
            raise

    def run_linear_regression(self, input_df: DataFrame, feature_columns_list: list,
                              label_column_name: str) -> DataFrame:
        """
        Perform Linear Regression on the given DataFrame.

        :param input_df: DataFrame to be used for linear regression.
        :param feature_columns_list: List of feature column names.
        :param label_column_name: Name of the label column.
        :return: DataFrame with regression predictions.
        """
        logger.info("Running Linear Regression...")
        try:
            transformed_df = self.transform_data(input_df, feature_columns_list, label_column_name)
            lr = LinearRegression(labelCol=label_column_name)
            lr_model = lr.fit(transformed_df)
            predictions = lr_model.transform(transformed_df)
            return predictions.select("features", "prediction", label_column_name)
        except Exception as e:
            logger.error(f"Error during Linear Regression: {e}")
            raise

    @staticmethod
    def write_output(df_to_write: DataFrame, output_path_dir: str):
        """
        Write DataFrame to the specified output path in Parquet format.

        :param df_to_write: DataFrame to be written.
        :param output_path_dir: Directory path where the output will be saved.
        """
        logger.info(f"Saving data to {output_path_dir}...")
        try:
            if not os.path.exists(output_path_dir):
                os.makedirs(output_path_dir)
            df_to_write.write.parquet(output_path_dir)
            logger.info(f"Data saved to {output_path_dir}.")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise

    def convert_to_pandas(self, processed_df):
        pass

    def read_csv(self, file_path):
        pass


# Example usage:
if __name__ == "__main__":
    engine = SparkEngine()
    spark_session = engine.get_spark_session()

    # Read data from CSV file
    data_path = "/Users/derrickbass/Public/adaptai/datasets/finance/ed_stats_country.csv"
    df = engine.read_csv_file(data_path)

    # Preprocess data
    feature_columns = ["Country Code", "Short Name", "Table Name", "Long Name", "2-alpha code", "Currency Unit",
                       "Special Notes", "Region", "Income Group", "WB-2 code", "National accounts base year",
                       "Other groups", "System of National Accounts", "Alternative conversion factor",
                       "PPP survey year", "Balance of Payments Manual in use", "External debt Reporting status",
                       "System of trade", "Government Accounting concept", "IMF data dissemination standard",
                       "Latest population census", "Latest household survey",
                       "Source of most recent Income and expenditure data", "Vital registration complete",
                       "Latest agricultural census", "Latest industrial data", "Latest trade data",
                       "Latest water withdrawal data"]
    label_column = "Country Code"
    preprocessed_df = engine.preprocess_data(df, feature_columns, label_column)

    # Perform KMeans clustering
    kmeans_df = SparkEngineUtils(spark_session).run_kmeans(preprocessed_df, feature_columns)

    # Write output to Parquet file
    output_path_dir = "/Users/derrickbass/Public/adaptai/output"
    SparkEngineUtils.write_output(kmeans_df, output_path_dir)

    # Close Spark session
    engine.close_spark_session()
