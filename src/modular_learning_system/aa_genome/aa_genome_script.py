import logging
import os
from typing import List

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame

from modular_learning_system.aa_genome.aa_genome import AA_Genome
from modular_learning_system.neurotech_network.neurotech_network_script import NeurotechNetwork
from modular_learning_system.spark_engine.spark_engine import SparkEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AAGenomeScript:
    def __init__(self):
        self.spark_engine = SparkEngine()
        self.neurotech_network = NeurotechNetwork()

    def process_data(self, df: SparkDataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Process the Spark DataFrame, clean the data, and convert it to NumPy.
        :param df: Input Spark DataFrame
        :param feature_cols: List of features to be processed
        :return: NumPy array of processed data
        """
        # Validate feature columns
        if not all(col in df.columns for col in feature_cols):
            raise KeyError(f"One or more feature columns {feature_cols} are not present in the DataFrame.")

        # Preprocess data
        logger.info("Preprocessing data using SparkEngine.")
        cleaned_data = self.spark_engine.preprocess_data(df, feature_cols)

        # Convert Spark DataFrame to List[List[float]], then to NumPy array
        cleaned_data_list = cleaned_data.select(*feature_cols).rdd.map(lambda row: list(row)).collect()
        processed_data = AA_Genome._convert_to_numpy(cleaned_data_list, feature_cols)

        logger.info("Data successfully processed into NumPy array.")
        return processed_data

    def run_pipeline(self, df: SparkDataFrame, feature_cols: List[str]):
        """
        Run the full data processing and analysis pipeline.
        :param df: Input Spark DataFrame
        :param feature_cols: List of feature columns
        :return: Results of the NeuroTechNetwork analysis
        """
        try:
            logger.info("Running data processing pipeline.")
            processed_data = self.process_data(df, feature_cols)
            results = self.neurotech_network.analyze_data(processed_data)
            logger.info("Pipeline executed successfully.")
            return results
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    :param file_path: Path to the CSV file
    :return: Pandas DataFrame with the loaded data
    """
    if os.path.exists(file_path):
        logger.info(f"Loading data from {file_path}.")
        return pd.read_csv(file_path)
    else:
        logger.error(f"The file {file_path} does not exist.")
        raise FileNotFoundError(f"The file {file_path} does not exist.")


def load_and_preprocess_data(paths: List[str]) -> SparkDataFrame:
    """
    Load and preprocess data from various CSV files.
    :param paths: List of file paths for the datasets
    :return: Combined Spark DataFrame
    """
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame

    for path in paths:
        try:
            df = load_data_from_csv(path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except FileNotFoundError as e:
            logger.warning(f"Error loading file: {e}")

    if combined_df.empty:
        logger.error("No data was loaded from the specified paths.")
        raise ValueError("No valid data could be loaded from the provided paths.")

    # Convert the combined Pandas DataFrame to a Spark DataFrame
    logger.info("Converting combined Pandas DataFrame to Spark DataFrame.")
    spark = SparkSession.builder.appName(f"AA_Genome_Session_{os.getpid()}").getOrCreate()
    spark_df = spark.createDataFrame(combined_df)
    return spark_df


def main():
    """
    Main function to run the data processing and analysis pipeline.
    """
    # Configurable paths and columns (update as per your project)
    paths = ['data/file1.csv', 'data/file2.csv']  # Example paths (update as needed)
    feature_cols = ['feature1', 'feature2', 'feature3']  # Replace with actual feature columns in your dataset

    try:
        # Step 1: Load and preprocess data
        logger.info("Starting the data loading and preprocessing step.")
        df_spark = load_and_preprocess_data(paths)

        # Step 2: Process data using AAGenomeScript
        logger.info("Initializing AAGenomeScript for further processing.")
        aa_genome_script = AAGenomeScript()

        # Step 3: Run the pipeline
        results = aa_genome_script.run_pipeline(df_spark, feature_cols)
        logger.info("Processing and analysis pipeline completed successfully.")

        # Display or save results
        print("Pipeline Results:")
        print(results)

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
