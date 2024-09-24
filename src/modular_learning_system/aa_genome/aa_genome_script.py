import os
from typing import List

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame

from adapt_backend.ml_models import AA_Genome  # Adjust the import to match the module structure
from modular_learning_system.neurotech_network.neurotech_network_script import NeurotechNetwork
from modular_learning_system.spark_engine.spark_engine import SparkEngine


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
        cleaned_data = self.spark_engine.preprocess_data(df, feature_cols)
        aa_genome = AA_Genome(df=cleaned_data)
        processed_data = aa_genome._convert_to_numpy(cleaned_data, feature_cols)
        return processed_data

    def run_pipeline(self, df: SparkDataFrame, feature_cols: List[str]):
        """
        Run the full data processing and analysis pipeline.

        :param df: Input Spark DataFrame
        :param feature_cols: List of feature columns
        :return: Results of the NeuroTechNetwork analysis
        """
        processed_data = self.process_data(df, feature_cols)
        results = self.neurotech_network.analyze_data(processed_data)
        return results


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    :param file_path: Path to the CSV file
    :return: Pandas DataFrame with the loaded data
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")


def load_and_preprocess_data(paths: List[str]) -> SparkDataFrame:
    """
    Load and preprocess data from various CSV files.

    :param paths: List of file paths for the datasets
    :return: Preprocessed Spark DataFrame
    """
    combined_df = pd.DataFrame()

    for path in paths:
        try:
            df = load_data_from_csv(path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except FileNotFoundError as e:
            print(f"Error: {e}")

    # Convert the combined Pandas DataFrame to a Spark DataFrame
    spark = SparkSession.builder.appName('AA_Genome_Session').getOrCreate()
    spark_df = spark.createDataFrame(combined_df)
    return spark_df


def main():
    # File paths to your CSV datasets (update with actual paths)
    paths = ['data/file1.csv', 'data/file2.csv']  # Example paths

    # Load and preprocess data
    df_spark = load_and_preprocess_data(paths)

    # Define feature columns based on your data
    feature_cols = ['feature1', 'feature2', 'feature3']  # Update with actual feature columns

    # Process data using AAGenomeScript
    aa_genome_script = AAGenomeScript()
    processed_data = aa_genome_script.process_data(df_spark, feature_cols)

    # Analyze processed data using NeuroTechNetwork
    results = aa_genome_script.run_pipeline(df_spark, feature_cols)
    print(results)


if __name__ == '__main__':
    main()
