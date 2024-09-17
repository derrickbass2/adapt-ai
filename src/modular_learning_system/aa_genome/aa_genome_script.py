import os

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from adapt_backend.ml_models import AA_Genome  # Adjust the import to match the module structure


def load_data_from_csv(file_path):
    """
    Load data from a CSV file.

    :param file_path: Path to the CSV file
    :return: DataFrame with the loaded data
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")


def load_and_preprocess_data():
    """
    Load and preprocess data from various datasets.

    :return: Preprocessed Spark DataFrame
    """
    # Define paths to datasets
    paths = [
        # List of file paths...
    ]

    # Initialize an empty DataFrame to combine all datasets
    combined_df = pd.DataFrame()

    # Load and combine data
    for path in paths:
        try:
            df = load_data_from_csv(path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except FileNotFoundError as e:
            print(e)

    # Convert combined DataFrame to Spark DataFrame
    spark = SparkSession.builder.appName('AA_Genome_Session').getOrCreate()
    spark_df = spark.createDataFrame(combined_df)
    return spark_df


def main():
    X_dimension = 10
    df = load_and_preprocess_data()

    # Here, assuming that you have 'x' and 'y' columns in your combined data
    x = df.select("x").rdd.map(lambda row: row[0]).collect()
    y = df.select("y").rdd.map(lambda row: row[0]).collect()

    x = np.array(x).reshape(-1, X_dimension)
    y = np.array(y).reshape((-1,))

    df_spark = SparkSession.builder.appName('AA_Genome_Session').getOrCreate().createDataFrame(
        zip(x.tolist(), y.tolist()), ["x", "y"])

    aa_genome = AA_Genome(df_spark, dimensions=X_dimension)
    trained_model = aa_genome.train_AA_genome_model()

    # Save the model to a file if needed
    # For example, save the model in the 'adapt_backend' directory
    trained_model.save('/Users/derrickbass/Public/adaptai/src/adapt_backend/psych_aa_genome_model')


if __name__ == '__main__':
    main()
