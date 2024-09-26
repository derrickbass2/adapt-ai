# File: /Users/derrickbass/Public/adaptai/src/modular_learning_system/aa_genome/data_loader.py

from pyspark.sql import SparkSession


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Load data from CSV and return a Spark DataFrame.
        :return: Spark DataFrame
        """
        spark = SparkSession.builder.appName("AA_Genome").getOrCreate()
        return spark.read.csv(self.file_path, header=True, inferSchema=True)
