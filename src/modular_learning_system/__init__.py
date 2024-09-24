# /Users/derrickbass/Public/adaptai/src/modular_learning_system/__init__.py

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split


class SparkEngine:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("Modular Learning System") \
            .getOrCreate()

    def read_csv(self, file_path: str) -> pd.DataFrame:
        """
        Read a CSV file into a Pandas DataFrame.
        """
        return pd.read_csv(file_path)

    def write_parquet(self, processed_df: pd.DataFrame, output_path: str):
        """
        Write a Pandas DataFrame to a Parquet file.
        """
        spark_df = self.spark.createDataFrame(processed_df)
        spark_df.write.parquet(output_path)

    def cluster_data(self, df_preprocessed: pd.DataFrame, num_clusters: int) -> np.ndarray:
        """
        Cluster data using KMeans and return the cluster labels.
        """
        spark_df = self.spark.createDataFrame(df_preprocessed)
        assembler = VectorAssembler(inputCols=spark_df.columns, outputCol="features")
        transformed_data = assembler.transform(spark_df)

        kmeans = KMeans(k=num_clusters, seed=1)
        model = kmeans.fit(transformed_data)
        return model.transform(transformed_data).select("prediction").toPandas().to_numpy().flatten()

    def preprocess_data(self, df: pd.DataFrame, param: str, label_col: str) -> pd.DataFrame:
        """
        Preprocess the DataFrame by handling missing values and encoding categorical features.
        """
        df = df.dropna()
        if param in df.columns:
            df[param] = pd.get_dummies(df[param])
        return df

    def predict(self, model, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the provided model.
        """
        spark_df = self.spark.createDataFrame(df)
        predictions = model.transform(spark_df)
        return predictions.select("prediction").toPandas().to_numpy().flatten()

    def train_model(self, df_preprocessed: pd.DataFrame, label_col: str) -> Pipeline:
        """
        Train a model using the preprocessed DataFrame and return the trained model.
        """
        train_df, test_df = train_test_split(df_preprocessed, test_size=0.2, random_state=42)
        assembler = VectorAssembler(inputCols=train_df.columns.drop(label_col).tolist(), outputCol="features")
        pipeline = Pipeline(stages=[assembler, KMeans(k=3)])  # Example model; replace with actual
        pipeline_model = pipeline.fit(train_df)
        return pipeline_model
