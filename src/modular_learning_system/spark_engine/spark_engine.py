import os

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame


class SparkEngineUtils:
    def __init__(self, spark):
        self.spark = spark

    def advanced_cleaning(self, df: DataFrame) -> DataFrame:
        """
        Perform advanced cleaning and transformations, such as missing value handling,
        outlier removal, or complex filtering.
        """
        print("Performing advanced cleaning...")
        cleaned_df = df.dropna()  # Example of dropping rows with missing values
        # Add more transformations as needed
        return cleaned_df

    def transform_data(self, df: DataFrame, feature_cols: list, label_col: str = None) -> DataFrame:
        """
        Assemble feature vectors for modeling or clustering.
        """
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        transformed_df = assembler.transform(df)

        if label_col:
            transformed_df = transformed_df.select("features", label_col)
        else:
            transformed_df = transformed_df.select("features")

        return transformed_df

    def run_kmeans(self, df: DataFrame, feature_cols: list, k: int = 3):
        """
        Perform KMeans clustering on the given DataFrame.
        """
        transformed_df = self.transform_data(df, feature_cols)
        kmeans = KMeans(k=k)
        model = kmeans.fit(transformed_df)
        predictions = model.transform(transformed_df)
        return predictions.select('features', 'prediction')

    def run_linear_regression(self, df: DataFrame, feature_cols: list, label_col: str):
        """
        Perform Linear Regression on the given DataFrame.
        """
        transformed_df = self.transform_data(df, feature_cols, label_col)
        lr = LinearRegression(featuresCol='features', labelCol=label_col)
        lr_model = lr.fit(transformed_df)
        predictions = lr_model.transform(transformed_df)
        return predictions.select("features", "prediction", label_col)

    def write_output(self, df: DataFrame, output_path: str):
        """
        Write DataFrame to the specified output path in Parquet format.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df.write.parquet(output_path)
        print(f"Data saved to {output_path}")
