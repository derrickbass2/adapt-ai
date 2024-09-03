from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession


class SparkEngine:
    def __init__(self, spark_context):
        self.spark_context = spark_context
        self.spark_session = SparkSession.builder.getOrCreate()

    def read_csv(self, file_path):
        df = self.spark_session.read.csv(file_path, header=True, inferSchema=True)
        return df

    def preprocess_data(self, df, feature_cols):
        df_preprocessed = df.select(*feature_cols)
        return df_preprocessed

    def cluster_data(self, df_preprocessed, num_clusters):
        kmeans = KMeans(featuresCol="features", k=num_clusters, seed=1)
        model = kmeans.fit(df_preprocessed)
        df_clustered = model.transform(df_preprocessed)
        return df_clustered

    def write_parquet(self, df_clustered, output_path):
        df_clustered.write.parquet(output_path)
