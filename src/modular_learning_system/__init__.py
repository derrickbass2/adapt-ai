class SparkEngine:
    pass

    # def __init__(self, spark_context):
    #     self.spark_context = spark_context
    #     self.sql_context = SQLContext(spark_context)
    #     self.spark_session = SparkSession.builder.getOrCreate()
    #     self.spark = SparkSession.builder.getOrCreate()
    def read_csv(self, file_path):
        pass

    def preprocess_data(self, df, param, param1):
        pass

    def write_parquet(self, df_preprocessed, output_path):
        pass

    def cluster_data(self, df_preprocessed, num_clusters):
        pass
