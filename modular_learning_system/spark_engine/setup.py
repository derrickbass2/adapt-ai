from pyspark.sql import SparkSession

class SparkEngine:
    def __init__(self):
        self.spark = SparkSession.builder.appName('NOMAD').getOrCreate()

    def read_csv(self, path):
        return self.spark.read.option('header', 'true').csv(path)

    def write_parquet(self, df, path):
        df.write.parquet(path)

    def __del__(self):
        self.spark.stop()