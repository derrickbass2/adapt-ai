import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .appName("Datasets Training and Testing") \
        .master("local[*]") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    yield spark

    spark.stop()