import pytest
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from main import train_hospitality_model, test_hospitality_model
from pyspark.sql import SparkSession
import pytest
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from main import train_hospitality_model, test_hospitality_model
from pyspark.ml.clustering import KMeansModel

@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .appName("Datasets Training and Testing") \
        .master("local[*]") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    yield spark

    spark.stop()

def test_train_hospitality_model(spark_session):
    # Create test DataFrame
    df = spark_session.createDataFrame([(25, 50000), (30, 60000), (35, 70000)], ["Age", "Salary"])

    # Call the function
    result = train_hospitality_model(df)

    # Assert the result
    assert isinstance(result, KMeansModel)











