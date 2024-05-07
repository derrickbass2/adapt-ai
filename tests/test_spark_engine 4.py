import pytest
from spark_engine import SparkEngine
from pyspark.sql import SparkSession
class MyClass:
    def __init__(self, name, age):
        self.name = name
        self.age = age

def test_get_spark_session():
    engine = SparkEngine("test-nomad", MyClass)
    result = engine.get_spark_session()
    assert isinstance(result, SparkSession)

def test_close_spark_session():
    engine = SparkEngine("test-nomad", MyClass)
    engine.get_spark_session()
    engine.close_spark_session()

    # Check if the SparkSession has been closed
    with pytest.raises(Exception):
        engine.get_spark_session()