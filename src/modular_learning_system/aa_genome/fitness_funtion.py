# File: /Users/derrickbass/Public/adaptai/src/modular_learning_system/aa_genome/fitness_function.py

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType


class FitnessFunction:
    def __init__(self, target_value):
        self.target_value = target_value

    def calculate_fitness(self, df):
        """
        Calculate the fitness of individuals using distributed Spark computation.
        :param df: Spark DataFrame
        :return: DataFrame with fitness scores
        """
        fitness_udf = F.udf(lambda ind: abs(self.target_value - sum(ind)), FloatType())
        return df.withColumn('fitness', fitness_udf(df['individual']))
