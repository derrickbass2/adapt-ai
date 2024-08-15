import os
import sys
from typing import List, Optional, Any

import numpy as np
from pyspark.sql import SparkSession

sys.path.append(os.path.dirname(
    __file__) + '/../genetic_algorithm')  # Adjust path relative to the aa_genome package directoryimport genetic_algorithm  # Import the genetic_algorithm module

spark = SparkSession.builder.appName('aa_genome').getOrCreate()


class AA_Genome:
    def __init__(self, df: SparkSession, **kwargs):
        self.df = df
        self.params = kwargs

    @staticmethod
    def _convert_to_numpy(df: SparkSession, column_names: List[str]) -> np.ndarray:
        selected_cols = [c for c in column_names if c in df.columns]
        arr = df.select(*selected_cols).rdd.map(tuple).collect()
        return np.vstack(arr)

    def train_AA_genome_model(self, **kwargs) -> Optional[Any]:
        # Convert DataFrame columns to NumPy array
        x = self._convert_to_numpy(self.df, self.params.get('x', []))
        y = self._convert_to_numpy(self.df, self.params.get('y', []))

        # Implement the logic for training the AA genome model using x and y
        # Return trained model or None
        raise NotImplementedError("Method not implemented.")

    def test_AA_genome_model(self, model: Any, **kwargs) -> Optional[float]:
        # Implement the logic for testing the AA genome model using input dataframe and serialized model
        # Return metric score or None
        raise NotImplementedError("Method not implemented.")
