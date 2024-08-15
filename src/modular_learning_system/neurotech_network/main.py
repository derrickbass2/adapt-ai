from typing import Optional

import pyspark


class NeuroTechNetwork:
    def __init__(self):
        pass

    def train_NTN_model(self, df: pyspark.sql.DataFrame, **kwargs) -> Optional[str]:
        """
        Trains the Neuro Tech Network model using input dataframe.

        Parameters:
            df (pyspark.sql.DataFrame): Input dataframe.
            kwargs (Dict): Arbitrary keyword arguments.

        Returns:
            Optional[str]: Serialized model or None if failed.
        """
        raise NotImplementedError("Method not implemented.")

    def test_NTN_model(self, model: str, df: pyspark.sql.DataFrame, **kwargs) -> Optional[float]:
        """
        Tests the Neuro Tech Network model using input dataframe and serialized model.

        Parameters:
            model (str): Serialized model.
            df (pyspark.sql.DataFrame): Input dataframe.
            kwargs (Dict): Arbitrary keyword arguments.

        Returns:
            Optional[float]: Metric score or None if failed.
        """
        raise NotImplementedError("Method not implemented.")
