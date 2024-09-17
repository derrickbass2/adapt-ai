from typing import Optional, Dict

from pyspark.sql import DataFrame
from tensorflow import keras


class NeuroTechNetwork:
    def __init__(self):
        self.model = None

    def train_NTN_model(self, df: DataFrame, **kwargs) -> Optional[str]:
        """
        Trains the NeuroTech Network model using the input dataframe.

        Parameters:
            df (pyspark.sql.DataFrame): Input dataframe.
            kwargs (Dict): Arbitrary keyword arguments.

        Returns:
            Optional[str]: Serialized model or None if failed.
        """
        try:
            # Convert Spark DataFrame to Pandas DataFrame for TensorFlow compatibility
            pandas_df = df.toPandas()
            X = pandas_df.iloc[:, :-1].values
            y = pandas_df.iloc[:, -1].values

            # Define the model
            model = keras.Sequential([
                keras.layers.Dense(units=64, input_shape=(X.shape[1],), activation='relu'),
                keras.layers.Dense(units=32, activation='relu'),
                keras.layers.Dense(units=1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X, y, epochs=10)

            # Save and serialize the model
            self.model = model
            return self.serialize_model()
        except Exception as e:
            print(f"Error training model: {e}")
            return None

    def test_NTN_model(self, model_str: str, df: DataFrame, **kwargs) -> Optional[float]:
        """
        Tests the NeuroTech Network model using input dataframe and serialized model.

        Parameters:
            model_str (str): Serialized model.
            df (pyspark.sql.DataFrame): Input dataframe.
            kwargs (Dict): Arbitrary keyword arguments.

        Returns:
            Optional[float]: Metric score or None if failed.
        """
        try:
            # Deserialize the model
            self.deserialize_model(model_str)

            # Convert Spark DataFrame to Pandas DataFrame for TensorFlow compatibility
            pandas_df = df.toPandas()
            X = pandas_df.iloc[:, :-1].values
            y = pandas_df.iloc[:, -1].values

            # Evaluate the model
            loss = self.model.evaluate(X, y)
            return loss
        except Exception as e:
            print(f"Error testing model: {e}")
            return None

    def serialize_model(self) -> str:
        """
        Serializes the trained model.

        Returns:
            str: Serialized model string.
        """
        if self.model is not None:
            import io
            import base64

            # Save the model to a bytes buffer
            buffer = io.BytesIO()
            self.model.save_weights(buffer)
            model_bytes = buffer.getvalue()

            # Encode model bytes to base64 string
            model_str = base64.b64encode(model_bytes).decode('utf-8')
            return model_str
        return ""

    def deserialize_model(self, model_str: str):
        """
        Deserializes a model string back into a model.

        Parameters:
            model_str (str): The model string to deserialize.
        """
        import io
        import base64

        if model_str:
            # Decode base64 string to bytes
            model_bytes = base64.b64decode(model_str)

            # Load the model from bytes buffer
            buffer = io.BytesIO(model_bytes)
            self.model = keras.Sequential()
            self.model.load_weights(buffer)


# Example of usage
def example_usage():
    # Initialize Spark DataFrame (example)
    # df = ...  # Load or create your DataFrame here

    neurotech_network = NeuroTechNetwork()

    # Example training
    model_str = neurotech_network.train_NTN_model(df)

    # Example testing
    test_loss = neurotech_network.test_NTN_model(model_str, df)
    print(f"Test Loss: {test_loss}")


if __name__ == "__main__":
    example_usage()
