import base64
import io
from typing import Optional

import tensorflow as tf
from pyspark.sql import DataFrame
from tensorflow.keras import models


class NeuroTechNetwork:
    def __init__(self):
        self.model = None

    def train_NTN_model(self, df: DataFrame, **kwargs) -> Optional[str]:
        try:
            pandas_df = df.toPandas()
            X = pandas_df.iloc[:, :-1].values
            y = pandas_df.iloc[:, -1].values

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=64, input_shape=(X.shape[1],), activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=10, batch_size=32, **kwargs)

            self.model = model
            return self.serialize_model()
        except Exception as e:
            print(f"Error training model: {e}")
            return None

    def test_NTN_model(self, model_str: str, df: DataFrame, **kwargs) -> Optional[float]:
        try:
            self.deserialize_model(model_str)
            pandas_df = df.toPandas()
            X = pandas_df.iloc[:, :-1].values
            y = pandas_df.iloc[:, -1].values

            loss = self.model.evaluate(X, y, **kwargs)
            return loss
        except Exception as e:
            print(f"Error testing model: {e}")
            return None

    def serialize_model(self) -> str:
        if self.model is not None:
            buffer = io.BytesIO()
            self.model.save_weights(buffer)
            model_bytes = buffer.getvalue()
            model_str = base64.b64encode(model_bytes).decode()
            return model_str
        return ""

    def deserialize_model(self, model_str: str):
        if model_str:
            model_bytes = base64.b64decode(model_str)
            buffer = io.BytesIO(model_bytes)
            buffer.seek(0)
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=64, input_shape=(None, 64), activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=1)
            ])
            self.model.load_weights(buffer)
