import logging
import os

from pyspark.sql import DataFrame
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model

logging.basicConfig(level=logging.INFO)


class NeurotechNetwork:
    def __init__(self, model_path: str = "neurotech_model.h5"):
        self.model_path: str = model_path
        logging.info("Initialized NeurotechNetwork with model path: %s", self.model_path)

    def prepare_data(self, df: DataFrame) -> tuple:
        features = df.select("feature1", "feature2").toPandas()
        target = df.select("target").toPandas()
        return features, target

    def build_model(self) -> Sequential:
        model = Sequential([
            Dense(32, activation='relu', input_dim=2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, df: DataFrame) -> str:
        logging.info("Training model...")
        features, target = self.prepare_data(df)
        model = self.build_model()
        model.fit(features, target, epochs=10, batch_size=2, verbose=1)
        model.save(self.model_path)
        logging.info("Model saved to %s", self.model_path)
        return self.model_path

    def evaluate_model(self, df: DataFrame) -> float:
        model = self.load_model()
        features, target = self.prepare_data(df)
        loss, accuracy = model.evaluate(features.values, target.values, verbose=0)
        logging.info("Model evaluated with accuracy: %.2f", accuracy)
        return accuracy

    def load_model(self):
        if os.path.isfile(self.model_path):
            return load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
