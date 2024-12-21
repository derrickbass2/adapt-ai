import os

from tensorflow.python.keras import Dense


class NeurotechNetwork:
    def __init__(self):
        self.model_path = "/Users/dbass/PycharmProjects/adapt-ai-real/neurotech_model.h5"

    def train_model(self, df):
        # Splitting features and target
        features = df.select("feature1", "feature2").toPandas()
        target = df.select("target").toPandas()

        # Define a simple sequential model
        from tensorflow.python.keras import Sequential
        model = Sequential([
            Dense(),
            Dense(),
            Dense()
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(features, target, epochs=10, batch_size=2, verbose=0)

        # Save the model
        model.save(self.model_path)

        return self.model_path

    def evaluate_model(self, df):
        # Load the trained model
        model = self.load_model()

        # Splitting features and target
        features = df.select("feature1", "feature2").toPandas()
        target = df.select("target").toPandas()

        # Evaluate the model
        loss, accuracy = model.evaluate(features.values, target.values, verbose=0)

        return accuracy

    def load_model(self):
        # Load the model from the saved path
        if os.path.isfile(self.model_path):
            from tensorflow.python import keras
            return keras.models.load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
