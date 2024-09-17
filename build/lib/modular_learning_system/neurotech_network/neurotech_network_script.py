import fnmatch
import os

import numpy as np
from tensorflow import keras

DATAPATH = '/Users/derrickbass/Public/adaptai/datasets/'
MODELPATH = '/Users/derrickbass/Public/adaptai/models/'


class NeurotechNetwork:
    def __init__(self, name, data_source):
        self.name = name
        self.data_source = data_source
        self.model = self.create_model(10)  # Assuming X_dimension is 10

    def create_model(self, X_dimension):
        model = keras.Sequential([
            keras.layers.Dense(units=64, input_shape=(X_dimension,)),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dense(units=32),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_absolute_error')
        return model

    def get_neurotech_session(self):
        # Dummy implementation - replace with actual session retrieval logic
        return self

    def predict(self, input_data):
        # Convert input data to numpy array if not already
        input_data = np.array(input_data)
        # Make prediction using the model
        return self.model.predict(input_data).tolist()


def load_data(file_pattern):
    data = []
    for file in os.listdir(DATAPATH):
        if fnmatch.fnmatch(file, file_pattern):
            raw_data = np.load(os.path.join(DATAPATH, file))
            data.extend(raw_data)
    return np.array(data)


def main(X_dimension):
    X = load_data('*.npz')[:, :X_dimension]
    y = load_data('*.npz')[:, X_dimension:].reshape((-1,))

    model = create_model(X_dimension)

    history = model.fit(X, y, epochs=10)

    # Evaluate model
    accuracy = model.evaluate(X, y)

    # Save model
    model.save(os.path.join(MODELPATH, 'retail_neurotech_network_model'))


if __name__ == '__main__':
    X_dimension = 10
    main(X_dimension)
