import fnmatch
import os

import numpy as np
from tensorflow import keras

DATAPATH = '/Users/derrickbass/Desktop/Autonomod/datasets/'
MODELPATH = '/Users/derrickbass/Desktop/Autonomod/models/'


def load_data(file_pattern):
    data = []
    for file in os.listdir(DATAPATH):
        if fnmatch.fnmatch(file, file_pattern):
            raw_data = np.load(DATAPATH + file)
            data.extend(raw_data)
    return np.array(data)


def create_model(X_dimension):
    model = keras.Sequential([
        keras.layers.Dense(units=64, input_shape=(X_dimension,)),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dense(units=32),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dense(units=1)
    ])
    return model


def main(X_dimension):
    X = load_data('*.npz')[:, :X_dimension]
    y = load_data('*.npz')[:, X_dimension:].reshape((-1,))

    model = create_model(X_dimension)

    history = model.compile(optimizer='adam', loss='mean_absolute_error').fit(X, y, epochs=10)

    accuracy = model.evaluate(X, y)

    model.save(f"{MODELPATH}retail_neurotech_network_model")


if __name__ == '__main__':
    X_dimension = 10
    main(X_dimension)  # mls/neurotech_network/neurotech_network_script.py


def run():
    # Your existing code here
    return "Neurotech Network result"


if __name__ == "__main__":
    print(run())
