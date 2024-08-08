import fnmatch
import os

import numpy as np
from tensorflow import keras


def load_data(file_pattern):
    data = []
    for file in os.listdir('/Users/derrickbass/Public/adaptai/datasets'):
        if fnmatch.fnmatch(file, file_pattern):
            raw_data = np.load('/Users/derrickbass/P/adaptai/datasets/' + file)
            data.extend(raw_data)
    return np.array(data)


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(units=64),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dense(units=32),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dense(units=1)
    ])
    return model


def main():
    X_dimension = 10
    X = load_data('*.npz')[:, :X_dimension]
    y = load_data('*.npz')[:, X_dimension:].reshape((-1,))

    model = create_model()

    model.compile(optimizer='adam', loss='mean_absolute_error').fit(X, y, epochs=10)

    model.save("/Users/derrickbass/Public/adaptai/mls/aa_genome/genetic_algorithm_module")


if __name__ == '__main__':
    main()
