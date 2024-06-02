import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import fnmatch

def load_data(file_pattern):
    data = []
    for file in os.listdir('/Users/derrickbass/Desktop/autonomod/datasets'):
        if fnmatch.fnmatch(file, file_pattern):
            raw_data = np.load('/Users/derrickbass/Desktop/autonomod/datasets/' + file)
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

def main():
    X_dimension = 10
    X = load_data('*.npz')[:, :X_dimension]
    y = load_data('*.npz')[:, X_dimension:].reshape((-1,))

    model = create_model(X_dimension)

    history = model.compile(optimizer='adam', loss='mean_absolute_error').fit(X, y, epochs=10)

    model.save('/Users/derrickbass/Desktop/autonomod/models/psych_aa_genome_model')

if __name__ == '__main__':
    main()