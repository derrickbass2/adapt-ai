import fnmatch
import os
import numpy as np
from tensorflow import keras

DATAPATH = '/Users/derrickbass/Public/adaptai/datasets/'
MODELPATH = '/Users/derrickbass/Public/adaptai/mls/neurotech_network/'


def load_data(file_pattern):
    data = []
    for file in os.listdir(DATAPATH):
        if fnmatch.fnmatch(file, file_pattern):
            raw_data = np.load(DATAPATH + file)
            data.extend(raw_data)
    return np.array(data)


def create_model(input_dim):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(units=64),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dense(units=32),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dense(units=1)
    ])
    return model


def main():
    x_dimension = 10  # Set this here to avoid shadowing
    data = load_data('*.npz')
    X = data[:, :x_dimension]
    y = data[:, x_dimension:].reshape((-1,))

    model = create_model(X.shape[1])

    # Compile and fit the model
    model.compile(optimizer='adam', loss='mean_absolute_error')
    history = model.fit(X, y, epochs=10)

    # Evaluate the model
    accuracy = model.evaluate(X, y)

    # Save the model
    model.save(f"{MODELPATH}retail_neurotech_network_model")

    # Use history and accuracy
    print("Training history:", history.history)
    print("Model accuracy:", accuracy)


if __name__ == '__main__':
    main()
