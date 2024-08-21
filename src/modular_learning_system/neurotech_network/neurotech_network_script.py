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
            data.append(raw_data)
    return np.array(data)

def create_model(X_dimension):
    model = keras.Sequential([
        keras.layers.Dense(units=64, input_shape=(X_dimension,)),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dense(units=32),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def main(X_dimension):
    X = load_data('*.npz')[:, :X_dimension]
    y = load_data('*.npz')[:, X_dimension:].reshape((-1,))

    model = create_model(X_dimension)
    model.fit(X, y, epochs=10)
    accuracy = model.evaluate(X, y)
    model.save(f"{MODELPATH}retail_neurotech_network_model.h5")

    return accuracy

def predict(input_data):
    """
    Predict the output using the trained Neurotech Network model.

    Parameters:
        input_data (np.array): Input data for prediction.

    Returns:
        Predicted output.
    """
    model = keras.models.load_model(f"{MODELPATH}retail_neurotech_network_model.h5")
    return model.predict(input_data)

if __name__ == '__main__':
    X_dimension = 10
    main(X_dimension)

def run():
    X_dimension = 10
    result = main(X_dimension)
    return f"Neurotech Network training completed with accuracy: {result}"

if __name__ == "__main__":
    print(run())