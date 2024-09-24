from flask import Flask, request, jsonify

from .mnist import MNISTClassifier
from .neurotech_network_script import NeurotechNetwork

app = Flask(__name__)

# Initialize model instances
mnist_classifier = MNISTClassifier()
neurotech_network = NeurotechNetwork(name="Neurotech Model", data_source="data_source")


@app.route('/train_mnist', methods=['POST'])
def train_mnist():
    """
    Endpoint to train the MNIST classifier.
    """
    try:
        data = request.json
        train_images = np.array(data['train_images'])
        train_labels = np.array(data['train_labels'])
        val_images = np.array(data['val_images'])
        val_labels = np.array(data['val_labels'])

        mnist_classifier.build_and_train((train_images, train_labels), (val_images, val_labels))
        return jsonify({"message": "MNIST model trained successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/predict_mnist', methods=['POST'])
def predict_mnist():
    """
    Endpoint to make predictions with the MNIST classifier.
    """
    try:
        data = request.json
        images = np.array(data['images'])
        predictions = mnist_classifier.predict(images)
        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/train_neurotech', methods=['POST'])
def train_neurotech():
    """
    Endpoint to train the Neurotech Network.
    """
    try:
        data = request.json
        data_files = neurotech_network.load_data(data['data_source'])
        neurotech_network.train_model(data_files)
        neurotech_network.save_model()
        return jsonify({"message": "Neurotech model trained and saved successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/evaluate_neurotech', methods=['POST'])
def evaluate_neurotech():
    """
    Endpoint to evaluate the Neurotech model.
    """
    try:
        data = request.json
        X_test = np.array(data['X_test'])
        y_test = np.array(data['y_test'])
        loss = neurotech_network.evaluate_model(X_test, y_test)
        return jsonify({"loss": loss}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
