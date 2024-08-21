from typing import Any

# Import key classes or functions from main.py and mnist.py
from .main import NeuroTechNetwork
from .mnist import MNISTClassifier

__all__ = [
    'NeuroTechNetwork',
    'MNISTClassifier'
]


def initialize_neurotech_network(params: dict) -> Any:
    """
    Initialize the NeuroTech Network with provided parameters.

    :param params: Dictionary of parameters for initialization
    :return: Initialized NeuroTechNetwork instance
    """
    try:
        return NeuroTechNetwork(**params)
    except Exception as e:
        print("Error initializing NeuroTech Network: {e}")
        return None


def load_mnist_model(model_path: str) -> Any:
    """
    Load a pre-trained MNIST model from the specified path.

    :param model_path: Path to the saved MNIST model
    :return: Loaded MNIST model
    """
    try:
        classifier = MNISTClassifier()
        classifier.load_model(model_path)
        return classifier
    except Exception as e:
        print("Error loading MNIST model: {e}")
        return None
