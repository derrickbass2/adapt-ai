from typing import Any

# Import key classes or functions from main.py and mnist.py
from .main import NeurotechNetwork
from .mnist import MNISTClassifier

__all__ = [
    'NeurotechNetwork',
    'MNISTClassifier'
]

def initialize_neurotech_network(params: dict) -> Any:
    """
    Initialize the Neurotech Network with provided parameters.
    
    :param params: Dictionary of parameters for initialization
    :return: Initialized NeurotechNetwork instance
    """
    return NeurotechNetwork(**params)

def load_mnist_model(model_path: str) -> Any:
    """
    Load a pre-trained MNIST model from the specified path.
    
    :param model_path: Path to the saved MNIST model
    :return: Loaded MNIST model
    """
    classifier = MNISTClassifier()
    classifier.load_model(model_path)
    return classifier