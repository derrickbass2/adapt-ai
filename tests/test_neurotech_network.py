import pytest
from neurotech_network import NeurotechNetwork
class MyClass:
    def __init__(self, name, age):
        self.name = name
        self.age = age

def test_get_neurotech_session():
    engine = NeurotechNetwork("test-nomad", MyClass)
    result = engine.get_neurotech_session()
    assert isinstance(result, NeurotechNetwork)
