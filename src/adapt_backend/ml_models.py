import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential


# Genetic Algorithm Class
class AA_Genome:
    def __init__(self, dimensions=10, target_value=0.5, pop_size=100, generations=1000, mutation_rate=0.01):
        """
        Genetic Algorithm initialization.
        """
        self.dimensions = dimensions
        self.target_value = target_value
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self._initialize_population()
        self.best_solution = None

    def _initialize_population(self):
        if self.pop_size <= 0 or self.dimensions <= 0:
            raise ValueError("Population size and dimensions must be positive integers.")
        return np.random.rand(self.pop_size, self.dimensions)

    def _evaluate_fitness(self, individual, features=None, labels=None):
        try:
            if features is not None and labels is not None:
                predictions = np.dot(features, individual)
                loss = np.mean((predictions - labels) ** 2)
                return loss  # Mean Squared Error
            return np.abs(np.sum(individual) - self.target_value)
        except Exception as e:
            raise RuntimeError(f"Error in fitness evaluation: {str(e)}")

    def _select_parents(self, features=None, labels=None):
        fitness_scores = np.array([self._evaluate_fitness(ind, features, labels) for ind in self.population])
        parent_indices = np.argsort(fitness_scores)[:2]
        return self.population[parent_indices]

    def _crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dimensions - 1)
        return np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

    def _mutate(self, child):
        if np.random.rand() < self.mutation_rate:
            mutation_index = np.random.randint(0, self.dimensions)
            child[mutation_index] = np.random.rand()
        return child

    def train(self, features=None, labels=None):
        """
        Run the genetic algorithm optimization process.
        """
        for generation in range(self.generations):
            parents = self._select_parents(features, labels)
            new_population = [self._mutate(self._crossover(parents[0], parents[1])) for _ in range(self.pop_size)]
            self.population = np.array(new_population)
            self.best_solution = self._select_parents(features, labels)[0]

    def get_best_solution(self):
        if self.best_solution is None:
            raise ValueError("Model not trained yet. Call 'train' first.")
        return self.best_solution


# Spark Engine Class
class Spark_Engine:
    def __init__(self, data):
        """
        A mock Spark-like data processing engine. Can be extended to use PySpark if needed.
        :param data: Input data to be processed.
        """
        self.data = data

    def map(self, func):
        """
        Mimic the Spark 'map' functionality.
        """
        self.data = np.array([func(x) for x in self.data])
        return self

    def filter(self, func):
        """
        Mimic the Spark 'filter' functionality.
        """
        self.data = np.array([x for x in self.data if func(x)])
        return self

    def collect(self):
        """
        Mimic the Spark 'collect' functionality.
        """
        return self.data

    def reduce(self):
        """
        Mimic the Spark 'reduce' functionality.
        """
        return np.array(self.data).flatten().tolist()

    def __repr__(self):
        return f"<Spark_Engine data={self.data}>"


# Neural Network Class
class Neurotech_Network:
    __slots__ = ('is_classification', 'num_classes', 'hidden_layers', 'learning_rate', 'model')

    def __init__(self, input_shape, is_classification=True, num_classes=2, hidden_layers=None, learning_rate=0.001):
        """
        Initializes a neural network model using TensorFlow/Keras.
        :param input_shape: Input shape (number of features).
        :param is_classification: If True, configure for classification; otherwise, regression.
        :param num_classes: Number of classes for classification.
        :param hidden_layers: List of hidden layer sizes.
        :param learning_rate: Learning rate for optimization.
        """
        if not isinstance(input_shape, tuple) or len(input_shape) != 1:
            raise ValueError("Input shape must be a tuple of size 1 (e.g., `(n_features,)`).")
        if hidden_layers is None:
            hidden_layers = [64, 32]

        self.is_classification = is_classification
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        """
        Build the Sequential Keras model.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))  # Use explicit Input layer
        for layer_size in self.hidden_layers:
            model.add(Dense(layer_size, activation="relu"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if self.is_classification:
            model.add(Dense(self.num_classes, activation="softmax"))
            model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def train(self, features, labels, epochs=10, batch_size=32):
        """
        Train the neural network model.
        :param features: Training features.
        :param labels: Training labels.
        :param epochs: Number of epochs to train for.
        :param batch_size: Batch size.
        """
        if features is None or labels is None:
            raise ValueError("Features and labels must not be None.")
        self.model.fit(features, labels, epochs=epochs, batch_size=batch_size)

    def predict(self, features):
        """
        Predict using the trained model.
        :param features: Input features for prediction.
        """
        return self.model.predict(features)

    def evaluate(self, features, labels):
        """
        Evaluate the model on a test set.
        :param features: Test features.
        :param labels: Test labels.
        """
        return self.model.evaluate(features, labels)


# Example usage (you can remove this from your production codebase)
if __name__ == "__main__":
    # Example for create_neurotech_model usage
    neurotech = create_neurotech_model(input_shape=(10,), num_classes=3, hidden_layers=[128, 64])
    print("Neurotech model created successfully")
    neurotech.model.summary()
    aa.train()
    print("Best solution:", aa.get_best_solution())

    # Example for Spark_Engine
    spark = Spark_Engine(data=np.array([1, 2, 3, 4, 5]))
    print("Filtered data:", spark.filter(lambda x: x % 2 == 0).collect())

    # Example for Neurotech_Network
    # Removed standalone usage of Neurotech_Network direct instantiation
    nn.model.summary()


def create_neurotech_model(input_shape=(10,), is_classification=True, num_classes=2, hidden_layers=None,
                           learning_rate=0.001):
    """
    Factory method to create and return a Neurotech_Network instance.
    :param input_shape: Input shape (number of features).
    :param is_classification: If True, configure for classification; otherwise, regression.
    :param num_classes: Number of classes for classification.
    :param hidden_layers: List of hidden layer sizes. Defaults to [64, 32].
    :param learning_rate: Learning rate for optimization.
    :return: Instance of Neurotech_Network.
    """
    try:
        return Neurotech_Network(
            input_shape=input_shape,
            is_classification=is_classification,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate
        )
    except Exception as e:
        raise RuntimeError(f"Error creating Neurotech_Network instance: {str(e)}")
