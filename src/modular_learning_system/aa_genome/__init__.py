import os
import pickle
import random
from typing import List, Optional, Any

import numpy as np
import tensorflow as tf
from pyspark.sql import DataFrame as SparkDataFrame

from modular_learning_system.aa_genome.aa_genome import AA_Genome  # Corrected class name

__all__ = ['AA_Genome']


class AA_Genome:
    def __init__(self, df: SparkDataFrame, **kwargs):
        self.df = df
        self.params = kwargs

    @staticmethod
    def _convert_to_numpy(df: SparkDataFrame, column_names: List[str]) -> np.ndarray:
        """
        Convert specified columns of a Spark DataFrame to a NumPy array.

        :param df: Spark DataFrame
        :param column_names: List of column names to be converted
        :return: NumPy array with the data from specified columns
        """
        selected_cols = [c for c in column_names if c in df.columns]
        arr = df.select(*selected_cols).rdd.map(tuple).collect()
        return np.vstack(arr)

    @staticmethod
    def mate(parent1: dict, parent2: dict) -> dict:
        """Mate two parent genomes to produce offspring."""
        return {key: parent1[key] if random.random() > 0.5 else parent2[key] for key in parent1.keys()}

    def mutate(self, genome: dict, mutation_rate: float = 0.01) -> dict:
        """Mutate the genome by randomly altering its traits."""
        for key in genome.keys():
            if random.random() < mutation_rate:
                genome[key] = random.choice(self.possible_values_for_trait())
        return genome

    @staticmethod
    def possible_values_for_trait() -> list:
        """Define possible values for a given trait."""
        return [0, 1, 2]  # Placeholder

    @staticmethod
    def random_genome() -> dict:
        """Generate a random genome."""
        return {'trait1': random.random(), 'trait2': random.random()}  # Placeholder

    def natural_selection(self, population: list) -> list:
        """Perform natural selection to choose the fittest individuals."""
        population.sort(key=lambda genome: self.fitness(genome))
        return population[:len(population) // 2]

    @staticmethod
    def fitness(genome: dict) -> float:
        """Calculate the fitness of a genome."""
        return sum(genome.values())  # Placeholder

    def evolve_population(self, population: list) -> list:
        """Evolve the population using genetic algorithms."""
        selected_population = self.natural_selection(population)
        new_population = []
        for _ in range(len(selected_population) // 2):
            parent1, parent2 = random.sample(selected_population, 2)
            offspring1 = self.mate(parent1, parent2)
            offspring2 = self.mate(parent2, parent1)
            new_population.extend([self.mutate(offspring1), self.mutate(offspring2)])
        return new_population

    def train_AA_genome_model(self) -> Optional[Any]:
        """
        Train the AA genome model using the provided parameters.

        :param kwargs: Additional parameters for training
        :return: Trained model or None
        """
        x = self._convert_to_numpy(self.df, self.params.get('x', []))
        y = self._convert_to_numpy(self.df, self.params.get('y', []))

        population = [self.random_genome() for _ in range(self.params.get('pop_size', 100))]
        for generation in range(self.params.get('generations', 1000)):
            population = self.evolve_population(population)

        model = self.create_model(x.shape[1])
        model.fit(x, y, epochs=10)

        # Ensure MODELPATH directory exists
        os.makedirs(self.params.get('model_path', 'model_path'), exist_ok=True)

        # Save model
        model_save_path = os.path.join(self.params.get('model_path', 'model_path'), 'aa_genome_model')
        try:
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            return None

        return model

    @staticmethod
    def create_model(input_dim: int) -> Any:
        """
        Create and compile a neural network model.

        :param input_dim: Number of input features
        :return: Compiled Keras model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, input_shape=(input_dim,)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(units=32),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_absolute_error')
        return model

    def save(self, path: str):
        """Serialize the AA_Genome model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'AA_Genome':
        """Deserialize the AA_Genome model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
