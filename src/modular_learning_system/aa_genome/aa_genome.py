import numpy as np
from pyspark.sql import DataFrame


class AA_Genome:
    def __init__(self, data: DataFrame, dimensions: int, target_value: float, pop_size: int = 100,
                 generations: int = 1000, mutation_rate: float = 0.01):
        """
        Initialize the AA_Genome class.

        :param data: Spark DataFrame containing the data
        :param dimensions: Number of dimensions for input data
        :param target_value: The target value the model is attempting to reach
        :param pop_size: Population size for the genetic algorithm
        :param generations: Number of generations for the algorithm
        :param mutation_rate: Mutation rate for evolving the population
        """
        self.data = data
        self.dimensions = dimensions
        self.target_value = target_value
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self._initialize_population()

    def _initialize_population(self):
        """
        Initialize the population for the genetic algorithm.
        """
        # Example: Randomly generate initial population
        return np.random.rand(self.pop_size, self.dimensions)

    def fitness(self, individual):
        """
        Calculate the fitness of an individual.
        This could be based on how close the individual is to the target_value.
        """
        # Example fitness function: difference from target_value
        prediction = np.sum(individual)  # You can customize this
        return abs(self.target_value - prediction)

    def evolve(self):
        """
        Evolve the population over generations.
        """
        for generation in range(self.generations):
            fitness_scores = np.array([self.fitness(ind) for ind in self.population])

            # Select individuals based on their fitness
            selected = self.selection(fitness_scores)

            # Perform crossover and mutation to create new population
            self.population = self.crossover_and_mutation(selected)

            print(f"Generation {generation}: Best fitness {min(fitness_scores)}")

    def selection(self, fitness_scores):
        """
        Select individuals based on their fitness scores.
        """
        # Example: Select the best half of the population
        sorted_indices = np.argsort(fitness_scores)
        return self.population[sorted_indices[:self.pop_size // 2]]

    def crossover_and_mutation(self, selected):
        """
        Perform crossover and mutation to create the next generation.
        """
        # Example: Combine and mutate selected individuals
        next_generation = []
        for i in range(self.pop_size):
            parent1 = selected[np.random.randint(len(selected))]
            parent2 = selected[np.random.randint(len(selected))]

            # Crossover (combine parents)
            child = (parent1 + parent2) / 2

            # Mutation
            mutation = np.random.rand(self.dimensions) < self.mutation_rate
            child[mutation] = np.random.rand(np.sum(mutation))

            next_generation.append(child)
        return np.array(next_generation)

    def train_AA_genome_model(self):
        """
        Train the genetic algorithm.
        """
        self.evolve()
        print("Training completed")
        return self

    def get_best_solution(self):
        """
        Get the best solution from the population.
        """
        fitness_scores = np.array([self.fitness(ind) for ind in self.population])
        best_index = np.argmin(fitness_scores)
        return self.population[best_index]
