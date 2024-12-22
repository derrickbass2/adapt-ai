import os

import numpy as np
import pandas as pd
from joblib import dump, load
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression


# ---------------------------------------------
# Genetic Algorithm Class (AA_Genome)
# ---------------------------------------------
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
        return np.random.rand(self.pop_size, self.dimensions)

    def _evaluate_fitness(self, individual, features=None, labels=None):
        """
        Use input data for fitness (if provided); otherwise, use target value.
        """
        if features is not None and labels is not None:
            predictions = np.dot(features, individual)
            loss = np.mean((predictions - labels) ** 2)  # Mean Squared Error
            return loss
        return np.abs(np.sum(individual) - self.target_value)

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

    def train_AA_genome_model(self, features=None, labels=None):
        for generation in range(self.generations):
            parents = self._select_parents(features, labels)
            new_population = []

            for _ in range(self.pop_size):
                parent1, parent2 = np.random.permutation(parents)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            self.population = np.array(new_population)
            self.best_solution = self._select_parents(features, labels)[0]

    def get_best_solution(self):
        if self.best_solution is None:
            raise ValueError("Model not trained yet. Call 'train_AA_genome_model' first.")
        return self.best_solution


# ---------------------------------------------
# Keras Neural Network Model
# ---------------------------------------------
def create_neurotech_model(input_shape, is_classification=True, num_classes=2, hidden_layers=None):
    if not isinstance(input_shape, tuple) or len(input_shape) != 1:
        raise ValueError("`input_shape` must be a tuple of size 1 (e.g., (n_features,))")
    if hidden_layers is None:
        hidden_layers = [64, 32]

    model = Sequential()
    model.add(Dense(hidden_layers[0], input_shape=input_shape, activation="relu"))
    for layer_size in hidden_layers[1:]:
        model.add(Dense(layer_size, activation="relu"))

    if is_classification:
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


# ---------------------------------------------
# SparkEngine Class
# ---------------------------------------------
class SparkEngine:
    @staticmethod
    def read_csv(file_path):
        """
        Reads a CSV file into a DataFrame.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    @staticmethod
    def preprocess_data(df, feature_cols, label_col):
        """
        Preprocesses the input data for clustering or training.
        """
        if not all(col in df.columns for col in feature_cols):
            raise ValueError("Some feature columns are missing in the DataFrame.")
        if label_col and label_col not in df.columns:
            raise ValueError("The label column is missing in the DataFrame.")
        df = df.dropna()
        X = df[feature_cols]
        y = df[label_col] if label_col else None
        return X, y

    @staticmethod
    def cluster_data(df, num_clusters):
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if not numeric_cols.any():
            raise ValueError("No numeric columns found for clustering.")
        kmeans = KMeans(n_clusters=num_clusters)
        df["cluster"] = kmeans.fit_predict(df[numeric_cols])
        return df

    @staticmethod
    def train_model(X, y, is_classification=True):
        """
        Trains a Logistic Regression or Linear Regression model.
        """
        if is_classification:
            model = LogisticRegression()
        else:
            model = LinearRegression()
        model.fit(X, y)
        return model

    @staticmethod
    def predict(X, model_path):
        """
        Loads a pre-trained model and predicts results.
        """
        ext = os.path.splitext(model_path)[1]
        if ext == ".joblib":
            model = load(model_path)
        elif ext == ".h5":
            model = load_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {ext}")
        return model.predict(X)

    @staticmethod
    def save_model(model, model_path):
        """
        Saves a given model to the specified path.
        """
        ext = os.path.splitext(model_path)[1]
        if ext == ".joblib":
            dump(model, model_path)
        elif ext == ".h5":
            model.save(model_path)
        else:
            raise ValueError("Unsupported file extension for save_model()")
