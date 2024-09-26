import argparse

import numpy as np
from pyspark.sql import SparkSession

from aa_genome import AA_Genome  # Adjusted import path to match project structure


def generate_synthetic_data(rows: int, dimensions: int):
    """
    Generate synthetic data for testing purposes.
    :param rows: Number of rows to generate
    :param dimensions: Number of dimensions (features) for the input data
    :return: Tuple of numpy arrays (x, y)
    """
    x = np.random.rand(rows, dimensions)
    y = np.sum(x, axis=1) + np.random.normal(scale=0.1, size=rows)  # Target is a noisy sum of input
    return x, y


def parse_arguments():
    """
    Parse command-line arguments.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_value", type=float, default=5.0, help="Target value for the model")
    parser.add_argument("--dimensions", type=int, default=10, help="Number of dimensions for the input data")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size for the genetic algorithm")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations to train")
    parser.add_argument("--mutation_rate", type=float, default=0.01, help="Mutation rate for the algorithm")
    return parser.parse_args()


def main(args):
    # Generate synthetic data
    x, y = generate_synthetic_data(1000, args.dimensions)

    # Initialize Spark session
    spark = SparkSession.builder.appName('AA_Genome_Session').getOrCreate()

    # Create Spark DataFrame
    df = spark.createDataFrame(zip(x.tolist(), y.tolist()), ["x", "y"])

    # Initialize and train the AA_Genome model
    aa_genome = AA_Genome(
        data=df,
        dimensions=args.dimensions,
        target_value=args.target_value,
        pop_size=args.pop_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate
    )

    # Train the model
    aa_genome.train_AA_genome_model()

    # Retrieve and print the best solution
    best_solution = aa_genome.get_best_solution()
    print(f'Best Solution: {best_solution}')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
