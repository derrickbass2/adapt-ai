import argparse
import numpy as np
from pyspark.sql import SparkSession
from adapt_backend.ml_models import AA_Genome  # Adjusted import path


def generate_synthetic_data(rows: int):
    """
    Generate synthetic data for testing purposes.

    :param rows: Number of rows to generate
    :return: Tuple of numpy arrays (x, y)
    """
    x = np.random.rand(rows, 10)
    y = np.sin(x[:, 0]) + np.random.normal(scale=0.1, size=rows)
    return x, y


def parse_arguments():
    """
    Parse command-line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_value", type=float, default=0.5, help="Target value for the model")
    parser.add_argument("--dimensions", type=int, default=10, help="Number of dimensions")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size for the algorithm")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations for training")
    parser.add_argument("--mutation_rate", type=float, default=0.01, help="Mutation rate for the algorithm")
    return parser.parse_args()


def main(args):
    # Generate synthetic data
    x, y = generate_synthetic_data(1000)

    # Initialize Spark session
    spark = SparkSession.builder.appName('AA_Genome_Session').getOrCreate()

    # Create Spark DataFrame
    df = spark.createDataFrame(zip(x.tolist(), y.tolist()), ["x", "y"])

    # Initialize and train the AA_Genome model
    aa_genome = AA_Genome(
        df,
        dimensions=args.dimensions,
        target_value=args.target_value,
        pop_size=args.pop_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate
    )

    trained_model = aa_genome.train_AA_genome_model()

    # Example: Retrieve the best solution if applicable
    best_solution = aa_genome.get_best_solution()  # Make sure this method exists in AA_Genome
    print(f'Best Solution: {best_solution}')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)